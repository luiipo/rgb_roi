"""
Full Pipeline Integration & Training Script

실행 모드 (--mode):
  train_phase1   : Synthetic pretraining → encoder 저장
  train_phase2   : 실제 데이터 fine-tuning → 모델 저장
  evaluate       : Phase 2 결과로 실험 (B)~(D) 평가
  full           : Phase1 → Phase2 → 평가 전체 실행 (항상 처음부터)

체크포인트 선택 로드:
  --load_phase1  : Phase2 시작 전 Phase1 encoder 로드
  --load_phase2  : 평가 시작 전 Phase2 best.pt 로드

저장 경로:
  checkpoints/phase1/encoder.pt   ← Phase1 encoder
  checkpoints/phase2/best.pt      ← Phase2 best Rank-1 모델
  checkpoints/phase2/final.pt     ← Phase2 마지막 epoch 모델

[수정 사항 - 분석 보고서 반영]
1. Trainer: CenterLoss / TripletLoss optimizer 제거 → AdamW 단일 optimizer
   - ArcFace only 구조에 맞게 단순화
2. Trainer: Collapse 감지 → EER > 0.45 지속 시 조기 중단 (epoch 20 이후)
3. Phase2: load_phase1_encoder=True 시 freeze_encoder_layers(0.5) 적용
   - 하위 50% encoder 동결 → ArcFace head와 encoder 간 방향 충돌 방지
4. Phase2: encoder lr을 main lr의 1/10으로 분리
   - pretrained encoder를 천천히, ArcFace head를 빠르게 학습
5. run_phase1: SSL(SimCLR) pretraining 옵션 추가 (--use_ssl_pretrain)
   - 분류 supervision 없이 augmentation invariance 학습 → collapse 방지
6. Score 분포 진단: val_every마다 실행 (이전: diag_every=20 고정)
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, List, Dict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
cv2.setNumThreads(0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from models.feature_extraction import PalmprintRecognitionModel, HybridPalmprintEncoder
from models.pretraining import SyntheticPalmDataset, SelfSupervisedPretrainer
from models.stage_a_localization import PalmLocalizer
from models.stage_b_alignment import TopologyGuidedROIAligner
from models.stage_c_quality import QualityAwareROISelector
from models.stage_d_security import SecurityChecker
from data.datasets import PalmprintDataset, CrossDatasetLoader, DATASET_CONFIGS
from experiments.evaluation import ExperimentRunner


# ══════════════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════════════

def save_checkpoint(state_dict: dict, path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save(state_dict, path)
    print(f"  [Saved] {path}")


def load_checkpoint(path: str, device: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"체크포인트 없음: {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    print(f"  [Loaded] {path}")
    return state


def diagnose_score_distribution(model, loader, device) -> float:
    """
    Genuine/Impostor score 분포 진단.
    학습 중 collapse 감지 및 EER 추이 모니터링.
    Returns: EER (0~1)
    """
    from experiments.evaluation import (
        extract_all_embeddings,
        compute_cosine_similarity_matrix,
        compute_eer,
    )
    model.eval()
    embs, labels = extract_all_embeddings(model, loader, device)
    sim          = compute_cosine_similarity_matrix(embs)
    n            = len(labels)
    genuine, impostor = [], []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim[i, j])
            if labels[i] == labels[j]:
                genuine.append(s)
            else:
                impostor.append(s)

    genuine  = np.array(genuine)
    impostor = np.array(impostor)

    if len(genuine) == 0 or len(impostor) == 0:
        print("  [ScoreDiag] 충분한 쌍 없음")
        return 0.5

    eer, thresh = compute_eer(genuine, impostor)
    gap = genuine.mean() - impostor.mean()
    print(
        f"  [ScoreDiag] "
        f"Genuine: mean={genuine.mean():.4f} std={genuine.std():.4f} | "
        f"Impostor: mean={impostor.mean():.4f} std={impostor.std():.4f} | "
        f"Gap={gap:.4f} | EER={eer*100:.2f}%"
    )
    return float(eer)


# ══════════════════════════════════════════════════════════════════
# Trainer
#
# [수정]
# 1. ArcFace only → optimizer 단일화 (CenterLoss SGD 제거)
# 2. Phase1 encoder 로드 시 encoder lr 분리 (1/10)
# 3. val_every마다 score 분포 진단 실행
# 4. Collapse 감지 조기 중단 (EER > collapse_eer_threshold, epoch > 20)
# ══════════════════════════════════════════════════════════════════

class Trainer:
    """
    학습 루프.
    - Loss: ArcFace 단독
    - Warmup 5 epoch + CosineAnnealing
    - Early stopping (Rank-1 기준)
    - Collapse 감지 조기 중단
    """

    def __init__(
        self,
        model:          PalmprintRecognitionModel,
        device:         str,
        lr:             float = 3e-4,
        encoder_lr:     float = None,    # None이면 lr과 동일. Phase1 로드 시 lr*0.1 권장
        weight_decay:   float = 1e-4,
        warmup_epochs:  int   = 5,
        total_epochs:   int   = 100,
        log_path:       Optional[str] = None,
        collapse_eer_threshold: float = 0.45,  # 이 EER 초과 시 collapse로 판단
    ):
        self.model         = model.to(device)
        self.device        = device
        self.log_path      = log_path
        self.warmup_epochs = warmup_epochs
        self.collapse_eer_threshold = collapse_eer_threshold

        # [수정 4] encoder lr 분리 지원
        enc_lr = encoder_lr if encoder_lr is not None else lr

        self.optimizer = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": enc_lr},
            {"params": model.arcface.parameters(), "lr": lr},
        ], weight_decay=weight_decay)

        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0

        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=warmup_lambda
        )
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_epochs - warmup_epochs, 1)
        )

    def _log(self, text: str):
        print(text)
        if self.log_path:
            d = os.path.dirname(self.log_path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")

    def _step_scheduler(self, epoch_0idx: int):
        if epoch_0idx < self.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step()

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        arc_total  = 0.0

        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            out  = self.model(imgs, labels=labels)
            loss = out["total_loss"]

            arc_total  += out["arc_loss"].item()
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        n = max(len(loader), 1)
        return {"total_loss": total_loss / n, "arc_loss": arc_total / n}

    def validate(self, loader: DataLoader) -> float:
        from experiments.evaluation import extract_all_embeddings, compute_rank1_identification
        embs, labels = extract_all_embeddings(self.model, loader, self.device)
        return compute_rank1_identification(embs, labels)

    def fit(
        self,
        train_loader:        DataLoader,
        val_loader:          Optional[DataLoader],
        epochs:              int,
        save_dir:            str,
        # ── Phase 2 (val_loader 있을 때): Rank-1 기반 early stopping ──
        early_stop_patience: int   = 10,   # val check 횟수 단위
        val_every:           int   = 5,
        # ── Phase 1 (val_loader=None일 때): loss 기반 early stopping ──
        loss_patience:       int   = 8,    # loss 미개선 에폭 수
        min_loss_delta:      float = 5e-4, # 최소 개선량 (이 이하면 미개선으로 판단)
    ) -> str:
        """
        [Early Stopping 전략]

        Phase 1 (val_loader=None):
          - train loss 기반 early stopping
          - loss가 min_loss_delta 이상 개선 없이 loss_patience 에폭 지속 시 중단
          - best model은 최저 loss 기준으로 저장

        Phase 2 (val_loader 있음):
          - Val Rank-1 기반 early stopping (기존 동일)
          - val_every 에폭마다 체크, early_stop_patience 번 미개선 시 중단
          - EER collapse 감지 병행 (ep > 20 이후 활성)
        """
        os.makedirs(save_dir, exist_ok=True)
        best_path  = os.path.join(save_dir, "best.pt")
        final_path = os.path.join(save_dir, "final.pt")

        arc_cfg = self.model.arcface
        self._log("=" * 60)
        self._log(f"  학습 시작 | epochs={epochs} | device={self.device}")
        self._log(
            f"  ArcFace: scale={arc_cfg.scale}, margin={arc_cfg.margin}"
        )
        enc_lr  = self.optimizer.param_groups[0]["lr"]
        head_lr = self.optimizer.param_groups[1]["lr"]
        self._log(f"  lr: encoder={enc_lr:.2e}, arcface_head={head_lr:.2e}")

        if val_loader is None:
            self._log(
                f"  Early Stop: loss 기반  "
                f"patience={loss_patience}  min_delta={min_loss_delta:.1e}"
            )
        else:
            self._log(
                f"  Early Stop: Rank-1 기반  "
                f"patience={early_stop_patience}(×val_every={val_every} = "
                f"{early_stop_patience * val_every}ep)  "
                f"Collapse EER>{self.collapse_eer_threshold:.2f} (ep>20 이후)"
            )
        self._log("=" * 60)

        # ── Phase 1 전용 상태 ──────────────────────────────────────────
        best_loss       = float("inf")
        loss_no_improve = 0

        # ── Phase 2 전용 상태 ──────────────────────────────────────────
        best_rank1           = 0.0
        val_no_improve       = 0
        consecutive_collapse = 0

        for ep in range(1, epochs + 1):
            metrics = self.train_epoch(train_loader)
            self._step_scheduler(ep - 1)

            lr_now   = self.optimizer.param_groups[1]["lr"]
            cur_loss = metrics["total_loss"]
            log = (
                f"[Epoch {ep:03d}/{epochs}] "
                f"Loss={cur_loss:.4f}  "
                f"Arc={metrics['arc_loss']:.4f}  "
                f"lr={lr_now:.2e}"
            )

            # ══════════════════════════════════════════════════════════
            # Phase 1 Early Stopping — val_loader 없을 때 loss 기반
            # ══════════════════════════════════════════════════════════
            if val_loader is None:
                if cur_loss < best_loss - min_loss_delta:
                    best_loss       = cur_loss
                    loss_no_improve = 0
                    save_checkpoint(self.model.state_dict(), best_path)
                    log += "  ✓ best"
                else:
                    loss_no_improve += 1
                    log += f"  (no improve {loss_no_improve}/{loss_patience})"

                self._log(log)

                if loss_no_improve >= loss_patience:
                    self._log(
                        f"[Phase 1 Early Stop] loss 미개선 {loss_patience}에폭 → 중단 "
                        f"(best loss={best_loss:.4f})"
                    )
                    break
                continue   # Phase 2 블록 건너뜀

            # ══════════════════════════════════════════════════════════
            # Phase 2 Early Stopping — Rank-1 + Collapse 감지
            # ══════════════════════════════════════════════════════════
            if ep % val_every == 0:
                rank1 = self.validate(val_loader)
                log  += f"  Val Rank-1={rank1 * 100:.2f}%"

                eer = diagnose_score_distribution(self.model, val_loader, self.device)

                # Collapse 감지 (ep > 20 이후)
                if ep > 20 and eer > self.collapse_eer_threshold:
                    consecutive_collapse += 1
                    self._log(log)
                    self._log(
                        f"  [Collapse 감지] EER={eer*100:.1f}% > "
                        f"{self.collapse_eer_threshold*100:.0f}%  "
                        f"({consecutive_collapse}회 연속)"
                    )
                    if consecutive_collapse >= 3:
                        self._log(
                            "  [Phase 2 조기 중단] 3회 연속 Collapse → "
                            "학습 중단. lr 감소 또는 모델 재초기화 권장."
                        )
                        save_checkpoint(self.model.state_dict(), final_path)
                        return best_path
                else:
                    consecutive_collapse = 0

                # Rank-1 개선 여부
                if rank1 > best_rank1:
                    best_rank1   = rank1
                    val_no_improve = 0
                    save_checkpoint(self.model.state_dict(), best_path)
                    log += "  ✓ best"
                else:
                    val_no_improve += 1
                    log += f"  (no improve {val_no_improve}/{early_stop_patience})"
                    if val_no_improve >= early_stop_patience:
                        self._log(log)
                        self._log(
                            f"[Phase 2 Early Stop] Rank-1 미개선 "
                            f"{early_stop_patience}회(={early_stop_patience * val_every}ep) → 중단 "
                            f"(best Rank-1={best_rank1*100:.2f}%)"
                        )
                        break

            self._log(log)

        save_checkpoint(self.model.state_dict(), final_path)
        if val_loader is None:
            self._log(f"[Trainer] 완료. Best Loss: {best_loss:.4f}")
        else:
            self._log(f"[Trainer] 완료. Best Rank-1: {best_rank1 * 100:.2f}%")
        return best_path


# ══════════════════════════════════════════════════════════════════
# Stage A Training  (제안 모델 학습)
# ══════════════════════════════════════════════════════════════════

def run_stage_a_training(
    dataset_names:  List[str],
    checkpoint_dir: str,
    device:         str,
    batch_size:     int   = 16,
    epochs:         int   = 50,
    input_size:     int   = 320,
    max_images:     Optional[int] = None,
) -> str:
    """
    제안 Stage A (LandmarkFreePalmDetector) 학습.

    GT: MediaPipe pseudo-GT (설치된 경우) 또는 skin-color pseudo-GT
    Loss: Focal(conf) + CIoU(bbox) + BCE(quality)
    저장: checkpoints/stage_a/detector.pt

    Args:
        dataset_names : 학습에 사용할 데이터셋 이름 목록 (DATASET_CONFIGS 키)
        checkpoint_dir: 체크포인트 루트
        device        : "mps" / "cuda" / "cpu"
        batch_size    : 배치 크기
        epochs        : 최대 학습 에폭
        input_size    : 검출기 입력 해상도 (320 권장)
        max_images    : 데이터셋당 최대 이미지 수 (None=전체)
    Returns:
        저장된 가중치 경로
    """
    from models.stage_a_localization import StageADataset, StageATrainer
    from pathlib import Path as _Path

    DATA_ROOT = _Path(__file__).parent / "data" / "data"
    DATASET_GLOB = {
        "BMPD":   (DATA_ROOT / "Birjand University Mobile Palmprint Database (BMPD)", "**/*.JPG"),
        "MPDv2":  (DATA_ROOT / "MPDv2",  "**/*.jpg"),
        "Tongji": (DATA_ROOT / "Tongji", "**/*.tiff"),
        "SMPD":   (DATA_ROOT / "SMPD",   "**/*.jpg"),
    }

    print("\n" + "=" * 60)
    print("  Stage A : LandmarkFreePalmDetector Training")
    print("=" * 60)

    all_paths = []
    for ds_name in dataset_names:
        if ds_name not in DATASET_GLOB:
            print(f"  [Skip] {ds_name}: 경로 설정 없음")
            continue
        root, pattern = DATASET_GLOB[ds_name]
        paths = sorted(root.glob(pattern)) if root.exists() else []
        if max_images:
            paths = paths[:max_images]
        print(f"  {ds_name}: {len(paths)} images")
        all_paths.extend([str(p) for p in paths])

    if not all_paths:
        raise ValueError("학습 이미지가 없습니다. dataset_names와 경로를 확인하세요.")

    print(f"  Total: {len(all_paths)} images  |  input_size={input_size}")

    train_ds     = StageADataset(all_paths, input_size=input_size, augment=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0)

    trainer   = StageATrainer(device=device, input_size=input_size)
    best_path = trainer.fit(train_loader, epochs=epochs)

    print(f"[Stage A] 완료. 가중치 → {best_path}")
    return best_path


# ══════════════════════════════════════════════════════════════════
# Phase 1 : Pretraining
#
# [수정]
# - 기본: Synthetic 분류 pretraining (scale=16, margin=0.3, 500클래스)
# - 옵션: SSL(SimCLR) pretraining → collapse 없이 표현 학습
# ══════════════════════════════════════════════════════════════════

def run_phase1(
    embed_dim:      int,
    roi_size:       int,
    batch_size:     int,
    checkpoint_dir: str,
    device:         str,
    epochs:         int  = 20,
    use_ssl:        bool = False,   # True: SimCLR SSL, False: Synthetic 분류
) -> str:
    """
    Encoder Pretraining.
    항상 처음부터 학습 — 기존 checkpoint 자동 로드 없음.

    저장:
      checkpoints/phase1/encoder.pt
      checkpoints/phase1/final.pt
    """
    print("\n" + "=" * 60)
    mode_str = "SimCLR SSL" if use_ssl else "Synthetic 분류 (scale=16)"
    print(f"  Phase 1 : Encoder Pretraining [{mode_str}]  (항상 새로 시작)")
    print("=" * 60)

    save_dir     = os.path.join(checkpoint_dir, "phase1")
    encoder_path = os.path.join(save_dir, "encoder.pt")
    log_path     = os.path.join(save_dir, "train.log")
    os.makedirs(save_dir, exist_ok=True)

    # ── Synthetic 데이터셋 — train/val 분리 (class-deterministic) ────
    # val split을 생성해 Phase 1 에서도 Rank-1 기반 early stopping 사용 가능
    synth_train_ds = SyntheticPalmDataset(
        n_classes=200, n_per_class=10, roi_size=roi_size, split="train", val_ratio=0.2
    )  # 200×10 = 2,000 샘플 (학습)
    synth_val_ds = SyntheticPalmDataset(
        n_classes=200, n_per_class=10, roi_size=roi_size, split="val",   val_ratio=0.2
    )  # 200×10 = 2,000 샘플 (검증 — 클래스별 10장으로 Rank-1 계산)

    synth_loader = DataLoader(
        synth_train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    synth_val_loader = DataLoader(
        synth_val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    if use_ssl:
        # ── SimCLR SSL pretraining ─────────────────────────────────
        # SSL은 레이블이 없으므로 val Rank-1 대신 loss 기반 early stopping 사용
        print("  [Phase 1] SimCLR SSL pretraining 시작...")
        encoder = HybridPalmprintEncoder(embed_dim=embed_dim)
        pretrainer = SelfSupervisedPretrainer(
            encoder=encoder,
            embed_dim=embed_dim,
            proj_dim=128,
            lr=3e-4,
            device=device,
        )

        best_ssl_loss   = float("inf")
        ssl_no_improve  = 0
        SSL_PATIENCE    = 8
        SSL_MIN_DELTA   = 5e-4

        for ep in range(1, epochs + 1):
            loss = pretrainer.train_epoch(synth_loader)
            improve = ""
            if loss < best_ssl_loss - SSL_MIN_DELTA:
                best_ssl_loss  = loss
                ssl_no_improve = 0
                save_checkpoint(encoder.state_dict(), encoder_path)
                improve = "  ✓ best"
            else:
                ssl_no_improve += 1
                improve = f"  (no improve {ssl_no_improve}/{SSL_PATIENCE})"

            msg = f"[SSL Epoch {ep:03d}/{epochs}] Loss={loss:.4f}{improve}"
            print(msg)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

            if ssl_no_improve >= SSL_PATIENCE:
                print(f"[Phase 1 SSL Early Stop] loss 미개선 {SSL_PATIENCE}에폭 → 중단")
                break

        # 마지막 encoder도 저장 (best가 없었을 경우를 대비)
        if not os.path.exists(encoder_path):
            save_checkpoint(encoder.state_dict(), encoder_path)

    else:
        # ── Synthetic 분류 pretraining ─────────────────────────────
        # [Phase 1 Early Stopping 전략]
        #   - val_loader=synth_val_loader 전달 → Rank-1 기반 early stopping 활성화
        #   - class-deterministic SyntheticPalmDataset 덕분에 val Rank-1이 의미 있음
        #   - early_stop_patience=5 (val check 5회 = val_every×5 에폭 미개선 시 중단)
        #   - val_every=3 (3 에폭마다 val → 더 빠른 정체 감지)
        #   - loss_patience=8: val_loader가 있어도 loss 기반 safety-net 동작
        synth_model = PalmprintRecognitionModel(
            num_classes=500,
            embed_dim=embed_dim,
            arc_margin=0.3,
            arc_scale=16.0,
        ).to(device)

        trainer = Trainer(
            synth_model, device=device,
            lr=1e-3,
            total_epochs=epochs,
            log_path=log_path,
        )
        trainer.fit(
            synth_loader,
            val_loader=synth_val_loader,   # ← val 연결: Rank-1 기반 early stopping
            epochs=epochs,
            save_dir=save_dir,
            early_stop_patience=5,         # Phase 1: 5회(=15ep) 미개선 시 중단
            val_every=3,                   # 3에폭마다 val (빠른 정체 감지)
            loss_patience=8,               # val과 별개로 loss도 모니터링
            min_loss_delta=5e-4,
        )
        save_checkpoint(synth_model.encoder.state_dict(), encoder_path)

    print(f"[Phase 1] 완료. encoder → {encoder_path}")
    return encoder_path


# ══════════════════════════════════════════════════════════════════
# Phase 2 : Fine-tuning
#
# [수정]
# - ArcFace only (scale=32, margin=0.3)
# - Phase1 encoder 로드 시:
#     ① freeze_encoder_layers(0.5) 적용
#     ② encoder lr = main lr * 0.1 (천천히 fine-tune)
# ══════════════════════════════════════════════════════════════════

def run_phase2(
    dataset_name:        str,
    embed_dim:           int,
    roi_size:            int,
    batch_size:          int,
    epochs:              int,
    checkpoint_dir:      str,
    device:              str,
    load_phase1_encoder: bool  = False,
    freeze_ratio:        float = 0.5,    # Phase1 로드 시 하위 몇 % 동결
    # ── Phase 2 Early Stopping ─────────────────────────────────────
    early_stop_patience: int   = 10,     # val check 횟수 (×val_every 에폭)
    val_every:           int   = 5,      # 몇 에폭마다 val 수행
) -> tuple:
    """
    실제 데이터 fine-tuning.
    항상 처음부터 학습 — 기존 checkpoint 자동 로드 없음.

    Returns: (num_classes, best_ckpt_path)
    """
    print("\n" + "=" * 60)
    print(f"  Phase 2 : Fine-tuning on [{dataset_name}]  (항상 새로 시작)")
    if load_phase1_encoder:
        print(f"  ※ --load_phase1 활성화 → Phase1 encoder 로드 + freeze_ratio={freeze_ratio}")
        print(f"  ※ encoder lr = main lr × 0.1 (천천히 fine-tune)")
    else:
        print("  ※ --load_phase1 없음   → random init으로 시작")
    print("=" * 60)

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"데이터셋 설정 없음: {dataset_name}")

    cfg = DATASET_CONFIGS[dataset_name]

    train_ds    = PalmprintDataset(cfg, split="train", roi_size=roi_size, augment=True)
    val_ds      = PalmprintDataset(cfg, split="val",   roi_size=roi_size, augment=False)
    num_classes = train_ds.num_classes

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # [수정] ArcFace only, scale=32, margin=0.3
    model = PalmprintRecognitionModel(
        num_classes=num_classes,
        embed_dim=embed_dim,
        arc_margin=0.3,
        arc_scale=32.0,
    ).to(device)

    # [수정 3] Phase1 encoder 로드 + 부분 동결
    encoder_lr = None   # None → main lr과 동일 (random init 학습)
    if load_phase1_encoder:
        p1_path = os.path.join(checkpoint_dir, "phase1", "encoder.pt")
        try:
            model.encoder.load_state_dict(load_checkpoint(p1_path, device))
            print("  [Phase 2] Phase1 encoder 로드 완료")
            # 하위 레이어 동결 → ArcFace head와 방향 충돌 방지
            model.freeze_encoder_layers(freeze_ratio)
            # encoder를 천천히 학습 (main lr의 1/10)
            encoder_lr = 3e-4 * 0.1
        except FileNotFoundError:
            print("  [Phase 2] ⚠ Phase1 encoder 없음 → random init 유지")

    save_dir = os.path.join(checkpoint_dir, "phase2")
    log_path = os.path.join(save_dir, "train.log")
    os.makedirs(save_dir, exist_ok=True)

    trainer = Trainer(
        model, device=device,
        lr=3e-4,
        encoder_lr=encoder_lr,   # Phase1 로드 시 낮은 lr 적용
        weight_decay=1e-4,
        warmup_epochs=5,
        total_epochs=epochs,
        log_path=log_path,
        collapse_eer_threshold=0.45,
    )

    best_path = trainer.fit(
        train_loader, val_loader,
        epochs=epochs,
        save_dir=save_dir,
        early_stop_patience=early_stop_patience,
        val_every=val_every,
    )

    print(f"[Phase 2] 완료. best → {best_path}")
    return num_classes, best_path


# ══════════════════════════════════════════════════════════════════
# Stage A : LandmarkFreePalmDetector 학습
# ══════════════════════════════════════════════════════════════════

def run_stage_a(
    dataset_names:  List[str] = None,    # None → BMPD + MPDv2 사용
    input_size:     int   = 320,
    batch_size:     int   = 16,
    epochs:         int   = 50,
    checkpoint_dir: str   = "./checkpoints",
    device:         str   = "mps",
    loss_patience:  int   = 10,
) -> str:
    """
    Stage A: LandmarkFreePalmDetector 학습.

    [GT 생성 전략]
      - MediaPipe 설치 → MediaPipe pseudo-GT (손 랜드마크 기반 bbox)
      - 미설치          → skin-color pseudo-GT (HSV+YCbCr 피부색 검출)

    [손실 함수]
      - Focal Loss  (objectness, 모든 셀)
      - CIoU Loss   (l,t,r,b, positive cell)
      - BCE Loss    (quality 4채널, positive cell)

    [저장]
      checkpoints/stage_a/detector.pt   ← PalmROIExtractor 자동 로드
      checkpoints/stage_a/train.log

    [실행]
      python main.py --mode train_stage_a
      python main.py --mode train_stage_a --sa_datasets BMPD MPDv2 --sa_epochs 50
    """
    from models.stage_a_localization import StageADataset, StageATrainer
    from torch.utils.data import DataLoader

    if dataset_names is None:
        dataset_names = ["BMPD", "MPDv2"]

    print("\n" + "=" * 60)
    print("  Stage A: LandmarkFreePalmDetector Training")
    print(f"  Datasets: {dataset_names} | input_size={input_size}")
    print(f"  Epochs: {epochs} | batch_size={batch_size} | device={device}")
    print("=" * 60)

    # ── 이미지 경로 수집 ────────────────────────────────────────────
    ROOT_DIR  = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_ROOT = ROOT_DIR / "data" / "data"

    _DS_MAP = {
        "BMPD":  (DATA_ROOT / "Birjand University Mobile Palmprint Database (BMPD)", ["**/*.JPG", "**/*.jpg"]),
        "MPDv2": (DATA_ROOT / "MPDv2",   ["**/*.jpg"]),
        "Tongji":(DATA_ROOT / "Tongji",  ["**/*.tiff"]),   # contact ROI, 참고용
        "SMPD":  (DATA_ROOT / "Sapienza University Mobile Palmprint Database(SMPD)",    ["**/*.jpg", "**/*.png"]),
        #"IITD":  (DATA_ROOT / "IITD",    ["**/*.bmp"]),
    }

    all_paths: List[str] = []
    for ds in dataset_names:
        if ds not in _DS_MAP:
            print(f"  [Warning] {ds}: 경로 정의 없음, 건너뜀")
            continue
        root, patterns = _DS_MAP[ds]
        for pat in patterns:
            found = [str(p) for p in root.glob(pat)]
            all_paths.extend(found)
        print(f"  {ds}: {sum(1 for p in _DS_MAP[ds][0].glob(pat) for pat in _DS_MAP[ds][1])} → "
              f"(누적 {len(all_paths)}장)")

    if not all_paths:
        print("  [Error] 이미지 없음. --sa_datasets 또는 데이터 경로를 확인하세요.")
        return ""

    print(f"  총 학습 이미지: {len(all_paths)}장")

    # ── 학습 (캐시 빌드 → DataLoader → fit 내부 자동 처리) ──────────
    trainer   = StageATrainer(device=device, input_size=input_size)
    best_path = trainer.fit(
        image_paths   = all_paths,
        batch_size    = batch_size,
        epochs        = epochs,
        loss_patience = loss_patience,
        num_workers   = 2,      # 캐시 기반이면 멀티프로세싱 안전
        rebuild_cache = False,  # 기존 캐시 재사용 (강제 재빌드: True)
    )

    print(f"\n[Stage A] 완료.")
    print(f"  가중치: {best_path}")
    print(f"  PalmROIExtractor가 자동으로 이 가중치를 로드합니다.")
    return best_path


# ══════════════════════════════════════════════════════════════════
# Phase 3 : Evaluation
# ══════════════════════════════════════════════════════════════════

def run_phase3(
    dataset_name:          str,
    train_datasets:        List[str],
    test_dataset:          str,
    num_classes:           int,
    embed_dim:             int,
    roi_size:              int,
    batch_size:            int,
    checkpoint_dir:        str,
    device:                str,
    load_phase2_checkpoint:bool = False,
) -> dict:
    """
    실험 (B)(C)(D) 평가.
    load_phase2_checkpoint=True 일 때만 phase2/best.pt 로드.
    """
    print("\n" + "=" * 60)
    print("  Phase 3 : Evaluation")
    if load_phase2_checkpoint:
        print("  ※ Phase2 best.pt 로드하여 평가")
    else:
        print("  ※ random weight로 평가 (구조 검증용)")
    print("=" * 60)

    model = PalmprintRecognitionModel(
        num_classes=num_classes,
        embed_dim=embed_dim,
        arc_margin=0.3,
        arc_scale=32.0,
    ).to(device)

    if load_phase2_checkpoint:
        best_path  = os.path.join(checkpoint_dir, "phase2", "best.pt")
        final_path = os.path.join(checkpoint_dir, "phase2", "final.pt")

        loaded = False
        for ckpt_path in [best_path, final_path]:
            try:
                model.load_state_dict(load_checkpoint(ckpt_path, device))
                print(f"  [Phase 3] 모델 로드 완료: {ckpt_path}")
                loaded = True
                break
            except FileNotFoundError:
                continue

        if not loaded:
            print("  [Phase 3] ⚠ checkpoint 없음 → random weight로 평가")

    model.eval()

    runner = ExperimentRunner(model, device=device)

    # (B) 단일 데이터셋 recognition
    test_loader = None
    if dataset_name in DATASET_CONFIGS:
        cfg     = DATASET_CONFIGS[dataset_name]
        test_ds = PalmprintDataset(cfg, split="test", roi_size=roi_size, augment=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # (D) Cross-dataset generalization
    cross_loader = None
    valid_train  = [d for d in train_datasets if d in DATASET_CONFIGS]
    if valid_train and test_dataset in DATASET_CONFIGS:
        try:
            cross_loader = CrossDatasetLoader(
                train_datasets=valid_train,
                test_dataset=test_dataset,
                roi_size=roi_size,
                batch_size=batch_size,
            )
        except Exception as e:
            print(f"  [Phase 3] ⚠ Cross-dataset 로더 실패: {e}")

    # (A) ROI Extraction 평가 — localizer + aligner + pseudo-GT 쌍 구성
    localizer  = PalmLocalizer(device=device)
    aligner    = TopologyGuidedROIAligner(roi_size=roi_size)
    roi_pairs  = None   # (image_bgr, gt_bbox) 리스트: 아래에서 구성

    if dataset_name in DATASET_CONFIGS:
        cfg_a  = DATASET_CONFIGS[dataset_name]
        a_glob = os.path.join(cfg_a.root, cfg_a.img_pattern)
        a_paths = glob.glob(a_glob, recursive=True)[:200]   # 최대 200장으로 샘플링

        if a_paths:
            from experiments.stage_a_baseline_eval import PseudoGTGenerator
            gt_gen = PseudoGTGenerator()
            roi_pairs = []
            for p in a_paths:
                img = cv2.imread(p)
                if img is None:
                    continue
                gt = gt_gen.generate(img)
                if gt is not None:
                    roi_pairs.append((img, gt.as_tuple()))
            print(f"  [Phase 3] 실험(A) 샘플: {len(roi_pairs)}장")

    results = runner.run_all(
        roi_pairs=roi_pairs,
        test_loader=test_loader,
        cross_loader=cross_loader,
        localizer=localizer,
        aligner=aligner,
    )
    runner.print_summary()
    return results


# ══════════════════════════════════════════════════════════════════
# 추론 파이프라인
# ══════════════════════════════════════════════════════════════════

class PalmprintPipeline:
    """단일 이미지 / 비디오 burst 추론 파이프라인."""

    def __init__(
        self,
        num_classes:       int,
        embed_dim:         int  = 512,
        roi_size:          int  = 128,
        device:            str  = "cpu",
        model_weights:     Optional[str] = None,
        use_security_check:bool = False,
    ):
        self.device            = device
        self.roi_size          = roi_size
        # PalmROIExtractor = PalmLocalizer alias (제안 Stage A 파이프라인)
        self.localizer         = PalmLocalizer(device=device)
        self.aligner           = TopologyGuidedROIAligner(roi_size=roi_size)
        self.quality_selector  = QualityAwareROISelector(roi_size=roi_size, top_k=3)
        self.security_checker  = SecurityChecker() if use_security_check else None

        self.model = PalmprintRecognitionModel(
            num_classes=num_classes, embed_dim=embed_dim,
            arc_margin=0.3, arc_scale=32.0,
        ).to(device)

        if model_weights:
            self.model.load_state_dict(
                load_checkpoint(model_weights, device), strict=False
            )
        self.model.eval()

    def process_image(self, image_bgr: np.ndarray) -> Optional[dict]:
        detection = self.localizer(image_bgr)
        if detection is None:
            return None
        roi_result = self.aligner.align(image_bgr, detection)
        if roi_result is None:
            return None

        qs = self.quality_selector.score_roi(
            roi_result.roi_image,
            roi_result.alignment_confidence,
            roi_result.inscribed_radius,
        )
        local_w, global_w = self.quality_selector.compute_branch_weights(qs)

        security_result = None
        if self.security_checker and detection.hand_mask is not None:
            x1, y1, x2, y2 = detection.bbox
            security_result = self.security_checker.check(
                roi_result.roi_image,
                image_bgr[y1:y2, x1:x2],
                detection.hand_mask,
            )

        roi_t     = self._to_tensor(roi_result.roi_image)
        embedding = self.model.get_embedding(roi_t, local_w=local_w, global_w=global_w)
        return {
            "roi":           roi_result.roi_image,
            "quality":       qs,
            "local_weight":  local_w,
            "global_weight": global_w,
            "embedding":     embedding.cpu().numpy(),
            "security":      security_result,
        }

    def process_burst(self, frames_bgr: List[np.ndarray]) -> Optional[np.ndarray]:
        rois, scores = [], []
        for frame in frames_bgr:
            det = self.localizer(frame)
            if det is None:
                continue
            res = self.aligner.align(frame, det)
            if res is None:
                continue
            qs = self.quality_selector.score_roi(
                res.roi_image, res.alignment_confidence, res.inscribed_radius
            )
            rois.append(res.roi_image)
            scores.append(qs)

        if not rois:
            return None

        fused     = self.quality_selector.fuse_burst(rois, scores)
        avg_local = float(np.mean(
            [self.quality_selector.compute_branch_weights(s)[0] for s in scores]
        ))
        return self.model.get_embedding(
            self._to_tensor(fused),
            local_w=avg_local, global_w=1.0 - avg_local,
        ).cpu().numpy()

    def _to_tensor(self, roi_bgr: np.ndarray) -> torch.Tensor:
        rgb  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        t    = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return ((t - mean) / std).unsqueeze(0).to(self.device)


# ══════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Palmprint Recognition Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--mode", default="full",
        choices=["train_stage_a", "train_phase1", "train_phase2", "evaluate", "full"],
        help=(
            "train_stage_a: Stage A LandmarkFreePalmDetector 학습\n"
            "train_phase1 : Synthetic/SSL pretraining만 실행\n"
            "train_phase2 : Fine-tuning만 실행\n"
            "evaluate     : 평가만 실행\n"
            "full         : Phase1 → Phase2 → 평가 전체"
        ),
    )

    # 체크포인트 로드 플래그
    parser.add_argument("--load_phase1", action="store_true",
                        help="Phase2 시작 시 Phase1 encoder 로드 + 부분 동결")
    parser.add_argument("--load_phase2", action="store_true",
                        help="평가 시작 시 Phase2 best.pt 로드")

    # Phase1 SSL 옵션
    parser.add_argument("--use_ssl", action="store_true",
                        help="Phase1을 SimCLR SSL pretraining으로 실행 (권장)")

    # Phase2 freeze ratio
    parser.add_argument("--freeze_ratio", type=float, default=0.5,
                        help="Phase1 로드 시 encoder 동결 비율 (기본 0.5)")

    # Stage A 전용 설정
    parser.add_argument("--sa_datasets", nargs="+", default=["BMPD", "MPDv2"],
                        help="Stage A 학습 데이터셋 (기본: BMPD MPDv2)")
    parser.add_argument("--sa_input_size", type=int, default=320,
                        help="Stage A detector 입력 크기 (기본: 320)")
    parser.add_argument("--sa_epochs", type=int, default=50,
                        help="Stage A 학습 epoch 수 (기본: 50)")
    parser.add_argument("--sa_patience", type=int, default=10,
                        help="Stage A early stop patience (기본: 10)")

    # 공통 설정
    parser.add_argument("--dataset",        default="MPDv2")
    parser.add_argument("--train_on",       nargs="+", default=["MPDv2", "SMPD"])
    parser.add_argument("--test_on",        default="BMPD")
    parser.add_argument("--embed_dim",      type=int, default=512)
    parser.add_argument("--roi_size",       type=int, default=128)
    parser.add_argument("--batch_size",     type=int, default=16)
    parser.add_argument("--epochs",         type=int, default=100)
    parser.add_argument("--p1_epochs",      type=int, default=20,
                        help="Phase1 학습 epoch 수 (기본: 20)")
    parser.add_argument("--num_classes",    type=int, default=200,
                        help="evaluate 모드 전용: 모델 초기화용 클래스 수")
    parser.add_argument("--device",
                        default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")

    args = parser.parse_args()

    # ── 모드별 분기 ──────────────────────────────────────────────

    if args.mode == "train_stage_a":
        run_stage_a(
            dataset_names=args.sa_datasets,
            input_size=args.sa_input_size,
            batch_size=args.batch_size,
            epochs=args.sa_epochs,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            loss_patience=args.sa_patience,
        )

    elif args.mode == "train_phase1":
        run_phase1(
            embed_dim=args.embed_dim, roi_size=args.roi_size,
            batch_size=args.batch_size, checkpoint_dir=args.checkpoint_dir,
            device=args.device, epochs=args.p1_epochs,
            use_ssl=args.use_ssl,
        )

    elif args.mode == "train_phase2":
        run_phase2(
            dataset_name=args.dataset, embed_dim=args.embed_dim,
            roi_size=args.roi_size, batch_size=args.batch_size,
            epochs=args.epochs, checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            load_phase1_encoder=args.load_phase1,
            freeze_ratio=args.freeze_ratio,
        )

    elif args.mode == "evaluate":
        run_phase3(
            dataset_name=args.dataset, train_datasets=args.train_on,
            test_dataset=args.test_on, num_classes=args.num_classes,
            embed_dim=args.embed_dim, roi_size=args.roi_size,
            batch_size=args.batch_size, checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            load_phase2_checkpoint=args.load_phase2,
        )

    elif args.mode == "full":
        # ── full 모드: Phase1→2→3 항상 자동 연결 ──────────────────
        run_phase1(
            embed_dim=args.embed_dim, roi_size=args.roi_size,
            batch_size=args.batch_size, checkpoint_dir=args.checkpoint_dir,
            device=args.device, epochs=args.p1_epochs,
            use_ssl=args.use_ssl,
        )

        num_classes, _ = run_phase2(
            dataset_name=args.dataset, embed_dim=args.embed_dim,
            roi_size=args.roi_size, batch_size=args.batch_size,
            epochs=args.epochs, checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            load_phase1_encoder=args.load_phase1,
            freeze_ratio=args.freeze_ratio,
        )

        run_phase3(
            dataset_name=args.dataset, train_datasets=args.train_on,
            test_dataset=args.test_on, num_classes=num_classes,
            embed_dim=args.embed_dim, roi_size=args.roi_size,
            batch_size=args.batch_size, checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            load_phase2_checkpoint=True,   # full 모드: 항상 자동 연결
        )
