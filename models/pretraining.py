"""
Pretraining Strategy
1. Self-Supervised Pretraining (SimCLR-style contrastive learning)
2. Synthetic Pseudo-Palm Pretraining
공개 데이터 부족 문제를 완화하여 small-data generalization을 개선.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Tuple, Optional
import os
import glob
import random


# ==================================================================
# 1. Self-Supervised Pretraining (SimCLR-style)
# ==================================================================

class PalmAugmentations:
    """손바닥 이미지에 특화된 증강 파이프라인."""

    def __init__(self, roi_size: int = 128):
        self.base = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(roi_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.3),   # 손바닥은 좌우 대칭 주의
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img_bgr: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """동일 이미지에서 두 가지 다른 augmentation view 생성."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        view1 = self.base(img_rgb)
        view2 = self.base(img_rgb)
        return view1, view2


class UnlabeledPalmDataset(Dataset):
    """레이블 없는 손바닥 이미지 디렉터리 로더."""

    def __init__(self, root_dir: str, roi_size: int = 128):
        self.paths = (
            glob.glob(os.path.join(root_dir, "**/*.jpg"), recursive=True)
            + glob.glob(os.path.join(root_dir, "**/*.png"), recursive=True)
        )
        self.augment = PalmAugmentations(roi_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        if img is None:
            img = np.zeros((128, 128, 3), dtype=np.uint8)
        img = cv2.resize(img, (128, 128))
        view1, view2 = self.augment(img)
        return view1, view2


class NTXentLoss(nn.Module):
    """SimCLR NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.T = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        z1, z2: (B, D) — 두 view의 projected feature
        """
        B = z1.size(0)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)             # (2B, D)
        sim = (z @ z.T) / self.T                   # (2B, 2B)

        # 대각 마스킹 (자기 자신 제외)
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))

        # 양성 쌍: (i, i+B) 와 (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B),
            torch.arange(0, B),
        ]).to(z.device)

        return F.cross_entropy(sim, labels)


class SimCLRProjectionHead(nn.Module):
    """SimCLR projection head (2-layer MLP)."""

    def __init__(self, in_dim: int = 512, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(True),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


class SelfSupervisedPretrainer:
    """
    SimCLR 방식의 self-supervised pretraining 루프.
    """

    def __init__(self, encoder: nn.Module, embed_dim: int = 512,
                 proj_dim: int = 128, lr: float = 3e-4, device: str = "cpu"):
        self.encoder = encoder.to(device)
        self.proj_head = SimCLRProjectionHead(embed_dim, proj_dim).to(device)
        self.loss_fn = NTXentLoss(temperature=0.07)
        self.optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(self.proj_head.parameters()),
            lr=lr, weight_decay=1e-4
        )
        self.device = device

    def train_epoch(self, loader: DataLoader) -> float:
        self.encoder.train()
        self.proj_head.train()
        total_loss = 0.0

        for view1, view2 in loader:
            view1, view2 = view1.to(self.device), view2.to(self.device)

            # Forward
            # HybridPalmprintEncoder는 tensor 직접 반환
            enc_out1 = self.encoder(view1)
            feat1 = enc_out1["embedding"] if isinstance(enc_out1, dict) else enc_out1
            enc_out2 = self.encoder(view2)
            feat2 = enc_out2["embedding"] if isinstance(enc_out2, dict) else enc_out2
            z1 = self.proj_head(feat1)
            z2 = self.proj_head(feat2)

            loss = self.loss_fn(z1, z2)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    def save_pretrained(self, save_path: str):
        torch.save(self.encoder.state_dict(), save_path)
        print(f"[Pretrain] Encoder saved to {save_path}")

    def run(self, loader: DataLoader, epochs: int = 100, save_every: int = 10,
            save_dir: str = "./checkpoints"):
        os.makedirs(save_dir, exist_ok=True)
        for ep in range(1, epochs + 1):
            loss = self.train_epoch(loader)
            print(f"[SSL Epoch {ep:03d}/{epochs}] Loss: {loss:.4f}")
            if ep % save_every == 0:
                self.save_pretrained(os.path.join(save_dir, f"ssl_ep{ep:03d}.pt"))


# ==================================================================
# 2. Synthetic Pseudo-Palm Pretraining
# ==================================================================

class SyntheticPalmGenerator:
    """
    실제 손바닥 없이 합성 pseudo-palmprint 생성.
    선(lines), 주름(wrinkles), 미세질감(texture)을 절차적으로 생성.

    [핵심 설계 원칙]
    generate_with_seed(class_id, sample_id) 를 사용할 것:
      - class_id  → 클래스 고유 주선(major lines) 구조를 결정 (고정)
      - sample_id → 인스턴스별 미세 주름 variation 결정
      → 같은 class_id 샘플들은 유사한 특징을 공유하므로
        ArcFace가 클래스 경계를 학습할 수 있음.

    generate(n) / _single() 은 완전 랜덤 생성으로,
    class label 없이 사용할 때만 호출할 것.
    """

    def __init__(self, size: int = 128):
        self.size = size

    # ── 외부에서 권장하는 API ──────────────────────────────────────────
    def generate_with_seed(self, class_id: int, sample_id: int) -> np.ndarray:
        """
        class_id 기반으로 주요 구조를 고정하고
        sample_id 로 인스턴스 variation을 추가한다.

        같은 class_id  → 주선 위치/개수가 동일 → 클래스 특징 공유
        다른 class_id  → 주선이 달라 클래스 간 구분 가능
        같은 클래스 내  → 미세 주름으로 intra-class variation 표현
        """
        s = self.size
        class_rng  = random.Random(class_id * 7919)          # 클래스 구조 고정
        sample_rng = random.Random(class_id * 7919 + sample_id + 1)  # 인스턴스 variation
        np_rng     = np.random.RandomState(class_id * 7919 + sample_id + 1)

        # 배경: 클래스별 피부색 기조 고정
        base_hue = class_rng.randint(5, 20)
        base = np.full((s, s, 3), [base_hue, 80, 200], dtype=np.uint8)
        base = cv2.cvtColor(base, cv2.COLOR_HSV2BGR)
        noise = np_rng.normal(0, 8, (s, s, 3)).astype(np.int16)
        base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 주선 (클래스 고정, 3~5개) — class_rng 으로 생성
        n_major = class_rng.randint(3, 5)
        for _ in range(n_major):
            base = self._draw_palm_line_rng(base, class_rng, major=True)

        # 미세 주름 (인스턴스 variation, 8~15개) — sample_rng 으로 생성
        n_minor = sample_rng.randint(8, 15)
        for _ in range(n_minor):
            base = self._draw_palm_line_rng(base, sample_rng, major=False)

        base = cv2.GaussianBlur(base, (3, 3), 0.5)
        return base

    # ── 기존 호환성 유지 (완전 랜덤) ─────────────────────────────────
    def generate(self, n: int = 1) -> List[np.ndarray]:
        """n개의 합성 손바닥 ROI 생성 (완전 랜덤, 레이블 없이 사용할 때)."""
        return [self._single() for _ in range(n)]

    def _single(self) -> np.ndarray:
        """완전 랜덤 단일 이미지 생성 (클래스 구분 없음)."""
        s = self.size
        rng = random.Random()   # 완전 랜덤
        base_hue = rng.randint(5, 20)
        base = np.full((s, s, 3), [base_hue, 80, 200], dtype=np.uint8)
        base = cv2.cvtColor(base, cv2.COLOR_HSV2BGR)
        noise = np.random.normal(0, 8, (s, s, 3)).astype(np.int16)
        base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        for _ in range(rng.randint(3, 5)):
            base = self._draw_palm_line_rng(base, rng, major=True)
        for _ in range(rng.randint(8, 15)):
            base = self._draw_palm_line_rng(base, rng, major=False)
        base = cv2.GaussianBlur(base, (3, 3), 0.5)
        return base

    def _draw_palm_line_rng(self, img: np.ndarray,
                            rng: random.Random, major: bool = True) -> np.ndarray:
        """rng 객체를 받아 재현 가능한 베지어 곡선 선을 그린다."""
        s = self.size
        p0 = (rng.randint(0, s), rng.randint(0, s))
        p1 = (rng.randint(0, s), rng.randint(0, s))
        p2 = (rng.randint(0, s), rng.randint(0, s))

        pts = []
        for t in np.linspace(0, 1, 100):
            x = int((1-t)**2*p0[0] + 2*(1-t)*t*p1[0] + t**2*p2[0])
            y = int((1-t)**2*p0[1] + 2*(1-t)*t*p1[1] + t**2*p2[1])
            pts.append((x, y))

        thickness     = rng.randint(1, 3) if major else 1
        color_offset  = rng.randint(-40, -15)
        for i in range(len(pts) - 1):
            c = img[pts[i][1] % s, pts[i][0] % s].astype(int)
            c = np.clip(c + color_offset, 0, 255).tolist()
            cv2.line(img, pts[i], pts[i+1], c, thickness)

        return img

    # ── 기존 _draw_palm_line (global random 사용, 하위 호환) ──────────
    def _draw_palm_line(self, img: np.ndarray, major: bool = True) -> np.ndarray:
        return self._draw_palm_line_rng(img, random.Random(), major)


class SyntheticPalmDataset(Dataset):
    """
    합성 손바닥 데이터셋 (클래스별 고정 구조 사용).

    [개선 사항]
    - generate_with_seed(class_id, sample_id) 로 class-deterministic 이미지 생성
      → 같은 클래스 샘플들이 공통 주선(major lines) 구조를 공유
      → ArcFace가 클래스 경계를 실제로 학습할 수 있음
    - split='train' / 'val' 파라미터로 학습/검증 분리
      → Phase 1에서 val_loader를 생성해 early stopping 활성화 가능

    이전 방식 (완전 랜덤):
      label = idx // n_per_class
      roi = generator.generate(1)[0]   ← 클래스와 무관한 랜덤 이미지
      → ArcFace loss 정체의 근본 원인

    현재 방식 (class-deterministic):
      roi = generator.generate_with_seed(class_id, sample_id)
      → 같은 class_id = 비슷한 구조 보장
    """

    def __init__(
        self,
        n_classes:   int   = 500,
        n_per_class: int   = 20,
        roi_size:    int   = 128,
        split:       str   = "train",  # "train" or "val"
        val_ratio:   float = 0.2,      # val로 사용할 비율 (클래스당 샘플 기준)
    ):
        self.n_classes = n_classes
        self.roi_size  = roi_size
        self.generator = SyntheticPalmGenerator(roi_size)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # train/val split (클래스당 샘플 인덱스를 나눔)
        n_val   = max(1, int(n_per_class * val_ratio))
        n_train = n_per_class - n_val

        if split == "train":
            self.n_per_class = n_train
            self._sample_offset = 0          # 클래스 내 시작 인덱스
        else:  # "val"
            self.n_per_class = n_val
            self._sample_offset = n_train    # train과 겹치지 않는 인덱스

    def __len__(self):
        return self.n_classes * self.n_per_class

    def __getitem__(self, idx):
        label      = idx // self.n_per_class
        within_idx = self._sample_offset + (idx % self.n_per_class)
        # class-deterministic: 같은 (label, within_idx) → 항상 같은 이미지
        roi     = self.generator.generate_with_seed(label, within_idx)
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img_t   = self.transform(roi_rgb)
        return img_t, torch.tensor(label, dtype=torch.long)
