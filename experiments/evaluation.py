"""
Experiments
(A) ROI Extraction 성능 평가  : IoU, ROI success rate, keypoint error, alignment consistency
(B) Recognition 성능 평가     : EER, TAR@FAR, Rank-1
(C) Efficiency 평가           : 모델 크기, FLOPs, 추론 시간, FPS
(D) Generalization 평가       : Cross-dataset (train A+B, test C)
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


# ==================================================================
# (A) ROI Extraction 평가
# ==================================================================

@dataclass
class ROIExtractionMetrics:
    iou_scores: List[float] = field(default_factory=list)
    success_flags: List[bool] = field(default_factory=list)   # IoU > threshold
    alignment_confs: List[float] = field(default_factory=list)
    iou_threshold: float = 0.5

    def add(self, pred_bbox, gt_bbox, alignment_conf: float):
        iou = compute_iou(pred_bbox, gt_bbox)
        self.iou_scores.append(iou)
        self.success_flags.append(iou >= self.iou_threshold)
        self.alignment_confs.append(alignment_conf)

    def summary(self) -> dict:
        return {
            "mean_iou": float(np.mean(self.iou_scores)) if self.iou_scores else 0.0,
            "success_rate": float(np.mean(self.success_flags)) if self.success_flags else 0.0,
            "mean_alignment_conf": float(np.mean(self.alignment_confs)) if self.alignment_confs else 0.0,
            "n_samples": len(self.iou_scores),
        }


def compute_iou(box1: Tuple, box2: Tuple) -> float:
    """
    box: (x1, y1, x2, y2)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def evaluate_roi_extraction(
    localizer,
    aligner,
    image_bbox_pairs: List[Tuple],  # [(image_bgr, gt_bbox), ...]
    iou_threshold: float = 0.5,
) -> dict:
    """
    Stage A + B 평가.
    image_bbox_pairs: 이미지와 ground-truth bounding box 쌍의 리스트.
    """
    metrics = ROIExtractionMetrics(iou_threshold=iou_threshold)

    for image_bgr, gt_bbox in image_bbox_pairs:
        detection = localizer(image_bgr)
        if detection is None:
            metrics.add((0, 0, 0, 0), gt_bbox, 0.0)
            continue

        roi_result = aligner.align(image_bgr, detection)
        align_conf = roi_result.alignment_confidence if roi_result else 0.0
        pred_bbox = detection.bbox

        metrics.add(pred_bbox, gt_bbox, align_conf)

    return metrics.summary()


# ==================================================================
# (B) Recognition 성능 평가 (EER, TAR@FAR, Rank-1)
# ==================================================================

def extract_all_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        embeddings : (N, D)
        labels     : (N,)
    """
    model.eval()
    all_embs, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            embs = out["embedding"]
            # 명시적 L2 정규화 강제
            embs = F.normalize(embs, dim=1)
            all_embs.append(embs.cpu().numpy())
            all_labels.append(labels.numpy())
                
    return np.concatenate(all_embs), np.concatenate(all_labels)


def compute_cosine_similarity_matrix(embs: np.ndarray) -> np.ndarray:
    """L2 정규화 후 cosine similarity matrix 계산."""
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_norm = embs / (norms + 1e-8)
    return embs_norm @ embs_norm.T


def compute_eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> Tuple[float, float]:
    """
    EER(Equal Error Rate) 계산.
    FAR curve와 FRR curve의 교차점을 보간법으로 정확히 찾습니다.
    Returns: (eer, threshold)
    """
    # 모든 가능한 임계값 = 전체 score 집합
    thresholds = np.sort(np.concatenate([genuine_scores, impostor_scores]))[::-1]

    fars = np.array([(impostor_scores >= t).mean() for t in thresholds])
    frrs = np.array([(genuine_scores  <  t).mean() for t in thresholds])

    # FAR - FRR 부호가 바뀌는 지점(교차점) 탐색
    diffs = fars - frrs
    sign_changes = np.where(np.diff(np.sign(diffs)))[0]

    if len(sign_changes) == 0:
        # 교차점이 없으면 차이가 가장 작은 지점 사용
        idx = np.argmin(np.abs(diffs))
        eer = float((fars[idx] + frrs[idx]) / 2.0)
        return eer, float(thresholds[idx])

    # 교차점 보간
    idx = sign_changes[0]
    # 두 점 사이 선형 보간
    t0, t1 = thresholds[idx], thresholds[idx + 1]
    d0, d1 = diffs[idx], diffs[idx + 1]

    # d0 + (d1-d0)*alpha = 0 → alpha = -d0 / (d1 - d0)
    if d1 != d0:
        alpha = -d0 / (d1 - d0)
    else:
        alpha = 0.5

    best_thresh = float(t0 + alpha * (t1 - t0))
    far_interp  = float(fars[idx]  + alpha * (fars[idx+1]  - fars[idx]))
    frr_interp  = float(frrs[idx]  + alpha * (frrs[idx+1]  - frrs[idx]))
    eer         = (far_interp + frr_interp) / 2.0

    return eer, best_thresh


def compute_tar_at_far(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    far_target: float = 1e-4,
) -> float:
    """TAR(True Accept Rate) @ 지정 FAR."""
    thresholds = np.sort(impostor_scores)[::-1]
    for t in thresholds:
        far = float((impostor_scores >= t).mean())
        if far <= far_target:
            tar = float((genuine_scores >= t).mean())
            return tar
    return 0.0


def compute_rank1_identification(
    embs: np.ndarray,
    labels: np.ndarray,
    gallery_ratio: float = 0.5,
    seed: int = 42,
) -> float:
    """
    Gallery / Probe 분리 후 Rank-1 identification accuracy.
    각 클래스의 절반을 gallery, 나머지를 probe로 사용.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)

    gallery_idx, probe_idx = [], []
    for c in classes:
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        n_gallery = max(1, int(len(c_idx) * gallery_ratio))
        gallery_idx.extend(c_idx[:n_gallery].tolist())
        probe_idx.extend(c_idx[n_gallery:].tolist())

    if not probe_idx:
        return 0.0

    gallery_embs = embs[gallery_idx]
    gallery_labels = labels[gallery_idx]
    probe_embs = embs[probe_idx]
    probe_labels = labels[probe_idx]

    # Gallery 정규화
    gnorm = np.linalg.norm(gallery_embs, axis=1, keepdims=True)
    gallery_norm = gallery_embs / (gnorm + 1e-8)
    pnorm = np.linalg.norm(probe_embs, axis=1, keepdims=True)
    probe_norm = probe_embs / (pnorm + 1e-8)

    sim = probe_norm @ gallery_norm.T   # (n_probe, n_gallery)
    top1_idx = np.argmax(sim, axis=1)
    top1_labels = gallery_labels[top1_idx]
    rank1 = float((top1_labels == probe_labels).mean())
    return rank1


def evaluate_recognition(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
    far_targets: Tuple[float, ...] = (1e-4, 1e-6),
) -> dict:
    """
    전체 recognition 평가 루틴.
    Returns dict with EER, TAR@FAR values, Rank-1.
    """
    print("[Exp-B] Extracting embeddings ...")
    embs, labels = extract_all_embeddings(model, test_loader, device)
    if len(embs) == 0:
        print("[Warning] Empty dataset. Skipping recognition evaluation.")
        return {
            "EER": 0,
            "Rank-1": 0,
            "n_genuine_pairs": 0,
            "n_impostor_pairs": 0
        }
    sim_matrix = compute_cosine_similarity_matrix(embs)

    # Genuine / Impostor 분리
    n = len(labels)
    genuine_scores, impostor_scores = [], []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_matrix[i, j])
            if labels[i] == labels[j]:
                genuine_scores.append(s)
            else:
                impostor_scores.append(s)

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    eer, eer_thresh = compute_eer(genuine_scores, impostor_scores)
    rank1 = compute_rank1_identification(embs, labels)

    result = {
        "EER": round(eer * 100, 4),
        "EER_threshold": round(eer_thresh, 4),
        "Rank-1": round(rank1 * 100, 4),
        "n_genuine_pairs": len(genuine_scores),
        "n_impostor_pairs": len(impostor_scores),
    }
    for far in far_targets:
        tar = compute_tar_at_far(genuine_scores, impostor_scores, far_target=far)
        result[f"TAR@FAR={far:.0e}"] = round(tar * 100, 4)

    return result


# ==================================================================
# (C) Efficiency 평가
# ==================================================================

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops_simple(model: torch.nn.Module, input_shape=(1, 3, 128, 128)) -> int:
    """
    간단한 FLOPs 추정 (thop 없을 경우 hook 기반 근사).
    """
    try:
        from thop import profile
        device = next(model.parameters()).device
        x = torch.zeros(input_shape).to(device)
        flops, _ = profile(model, inputs=(x,), verbose=False)
        return int(flops)
    except ImportError:
        # Hook 기반 근사
        total_flops = [0]

        def conv_hook(module, inp, out):
            b, c_out, h, w = out.shape
            _, c_in, kh, kw = module.weight.shape
            total_flops[0] += 2 * b * c_out * h * w * c_in * kh * kw

        def linear_hook(module, inp, out):
            total_flops[0] += 2 * inp[0].size(0) * module.in_features * module.out_features

        handles = []
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                handles.append(m.register_forward_hook(conv_hook))
            elif isinstance(m, torch.nn.Linear):
                handles.append(m.register_forward_hook(linear_hook))

        with torch.no_grad():
            device = next(model.parameters()).device
            model(torch.zeros(input_shape).to(device))

        for h in handles:
            h.remove()

        return total_flops[0]


def measure_inference_time(
    model: torch.nn.Module,
    input_shape=(1, 3, 128, 128),
    n_warmup: int = 10,
    n_runs: int = 100,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    평균 추론 시간(ms) 및 FPS 측정.
    """
    model.eval().to(device)
    dummy = torch.zeros(input_shape, device=device)

    # 워밍업
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)

    # 측정
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    mean_ms = (elapsed / n_runs) * 1000
    fps = 1000.0 / mean_ms

    return {
        "mean_inference_ms": round(mean_ms, 3),
        "FPS": round(fps, 1),
    }


def evaluate_efficiency(model: torch.nn.Module, device: str = "cpu") -> dict:
    """(C) Efficiency 평가 통합."""
    params = count_parameters(model)
    flops = estimate_flops_simple(model)
    timing = measure_inference_time(model, device=device)

    result = {
        "parameters_M": round(params / 1e6, 3),
        "FLOPs_G": round(flops / 1e9, 3),
        **timing,
    }
    print(f"[Exp-C] Params: {result['parameters_M']}M  "
          f"FLOPs: {result['FLOPs_G']}G  "
          f"Latency: {result['mean_inference_ms']}ms  "
          f"FPS: {result['FPS']}")
    return result


# ==================================================================
# (D) Generalization 평가 (Cross-Dataset)
# ==================================================================

def evaluate_generalization(
    model: torch.nn.Module,
    cross_loader,   # CrossDatasetLoader instance
    device: str = "cpu",
) -> dict:
    """
    train on A+B, test on C 형태의 cross-dataset generalization 평가.
    """
    print("[Exp-D] Cross-dataset generalization evaluation ...")

    test_loader = cross_loader.get_test_loader()
    if len(test_loader.dataset) == 0:
        print("[Warning] Empty cross dataset. Skipping...")
        return {"skipped": True}

    result = evaluate_recognition(model, test_loader, device=device)
    result["experiment"] = "cross_dataset_generalization"
    result["test_dataset"] = getattr(cross_loader.test_dataset.config, "name", "unknown")
    return result


# ==================================================================
# 통합 실험 러너
# ==================================================================

class ExperimentRunner:
    """
    (A)~(D) 실험을 순서대로 또는 선택적으로 실행하는 통합 클래스.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self.results: Dict[str, dict] = {}

    def run_all(
        self,
        roi_pairs: Optional[List] = None,          # (A)용
        test_loader: Optional[DataLoader] = None,  # (B)용
        cross_loader=None,                          # (D)용
        localizer=None,                             # (A)용
        aligner=None,                               # (A)용
    ):
        """(A)~(D) 전체 실험 실행."""

        # (A) ROI Extraction
        if roi_pairs and localizer and aligner:
            print("\n=== Experiment (A): ROI Extraction ===")
            self.results["A_roi_extraction"] = evaluate_roi_extraction(
                localizer, aligner, roi_pairs
            )
            print(self.results["A_roi_extraction"])

        # (B) Recognition
        if test_loader:
            print("\n=== Experiment (B): Recognition ===")
            self.results["B_recognition"] = evaluate_recognition(
                self.model, test_loader, self.device
            )
            print(self.results["B_recognition"])

        # (C) Efficiency
        print("\n=== Experiment (C): Efficiency ===")
        self.results["C_efficiency"] = evaluate_efficiency(self.model, self.device)

        # (D) Generalization
        if cross_loader:
            print("\n=== Experiment (D): Generalization ===")
            self.results["D_generalization"] = evaluate_generalization(
                self.model, cross_loader, self.device
            )
            print(self.results["D_generalization"])

        return self.results

    def print_summary(self):
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        for key, val in self.results.items():
            print(f"\n[{key}]")
            for k, v in val.items():
                print(f"  {k}: {v}")
