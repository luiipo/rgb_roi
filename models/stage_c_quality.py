"""
Stage C: Quality-Aware ROI Scoring
ROI마다 blur / exposure / scale / occlusion / alignment_confidence를 계산하고,
- 단일 이미지: 최고 점수 ROI 선택
- 비디오 burst: top-k ROI 가중 평균 융합
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch
import torch.nn as nn


@dataclass
class QualityScore:
    blur_score: float        # 높을수록 선명 (0~1)
    exposure_score: float    # 노출 적절성 (0~1)
    scale_score: float       # ROI 크기 적절성 (0~1)
    occlusion_score: float   # 가림 없음 (0~1)
    alignment_conf: float    # Stage B에서 전달 (0~1)
    total: float = field(init=False)

    # 가중치 (논문 설계에 따라 조정)
    W_BLUR = 0.30
    W_EXPO = 0.20
    W_SCALE = 0.15
    W_OCC = 0.20
    W_ALIGN = 0.15

    def __post_init__(self):
        self.total = (
            self.W_BLUR * self.blur_score
            + self.W_EXPO * self.exposure_score
            + self.W_SCALE * self.scale_score
            + self.W_OCC * self.occlusion_score
            + self.W_ALIGN * self.alignment_conf
        )


# ------------------------------------------------------------------
# 개별 품질 측정 함수
# ------------------------------------------------------------------

def _blur_score(roi_bgr: np.ndarray) -> float:
    """
    Laplacian variance → 선명도.
    임계값 기반으로 0~1 정규화.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # 경험적 상한 1000 (데이터셋에 따라 조정 가능)
    return float(min(lap_var / 1000.0, 1.0))


def _exposure_score(roi_bgr: np.ndarray) -> float:
    """
    밝기 히스토그램의 유효 범위 비율.
    극단적으로 어둡거나 과노출된 픽셀이 많으면 낮아짐.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = gray.mean()
    # 적정 밝기 범위 [60, 200]
    in_range = np.logical_and(gray >= 60, gray <= 200).mean()
    return float(in_range)


def _scale_score(inscribed_radius: float, roi_size: int = 128) -> float:
    """
    inscribed circle 반지름이 ROI 크기의 30~70% 이면 최적.
    """
    ratio = inscribed_radius / (roi_size / 2.0)
    # 목표 비율 0.5 중심의 가우시안 점수
    score = np.exp(-((ratio - 0.5) ** 2) / (2 * 0.2 ** 2))
    return float(score)


def _occlusion_score(roi_bgr: np.ndarray, hand_mask_roi: Optional[np.ndarray] = None) -> float:
    """
    ROI 내 손 마스크 면적 비율.
    마스크가 없으면 피부색 비율로 근사.
    """
    if hand_mask_roi is not None:
        return float((hand_mask_roi > 0).mean())

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 15, 50], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    skin = cv2.inRange(hsv, lower, upper)
    return float(skin.mean() / 255.0)


# ------------------------------------------------------------------
# 경량 Quality Head (학습 가능 버전, 선택적)
# ------------------------------------------------------------------

class QualityHead(nn.Module):
    """
    ROI feature에서 품질 점수 5개를 회귀하는 경량 MLP.
    feature extractor의 출력을 입력으로 받음.
    """

    def __init__(self, in_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 5),
            nn.Sigmoid(),   # 각 점수 0~1
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat: (B, in_dim)
        Returns: (B, 5) → [blur, exposure, scale, occlusion, alignment]
        """
        return self.net(feat)


# ------------------------------------------------------------------
# Stage C 메인 클래스
# ------------------------------------------------------------------

class QualityAwareROISelector:
    """
    단일 이미지 또는 비디오 burst에서 ROI 품질 평가 및 선택/융합.
    """

    def __init__(self, roi_size: int = 128, top_k: int = 3):
        self.roi_size = roi_size
        self.top_k = top_k
        self.quality_head: Optional[QualityHead] = None

    def set_learned_head(self, head: QualityHead):
        self.quality_head = head

    # ------------------------------------------------------------------
    # 단일 ROI 품질 평가
    # ------------------------------------------------------------------
    def score_roi(
        self,
        roi_bgr: np.ndarray,
        alignment_conf: float,
        inscribed_radius: float,
        hand_mask_roi: Optional[np.ndarray] = None,
    ) -> QualityScore:
        return QualityScore(
            blur_score=_blur_score(roi_bgr),
            exposure_score=_exposure_score(roi_bgr),
            scale_score=_scale_score(inscribed_radius, self.roi_size),
            occlusion_score=_occlusion_score(roi_bgr, hand_mask_roi),
            alignment_conf=alignment_conf,
        )

    # ------------------------------------------------------------------
    # 단일 이미지: 최고 점수 ROI 선택
    # ------------------------------------------------------------------
    def select_best(
        self,
        rois: List[np.ndarray],
        scores: List[QualityScore],
    ) -> Tuple[np.ndarray, QualityScore]:
        best_idx = int(np.argmax([s.total for s in scores]))
        return rois[best_idx], scores[best_idx]

    # ------------------------------------------------------------------
    # 비디오 burst: top-k ROI 가중 평균 융합
    # ------------------------------------------------------------------
    def fuse_burst(
        self,
        rois: List[np.ndarray],
        scores: List[QualityScore],
    ) -> np.ndarray:
        """
        품질 점수를 softmax weight로 변환해 가중 평균 융합.
        """
        total_scores = np.array([s.total for s in scores], dtype=np.float32)

        # top-k 선택
        k = min(self.top_k, len(rois))
        topk_idx = np.argsort(total_scores)[::-1][:k]
        topk_scores = total_scores[topk_idx]
        topk_rois = [rois[i] for i in topk_idx]

        # Softmax weights
        topk_scores = topk_scores - topk_scores.max()
        weights = np.exp(topk_scores)
        weights /= weights.sum()

        fused = np.zeros_like(topk_rois[0], dtype=np.float32)
        for w, roi in zip(weights, topk_rois):
            fused += w * roi.astype(np.float32)

        return np.clip(fused, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Quality-Guided Fusion Weight (for feature fusion in Stage D)
    # ------------------------------------------------------------------
    def compute_branch_weights(
        self, score: QualityScore
    ) -> Tuple[float, float]:
        """
        blur가 크면 local branch(CNN) 비중을 낮추고,
        alignment confidence가 높으면 global branch(Transformer) 비중을 높임.

        Returns: (local_weight, global_weight) — 합 = 1
        """
        # local weight: 선명도에 비례, blur가 낮으면 global에 의존
        local_w = score.blur_score * score.exposure_score
        # global weight: alignment confidence에 비례
        global_w = score.alignment_conf * score.scale_score

        total = local_w + global_w + 1e-6
        return float(local_w / total), float(global_w / total)
