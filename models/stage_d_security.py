"""
Stage D: Security-Aware Consistency Check (선택적 확장)
ROI의 texture continuity와 hand-level consistency를 판별해
ROI embedding attack / hand composition attack에 대한 취약성을 줄인다.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TextureContinuityAnalyzer:
    """
    ROI 내부의 texture가 실제 손바닥처럼 자연스럽게 연속적인지 검사.
    합성 ROI는 경계 부근에서 급격한 통계 변화가 생기는 경향.
    """

    def __init__(self, n_blocks: int = 4):
        self.n_blocks = n_blocks  # ROI를 n×n 블록으로 분할

    def analyze(self, roi_bgr: np.ndarray) -> float:
        """
        블록 간 LBP 분포 KL divergence의 평균.
        낮을수록 texture가 연속적 (자연스러운 손바닥).
        Returns: continuity_score (0~1, 높을수록 연속적)
        """
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        bh, bw = h // self.n_blocks, w // self.n_blocks

        lbp_hists = []
        for i in range(self.n_blocks):
            for j in range(self.n_blocks):
                block = gray[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                lbp_hists.append(self._lbp_histogram(block))

        # 인접 블록 간 KL divergence 평균
        divs = []
        for idx in range(len(lbp_hists) - 1):
            p = lbp_hists[idx] + 1e-6
            q = lbp_hists[idx + 1] + 1e-6
            kl = float(np.sum(p * np.log(p / q)))
            divs.append(kl)

        mean_div = np.mean(divs) if divs else 0.0
        # KL divergence를 0~1 continuity score로 변환 (낮은 divergence = 높은 continuity)
        continuity = float(np.exp(-mean_div / 2.0))
        return continuity

    def _lbp_histogram(self, block: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """간단한 LBP histogram (OpenCV 없이 numpy로 구현)."""
        h, w = block.shape
        lbp = np.zeros_like(block, dtype=np.uint8)
        for n in range(n_points):
            angle = 2 * np.pi * n / n_points
            dx, dy = int(round(radius * np.cos(angle))), int(round(radius * np.sin(angle)))
            shifted = np.roll(np.roll(block, dy, axis=0), dx, axis=1)
            lbp += (block >= shifted).astype(np.uint8) * (2 ** n)
        hist, _ = np.histogram(lbp, bins=256, range=(0, 255), density=True)
        return hist


class HandLevelConsistencyChecker:
    """
    ROI와 원본 손 이미지 사이의 색상/질감 일관성 검사.
    공격자가 ROI 영역만 합성했다면, ROI 바깥 손 영역과 통계가 달라짐.
    """

    def check(
        self,
        roi_bgr: np.ndarray,
        full_hand_bgr: np.ndarray,
        hand_mask: np.ndarray,
    ) -> float:
        """
        ROI의 색상 분포와 전체 손의 색상 분포 비교.
        Returns: consistency_score (0~1)
        """
        roi_hist = self._color_histogram(roi_bgr)

        # 손 마스크 적용 후 전체 손 영역 histogram
        masked = cv2.bitwise_and(full_hand_bgr, full_hand_bgr, mask=hand_mask)
        hand_hist = self._color_histogram(masked)

        # Bhattacharyya distance → similarity
        similarity = cv2.compareHist(
            roi_hist.astype(np.float32),
            hand_hist.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA
        )
        # Bhattacharyya distance 0=identical, 1=very different
        return float(1.0 - similarity)

    def _color_histogram(self, bgr: np.ndarray, bins: int = 32) -> np.ndarray:
        hist = np.zeros(bins * 3, dtype=np.float32)
        for ch in range(3):
            h, _ = np.histogram(bgr[:, :, ch], bins=bins, range=(0, 255))
            hist[ch * bins:(ch + 1) * bins] = h
        hist /= hist.sum() + 1e-6
        return hist


class SecurityAwareConsistencyModule(nn.Module):
    """
    texture_continuity + hand_level_consistency를 결합해
    spoof/embedding attack 여부를 분류하는 경량 모듈.

    학습 데이터: 실제 손바닥 ROI (label=0) vs 합성/스푸핑 ROI (label=1)
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()

        # 간단한 CNN 기반 이진 분류기
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, feature_dim),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 2),   # 0=실제, 1=스푸핑
        )

        # 핸드크래프트 통계 branch
        self.stat_head = nn.Sequential(
            nn.Linear(2, 32),   # [texture_continuity, hand_consistency]
            nn.ReLU(True),
            nn.Linear(32, 2),
        )

        # 두 branch 합산
        self.fusion = nn.Linear(4, 2)

    def forward(
        self,
        roi_tensor: torch.Tensor,                  # (B, 3, H, W)
        stat_features: torch.Tensor,               # (B, 2)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, 2) — 스푸핑 분류 logit
            probs:  (B, 2) — 소프트맥스 확률
        """
        # CNN branch
        cnn_feat = self.conv_layers(roi_tensor)
        cnn_feat = cnn_feat.view(cnn_feat.size(0), -1)
        cnn_out = self.classifier(cnn_feat)

        # Stat branch
        stat_out = self.stat_head(stat_features)

        # Fusion
        fused = self.fusion(torch.cat([cnn_out, stat_out], dim=1))
        return fused, F.softmax(fused, dim=1)


# ------------------------------------------------------------------
# Stage D 통합 인터페이스
# ------------------------------------------------------------------

class SecurityChecker:
    """Stage D 통합. 훈련된 모델 없이도 규칙 기반으로 동작."""

    def __init__(
        self,
        model: SecurityAwareConsistencyModule = None,
        device: str = "cpu",
        spoof_threshold: float = 0.5,
    ):
        self.texture_analyzer = TextureContinuityAnalyzer()
        self.consistency_checker = HandLevelConsistencyChecker()
        self.model = model
        self.device = device
        self.spoof_threshold = spoof_threshold

    def check(
        self,
        roi_bgr: np.ndarray,
        full_hand_bgr: np.ndarray,
        hand_mask: np.ndarray,
    ) -> dict:
        """
        Returns dict with:
          - texture_continuity (0~1)
          - hand_consistency   (0~1)
          - spoof_prob         (0~1, 높을수록 의심)
          - is_suspicious      (bool)
        """
        tc = self.texture_analyzer.analyze(roi_bgr)
        hc = self.consistency_checker.check(roi_bgr, full_hand_bgr, hand_mask)

        spoof_prob = 0.0
        if self.model is not None:
            roi_t = self._to_tensor(roi_bgr).to(self.device)
            stat_t = torch.tensor([[tc, hc]], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                _, probs = self.model(roi_t, stat_t)
            spoof_prob = float(probs[0, 1].item())
        else:
            # 규칙 기반: texture가 불연속하거나 색상이 일치하지 않으면 의심
            spoof_prob = float((1.0 - tc) * 0.5 + (1.0 - hc) * 0.5)

        return {
            "texture_continuity": tc,
            "hand_consistency": hc,
            "spoof_prob": spoof_prob,
            "is_suspicious": spoof_prob > self.spoof_threshold,
        }

    @staticmethod
    def _to_tensor(bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0)
