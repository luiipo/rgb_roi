"""
Stage B: Landmark-Free Adaptive ROI Alignment
finger valley point에 의존하지 않고, 손 마스크의 contour/palm center/
inscribed-circle/edge orientation 전체를 이용해 ROI 좌표계를 결정한다.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from models.stage_a_localization import PalmDetection


@dataclass
class ROIResult:
    roi_image: np.ndarray          # 정렬된 ROI (H x W x 3)
    transform_matrix: np.ndarray  # 2x3 affine matrix (원본 → ROI 공간)
    center: Tuple[float, float]
    angle_deg: float
    scale: float
    inscribed_radius: float
    alignment_confidence: float    # Stage C에서 품질 점수로 사용


class TopologyGuidedROIAligner:
    """
    Landmark-Free, Topology-Guided ROI Alignment.

    알고리즘:
    1. 손 마스크 contour에서 palm center 추정 (inscribed circle / distance transform)
    2. PCA 또는 orientation histogram으로 손 방향(주축) 계산
    3. 방향 기반 회전 보정 → rotation-invariant 정렬
    4. inscribed circle 기반 scale 정규화
    5. 최종 affine crop → 고정 크기 ROI
    """

    def __init__(self, roi_size: int = 128):
        self.roi_size = roi_size

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def align(
        self,
        image_bgr: np.ndarray,
        detection: PalmDetection,
    ) -> Optional[ROIResult]:
        """
        image_bgr : 원본 BGR 이미지
        detection : Stage A 결과
        Returns ROIResult or None
        """
        mask = self._get_or_build_mask(image_bgr, detection)
        if mask is None or mask.sum() < 1000:
            return None

        center, radius = self._inscribed_circle(mask)
        angle = self._dominant_orientation(mask, center)
        scale = (self.roi_size / 2.0) / max(radius, 1e-3)
        M = self._build_affine(center, angle, scale, self.roi_size)

        roi = cv2.warpAffine(
            image_bgr, M, (self.roi_size, self.roi_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        conf = self._alignment_confidence(mask, center, radius, angle)

        return ROIResult(
            roi_image=roi,
            transform_matrix=M,
            center=center,
            angle_deg=float(np.degrees(angle)),
            scale=scale,
            inscribed_radius=radius,
            alignment_confidence=conf,
        )

    # ------------------------------------------------------------------
    # Step 1: 마스크 확보
    # ------------------------------------------------------------------
    def _get_or_build_mask(
        self, image_bgr: np.ndarray, detection: PalmDetection
    ) -> Optional[np.ndarray]:
        if detection.hand_mask is not None:
            return detection.hand_mask

        # 마스크가 없으면 bbox 내부에서 피부색으로 재생성
        x1, y1, x2, y2 = detection.bbox
        roi_crop = image_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 15, 50], dtype=np.uint8)
        upper = np.array([25, 255, 255], dtype=np.uint8)
        m = cv2.inRange(hsv, lower, upper)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        full_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = m
        return full_mask

    # ------------------------------------------------------------------
    # Step 2: Inscribed circle → palm center & scale
    # ------------------------------------------------------------------
    def _inscribed_circle(
        self, mask: np.ndarray
    ) -> Tuple[Tuple[float, float], float]:
        """
        Distance transform의 최댓값 위치 = inscribed circle 중심.
        손바닥 중심(palm center)에 가장 가까운 추정.
        """
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist)
        center = (float(max_loc[0]), float(max_loc[1]))
        return center, float(max_val)

    # ------------------------------------------------------------------
    # Step 3: Dominant orientation (rotation-invariant alignment)
    # ------------------------------------------------------------------
    def _dominant_orientation(
        self, mask: np.ndarray, center: Tuple[float, float]
    ) -> float:
        """
        손 마스크의 PCA 주축 방향 → 회전 보정 각도 반환 (라디안).
        PCA 1st principal component = 손의 긴 축(중지 방향).
        """
        ys, xs = np.where(mask > 0)
        if len(xs) < 10:
            return 0.0

        pts = np.column_stack([xs - center[0], ys - center[1]]).astype(np.float32)
        cov = np.cov(pts.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # 가장 큰 고유값의 벡터 = 주축
        principal = eigenvectors[:, np.argmax(eigenvalues)]
        angle = np.arctan2(principal[1], principal[0])
        return angle

    # ------------------------------------------------------------------
    # Step 4: Affine matrix 구성
    # ------------------------------------------------------------------
    def _build_affine(
        self,
        center: Tuple[float, float],
        angle: float,
        scale: float,
        size: int,
    ) -> np.ndarray:
        """
        중심 center를 출력 이미지 중앙으로 옮기면서,
        angle 만큼 회전 + scale 정규화하는 2×3 affine matrix.
        """
        cx, cy = center
        half = size / 2.0
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        # [scale*cos, -scale*sin, tx]
        # [scale*sin,  scale*cos, ty]
        M = np.array([
            [scale * cos_a, -scale * sin_a, half - scale * (cos_a * cx - sin_a * cy)],
            [scale * sin_a,  scale * cos_a, half - scale * (sin_a * cx + cos_a * cy)],
        ], dtype=np.float64)
        return M

    # ------------------------------------------------------------------
    # Step 5: Alignment confidence
    # ------------------------------------------------------------------
    def _alignment_confidence(
        self,
        mask: np.ndarray,
        center: Tuple[float, float],
        radius: float,
        angle: float,
    ) -> float:
        """
        간단한 alignment 신뢰도:
        - inscribed circle 내부 마스크 coverage 비율
        - 마스크 면적 대비 convex hull 충실도
        두 값의 기하 평균.
        """
        h, w = mask.shape[:2]
        cx, cy = int(center[0]), int(center[1])
        r = int(radius)

        # 1) circle coverage
        circle_mask = np.zeros_like(mask)
        cv2.circle(circle_mask, (cx, cy), r, 255, -1)
        overlap = np.logical_and(mask > 0, circle_mask > 0).sum()
        circle_area = max(np.pi * r * r, 1)
        coverage = min(overlap / circle_area, 1.0)

        # 2) convex hull solidity
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            hull = cv2.convexHull(c)
            solidity = cv2.contourArea(c) / max(cv2.contourArea(hull), 1)
        else:
            solidity = 0.0

        return float(np.sqrt(coverage * solidity))
