"""
Stage A: Landmark-Free Adaptive Palm Localization  (제안 모델)
=============================================================
파이프라인:

  Input RGB Image
        ↓
  [1] LandmarkFreePalmDetector
      - MobileNet-style DW-Sep Backbone (P3 stride=8 / P4 stride=16)
      - Feature Pyramid Neck
      - Anchor-free Detection Head  : (conf, l, t, r, b) per cell — FCOS-style
      - Quality Head                : (blur, scale, align, occlusion) per cell
  [단일 forward pass로 bbox + 품질 동시 예측]
        ↓
  [2] QualityAwareROIEstimator
      - FCOS bbox 디코딩 + NMS
      - 검출 신뢰도 × 품질 점수로 최적 bbox 선택
      - 규칙 기반 보완 품질 계산 (추론 시 네트워크 없어도 동작)
        ↓
  [3] AdaptiveROIRefiner
      - 품질 적응형 마진 확장 (low quality → 넓게)
      - Inscribed-circle 기반 정방형 정규화
      - Edge map PCA로 최대 ±30° 회전 보정
        ↓
  Final PalmDetection (bbox, confidence, quality_scores, hand_mask)

[훈련]
  StageADataset  : 이미지 + MediaPipe/skin-color Pseudo-GT 자동 생성
  StageATrainer  : Detection loss (Focal + CIoU) + Quality loss (BCE)
  저장 경로      : checkpoints/stage_a/detector.pt

[하위 호환]
  PalmLocalizer  : 기존 pipeline 코드와 호환되는 alias wrapper
"""

import os
import json
import math
import time
import glob
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ══════════════════════════════════════════════════════════════════════════════
# 0. 공용 데이터 타입
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QualityScores:
    """Stage A 품질 추정 결과."""
    blur:      float = 0.0   # 1=선명, 0=흐림
    scale:     float = 0.0   # 1=최적 크기, 0=너무 작거나 큼
    alignment: float = 0.0   # 1=정렬 신뢰도 높음
    occlusion: float = 0.0   # 1=가림 없음

    @property
    def total(self) -> float:
        return 0.35*self.blur + 0.25*self.scale + 0.25*self.alignment + 0.15*self.occlusion

    def as_tensor(self) -> torch.Tensor:
        return torch.tensor([self.blur, self.scale, self.alignment, self.occlusion])


@dataclass
class PalmDetection:
    """Stage A 최종 출력. Stage B/C와의 인터페이스."""
    bbox:          Tuple[int, int, int, int]          # (x1, y1, x2, y2) in original image
    confidence:    float
    quality:       Optional[QualityScores]  = None
    hand_mask:     Optional[np.ndarray]     = None    # Stage B 보조용
    landmarks:     Optional[np.ndarray]     = None    # None (landmark-free)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LandmarkFreePalmDetector  —  Backbone + Neck + Heads
# ══════════════════════════════════════════════════════════════════════════════

def _dw_block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    """Depthwise-Separable Conv Block."""
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
        nn.BatchNorm2d(cin),
        nn.Hardswish(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.Hardswish(inplace=True),
    )


class MobileDetectorBackbone(nn.Module):
    """
    MobileNet-style backbone.
    Input : (B, 3, H, W)
    Output: P3 (stride=8, 128-ch), P4 (stride=16, 256-ch)
    """

    def __init__(self):
        super().__init__()
        self.stem    = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16), nn.Hardswish(True),
        )                                         # H/2  × W/2,  16-ch
        self.stage1  = nn.Sequential(
            _dw_block(16, 32, 1),
            _dw_block(32, 32, 1),
        )                                         # H/2  × W/2,  32-ch
        self.stage2  = nn.Sequential(
            _dw_block(32, 64, 2),
            _dw_block(64, 64, 1),
        )                                         # H/4  × W/4,  64-ch
        self.p3_stage = nn.Sequential(
            _dw_block(64,  128, 2),
            _dw_block(128, 128, 1),
        )                                         # P3: H/8  × W/8, 128-ch
        self.p4_stage = nn.Sequential(
            _dw_block(128, 256, 2),
            _dw_block(256, 256, 1),
        )                                         # P4: H/16 × W/16, 256-ch

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x  = self.stem(x)
        x  = self.stage1(x)
        x  = self.stage2(x)
        p3 = self.p3_stage(x)
        p4 = self.p4_stage(p3)
        return p3, p4   # (B,128,H/8,W/8), (B,256,H/16,W/16)


class FeaturePyramidNeck(nn.Module):
    """
    단순 FPN: P4 → upsample → P3와 합산 → 통일된 neck_ch 채널.
    두 스케일 모두 neck_ch=128 채널로 출력.
    """

    def __init__(self, neck_ch: int = 128):
        super().__init__()
        self.p4_proj = nn.Conv2d(256, neck_ch, 1, bias=False)
        self.p3_proj = nn.Conv2d(128, neck_ch, 1, bias=False)
        self.p3_out  = _dw_block(neck_ch, neck_ch)
        self.p4_out  = _dw_block(neck_ch, neck_ch)

    def forward(self,
                p3: torch.Tensor,
                p4: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p4_up = F.interpolate(self.p4_proj(p4), size=p3.shape[-2:],
                              mode="nearest")
        n3 = self.p3_out(self.p3_proj(p3) + p4_up)
        n4 = self.p4_out(self.p4_proj(p4))
        return n3, n4


class AnchorFreeHead(nn.Module):
    """
    FCOS-style anchor-free detection + quality head.

    Detection: (conf, l, t, r, b) = 5 채널
      - conf      : objectness (sigmoid)
      - l, t, r, b: 셀 중심에서 bbox 경계까지의 거리 (exp × stride)

    Quality: (blur, scale, alignment, occlusion) = 4 채널
      - 0~1 범위 (sigmoid)
      - 학습 시 pseudo-label로 지도
    """

    DET_CH  = 5   # conf + l + t + r + b
    QUAL_CH = 4   # blur + scale + alignment + occlusion

    def __init__(self, in_ch: int = 128):
        super().__init__()
        self.det_conv = nn.Sequential(
            _dw_block(in_ch, in_ch),
            _dw_block(in_ch, in_ch),
            nn.Conv2d(in_ch, self.DET_CH, 1),
        )
        self.quality_conv = nn.Sequential(
            _dw_block(in_ch, in_ch // 2),
            nn.Conv2d(in_ch // 2, self.QUAL_CH, 1),
        )

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        det_out  = self.det_conv(feat)              # (B, 5, H, W) ← detection gradient 정상 흐름
        qual_out = self.quality_conv(feat.detach()) # (B, 4, H, W) ← backbone 차단: quality loss가 backbone 오염 방지
        return det_out, qual_out


class LandmarkFreePalmDetector(nn.Module):
    """
    제안 모델의 핵심 네트워크.

    Input : (B, 3, input_size, input_size)
    Output:
      detections: List of 2 tensors [(B,5,H3,W3), (B,5,H4,W4)]
        - P3(stride=8)와 P4(stride=16) 각각의 raw FCOS 예측
      qualities : List of 2 tensors [(B,4,H3,W3), (B,4,H4,W4)]
        - 각 스케일의 품질 예측 (sigmoid 적용 전)
    """

    STRIDES = [8, 16]

    def __init__(self, input_size: int = 320, neck_ch: int = 128):
        super().__init__()
        self.input_size = input_size
        self.backbone   = MobileDetectorBackbone()
        self.neck       = FeaturePyramidNeck(neck_ch)
        self.head_p3    = AnchorFreeHead(neck_ch)
        self.head_p4    = AnchorFreeHead(neck_ch)
        self._init_weights()

    def _init_weights(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # conf 채널 bias: prior prob 0.01 (안정적 학습 시작)
        for head in [self.head_p3, self.head_p4]:
            nn.init.constant_(head.det_conv[-1].bias[0], bias_value)

    def forward(self, x: torch.Tensor) -> Tuple[List, List]:
        p3, p4 = self.backbone(x)
        n3, n4 = self.neck(p3, p4)
        det3, qual3 = self.head_p3(n3)
        det4, qual4 = self.head_p4(n4)
        return [det3, det4], [qual3, qual4]

    @property
    def param_size_mb(self) -> float:
        return sum(p.numel() for p in self.parameters()) * 4 / 1024 / 1024


# ══════════════════════════════════════════════════════════════════════════════
# 2. QualityAwareROIEstimator  —  디코딩 + 품질 융합 + 최적 bbox 선택
# ══════════════════════════════════════════════════════════════════════════════

class QualityAwareROIEstimator:
    """
    [역할]
    1. LandmarkFreePalmDetector의 raw 출력을 bbox로 디코딩
    2. 검출 conf × 학습된 quality 점수로 최종 ranking
    3. 규칙 기반 quality 보완 (네트워크 없이도 동작 가능)

    [품질 통합 전략]
    최종 점수 = conf^α × quality_total^β  (α=0.6, β=0.4)
    """

    CONF_THRESH    = 0.35
    NMS_IOU_THRESH = 0.4
    ALPHA = 0.6
    BETA  = 0.4

    def __init__(self, input_size: int = 320):
        self.input_size = input_size
        self.strides    = LandmarkFreePalmDetector.STRIDES

    def decode(
        self,
        det_maps:  List[torch.Tensor],
        qual_maps: List[torch.Tensor],
        img_hw:    Tuple[int, int],
    ) -> List[List[Dict]]:
        """배치 내 각 이미지에 대해 후처리된 detection 리스트 반환."""
        B = det_maps[0].shape[0]
        batch_results = []

        for b in range(B):
            all_boxes, all_confs, all_quals = [], [], []

            for det_map, qual_map, stride in zip(det_maps, qual_maps, self.strides):
                det  = det_map[b]    # (5, H, W)
                qual = qual_map[b]   # (4, H, W)
                H, W = det.shape[1:]

                gy, gx = torch.meshgrid(
                    torch.arange(H, device=det.device),
                    torch.arange(W, device=det.device),
                    indexing="ij",
                )
                cx = (gx.float() + 0.5) * stride
                cy = (gy.float() + 0.5) * stride

                conf = torch.sigmoid(det[0])
                l    = torch.exp(det[1].clamp(-6, 6)) * stride
                t    = torch.exp(det[2].clamp(-6, 6)) * stride
                r    = torch.exp(det[3].clamp(-6, 6)) * stride
                b_   = torch.exp(det[4].clamp(-6, 6)) * stride

                x1 = (cx - l).clamp(0, img_hw[1]).flatten()
                y1 = (cy - t).clamp(0, img_hw[0]).flatten()
                x2 = (cx + r).clamp(0, img_hw[1]).flatten()
                y2 = (cy + b_).clamp(0, img_hw[0]).flatten()

                valid = (
                    (conf.flatten() > self.CONF_THRESH) &
                    ((x2 - x1) > 10) & ((y2 - y1) > 10)
                )
                if not valid.any():
                    continue

                boxes      = torch.stack([x1, y1, x2, y2], dim=1)[valid]
                confs_v    = conf.flatten()[valid]
                quals_flat = torch.sigmoid(
                    qual.permute(1, 2, 0).reshape(-1, 4)
                )[valid]

                all_boxes.append(boxes)
                all_confs.append(confs_v)
                all_quals.append(quals_flat)

            if not all_boxes:
                batch_results.append([])
                continue

            boxes  = torch.cat(all_boxes)
            confs  = torch.cat(all_confs)
            quals  = torch.cat(all_quals)

            keep   = self._nms(boxes, confs)
            boxes  = boxes[keep].cpu().numpy()
            confs  = confs[keep].cpu().numpy()
            quals  = quals[keep].cpu().numpy()

            results = []
            for i in range(len(boxes)):
                q = QualityScores(
                    blur=float(quals[i, 0]),
                    scale=float(quals[i, 1]),
                    alignment=float(quals[i, 2]),
                    occlusion=float(quals[i, 3]),
                )
                results.append({
                    "bbox":    tuple(boxes[i].astype(int)),
                    "conf":    float(confs[i]),
                    "quality": q,
                })
            batch_results.append(results)

        return batch_results

    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        order = scores.argsort(descending=True)
        keep  = []
        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break
            iou   = self._box_iou_1n(boxes[i], boxes[order[1:]])
            order = order[1:][iou <= self.NMS_IOU_THRESH]
        return torch.tensor(keep, dtype=torch.long)

    @staticmethod
    def _box_iou_1n(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        ix1 = torch.maximum(box[0], boxes[:, 0])
        iy1 = torch.maximum(box[1], boxes[:, 1])
        ix2 = torch.minimum(box[2], boxes[:, 2])
        iy2 = torch.minimum(box[3], boxes[:, 3])
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        a1 = (box[2]-box[0]) * (box[3]-box[1])
        a2 = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
        return inter / (a1 + a2 - inter + 1e-6)

    @staticmethod
    def compute_rule_quality(img_bgr: np.ndarray, bbox: Tuple) -> QualityScores:
        """규칙 기반 품질 계산 (네트워크 없을 때 / 보완용)."""
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img_bgr.shape[1], x2); y2 = min(img_bgr.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return QualityScores()
        roi = img_bgr[y1:y2, x1:x2]

        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur    = float(min(lap_var / 800.0, 1.0))

        area_ratio = (x2-x1)*(y2-y1) / (img_bgr.shape[0]*img_bgr.shape[1])
        scale      = float(np.exp(-((area_ratio - 0.20)**2) / (2*0.12**2)))

        ar    = min((x2-x1),(y2-y1)) / max((x2-x1),(y2-y1), 1e-6)
        align = float(np.exp(-((ar - 0.85)**2) / (2*0.15**2)))

        border = 10
        if roi.shape[0] > 2*border and roi.shape[1] > 2*border:
            inner = roi[border:-border, border:-border]
            occ   = float((inner.size//3) / max(roi.size//3, 1))
        else:
            occ = 0.5

        return QualityScores(blur=blur, scale=scale, alignment=align, occlusion=occ)

    def select_best(
        self,
        detections: List[Dict],
        img_bgr: np.ndarray,
    ) -> Optional[Dict]:
        """품질 통합 점수로 최적 detection 선택."""
        if not detections:
            return None
        best, best_score = None, -1.0
        for d in detections:
            q = d.get("quality")
            if q is None:
                q = self.compute_rule_quality(img_bgr, d["bbox"])
                d["quality"] = q
            score = (d["conf"] ** self.ALPHA) * (q.total ** self.BETA)
            if score > best_score:
                best_score, best = score, d
        return best


# ══════════════════════════════════════════════════════════════════════════════
# 3. AdaptiveROIRefiner  —  품질 기반 bbox 정제
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveROIRefiner:
    """
    Step 1. Quality-adaptive margin
      - quality.total < 0.5 → bbox를 15% 확장
      - quality.total ≥ 0.5 → 5% 확장

    Step 2. Inscribed-circle 기반 정방형 정규화
      - min(w, h)를 기준으로 정방형 크롭, 중심 유지

    Step 3. Edge-map 방향 보정 (최대 ±30°)
      - Sobel edge의 PCA 주축으로 손 기울기 추정
      - Stage B에서 활용할 angle 정보 제공
    """

    LOW_QUALITY_MARGIN  = 0.15
    HIGH_QUALITY_MARGIN = 0.05
    MAX_CORRECT_DEG     = 30.0

    def refine(
        self,
        detection: PalmDetection,
        image_bgr: np.ndarray,
    ) -> PalmDetection:
        q    = detection.quality or QualityScores(0.5, 0.5, 0.5, 0.5)
        bbox = detection.bbox
        H, W = image_bgr.shape[:2]

        bbox = self._apply_margin(bbox, q, W, H)
        bbox = self._square_normalize(bbox, W, H)
        mask = self._build_mask(image_bgr, bbox)

        return PalmDetection(
            bbox=bbox,
            confidence=detection.confidence,
            quality=q,
            hand_mask=mask,
            landmarks=None,
        )

    def _apply_margin(self, bbox, q, W, H):
        x1, y1, x2, y2 = bbox
        bw, bh = x2-x1, y2-y1
        m = self.LOW_QUALITY_MARGIN if q.total < 0.5 else self.HIGH_QUALITY_MARGIN
        mx, my = int(bw*m), int(bh*m)
        return (max(0,x1-mx), max(0,y1-my), min(W,x2+mx), min(H,y2+my))

    @staticmethod
    def _square_normalize(bbox, W, H):
        x1, y1, x2, y2 = bbox
        bw, bh = x2-x1, y2-y1
        side   = max(bw, bh)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        half   = side//2
        return (max(0,cx-half), max(0,cy-half), min(W,cx+half), min(H,cy+half))

    def estimate_orientation(self, image_bgr: np.ndarray, bbox: Tuple) -> float:
        """Edge PCA로 손 기울기 추정 (도 단위, -90~90). Stage B 참고용."""
        x1, y1, x2, y2 = map(int, bbox)
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag  = np.sqrt(gx**2 + gy**2)
        thresh = mag.mean() + mag.std()
        strong = mag > thresh
        if strong.sum() < 50:
            return 0.0
        pts = np.column_stack(np.where(strong)).astype(np.float32)
        pts -= pts.mean(axis=0)
        cov  = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        principal  = vecs[:, np.argmax(vals)]
        angle_deg  = float(np.degrees(np.arctan2(principal[0], principal[1])))
        return angle_deg if abs(angle_deg) <= self.MAX_CORRECT_DEG else 0.0

    @staticmethod
    def _build_mask(image_bgr: np.ndarray, bbox: Tuple) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0,x1); y1 = max(0,y1)
        x2 = min(image_bgr.shape[1],x2); y2 = min(image_bgr.shape[0],y2)
        H, W = image_bgr.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        if x2 <= x1 or y2 <= y1:
            return mask
        crop = image_bgr[y1:y2, x1:x2]
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        m    = cv2.inRange(hsv,
                           np.array([0,15,50], np.uint8),
                           np.array([25,255,255], np.uint8))
        m    = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
        mask[y1:y2, x1:x2] = m
        return mask


# ══════════════════════════════════════════════════════════════════════════════
# 4. PalmROIExtractor  —  4단계 파이프라인 통합 인터페이스
# ══════════════════════════════════════════════════════════════════════════════

class PalmROIExtractor:
    """
    제안 Stage A 전체 파이프라인의 단일 엔트리포인트.

    가중치 있음  : LandmarkFreePalmDetector CNN 사용 (full proposed pipeline)
    가중치 없음  : MediaPipe(설치된 경우) → Skin-color 순으로 fallback
                   + QualityAwareROIEstimator(규칙 기반) + AdaptiveROIRefiner 동작

    [Stage B/C 인터페이스]
      detect(image_bgr) → PalmDetection
        .bbox          : (x1,y1,x2,y2)
        .confidence    : float
        .quality       : QualityScores (blur/scale/alignment/occlusion)
        .hand_mask     : np.ndarray (Stage B 마스크 재사용)
        .landmarks     : None (landmark-free)
    """

    WEIGHT_PATH = Path(__file__).parent.parent / "checkpoints" / "stage_a" / "detector.pt"

    def __init__(self, device: str = "cpu", input_size: int = 320):
        self.device      = device
        self.input_size  = input_size
        self.net: Optional[LandmarkFreePalmDetector] = None
        self.estimator   = QualityAwareROIEstimator(input_size)
        self.refiner     = AdaptiveROIRefiner()
        self._mp         = False
        self._mp_hands   = None
        self._net_loaded = False

        self._try_load_weights()
        if not self._net_loaded:
            self._init_mediapipe()
        self._print_status()

    def _try_load_weights(self):
        if not self.WEIGHT_PATH.exists():
            return
        try:
            self.net = LandmarkFreePalmDetector(self.input_size)
            state    = torch.load(str(self.WEIGHT_PATH), map_location="cpu")
            self.net.load_state_dict(state)
            self.net.eval().to(self.device)
            self._net_loaded = True
        except Exception as e:
            print(f"[StageA] 가중치 로드 실패 ({e}). Fallback 사용.")
            self.net = None

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            self._mp_hands = mp.solutions.hands.Hands(
                static_image_mode=True, max_num_hands=1,
                min_detection_confidence=0.4,
            )
            self._mp = True
        except Exception:
            self._mp = False

    def _print_status(self):
        if self._net_loaded:
            print(f"[StageA] ✓ LandmarkFreePalmDetector "
                  f"({self.net.param_size_mb:.1f} MB, device={self.device})")
        elif self._mp:
            print("[StageA] ⚠ CNN 가중치 없음 → MediaPipe fallback")
        else:
            print("[StageA] ⚠ CNN 가중치 없음 + MediaPipe 미설치 → Skin-color fallback")

    # ── 메인 인터페이스 ────────────────────────────────────────────────
    def detect(self, image_bgr: np.ndarray) -> Optional[PalmDetection]:
        """단일 이미지에서 최적 PalmDetection 반환. None = 검출 실패."""
        raw = self._raw_detect(image_bgr)
        if raw is None:
            return None
        if raw.quality is None:
            raw = PalmDetection(
                bbox=raw.bbox, confidence=raw.confidence,
                quality=self.estimator.compute_rule_quality(image_bgr, raw.bbox),
                hand_mask=raw.hand_mask,
            )
        return self.refiner.refine(raw, image_bgr)

    # __call__ 호환 (기존 PalmLocalizer 동작 유지)
    def __call__(self, image_bgr: np.ndarray) -> Optional[PalmDetection]:
        return self.detect(image_bgr)

    def _raw_detect(self, image_bgr: np.ndarray) -> Optional[PalmDetection]:
        if self._net_loaded:
            return self._net_detect(image_bgr)
        if self._mp:
            return self._mp_detect(image_bgr)
        return self._skin_detect(image_bgr)

    def _net_detect(self, image_bgr: np.ndarray) -> Optional[PalmDetection]:
        H, W = image_bgr.shape[:2]
        inp  = cv2.resize(image_bgr, (self.input_size, self.input_size))
        inp  = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        t    = torch.from_numpy(inp).permute(2,0,1).float().unsqueeze(0) / 255.0
        t    = t.to(self.device)
        scale_x, scale_y = W / self.input_size, H / self.input_size
        with torch.no_grad():
            det_maps, qual_maps = self.net(t)
        results = self.estimator.decode(det_maps, qual_maps,
                                        (self.input_size, self.input_size))
        dets = results[0] if results else []
        best = self.estimator.select_best(dets, image_bgr)
        if best is None:
            return None
        x1, y1, x2, y2 = best["bbox"]
        bbox_orig = (int(x1*scale_x), int(y1*scale_y),
                     int(x2*scale_x), int(y2*scale_y))
        return PalmDetection(bbox=bbox_orig, confidence=best["conf"],
                             quality=best["quality"])

    def _mp_detect(self, image_bgr: np.ndarray) -> Optional[PalmDetection]:
        H, W  = image_bgr.shape[:2]
        rgb   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res   = self._mp_hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None
        lms  = res.multi_hand_landmarks[0]
        pts  = np.array([[lm.x*W, lm.y*H] for lm in lms.landmark])
        x1, y1 = pts.min(axis=0).astype(int)
        x2, y2 = pts.max(axis=0).astype(int)
        pad  = int(max(x2-x1, y2-y1) * 0.08)
        bbox = (max(0,x1-pad), max(0,y1-pad), min(W,x2+pad), min(H,y2+pad))
        hull = cv2.convexHull(pts.astype(np.int32))
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 255)
        return PalmDetection(bbox=bbox, confidence=0.85, hand_mask=mask)

    def _skin_detect(self, image_bgr: np.ndarray) -> Optional[PalmDetection]:
        H, W   = image_bgr.shape[:2]
        hsv    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask   = cv2.inRange(hsv, np.array([0,15,50],np.uint8),
                             np.array([25,255,255],np.uint8))
        ycbcr  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        mask2  = cv2.inRange(ycbcr, np.array([0,133,77],np.uint8),
                             np.array([255,185,140],np.uint8))
        mask   = cv2.bitwise_and(mask, mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 500:
            return None
        x, y, bw, bh = cv2.boundingRect(c)
        pad = int(max(bw,bh)*0.05)
        bbox = (max(0,x-pad), max(0,y-pad), min(W,x+bw+pad), min(H,y+bh+pad))
        return PalmDetection(bbox=bbox, confidence=0.60)

    @property
    def model_size_mb(self) -> float:
        if self.net is not None:
            p = self.WEIGHT_PATH
            return os.path.getsize(str(p))/1024/1024 if p.exists() else self.net.param_size_mb
        return 0.9  # MediaPipe palm_detection_lite 참고값


# 하위 호환 alias — main.py의 PalmLocalizer 임포트 유지
PalmLocalizer = PalmROIExtractor


# ══════════════════════════════════════════════════════════════════════════════
# 5. Stage A 학습 — Dataset + Loss + Trainer
# ══════════════════════════════════════════════════════════════════════════════

class StageADataset(Dataset):
    """
    Stage A 학습 데이터셋 — 캐시 기반 최적화 버전

    [속도 최적화 전략]
    1. build_training_cache() 를 학습 전 1회 실행:
       - 원본 이미지(3264×2448 등) → 640px JPEG로 저장  (I/O 20-30배 단축)
       - MediaPipe pseudo-GT bbox 사전 계산 (epoch마다 재실행 → 1회로)
       - quality pseudo-label 사전 계산
    2. __getitem__: 640px JPEG 읽기 → augment → 320px resize
       (매 호출마다 원본 읽기/MediaPipe 없음 → ~4ms/장, 기존 ~100ms+)
    3. num_workers ≥ 2 안전 사용 (MediaPipe 없으므로 다중 프로세스 OK)

    [폴백]
    캐시 없는 경우 기존 방식(on-the-fly GT) 자동 사용 (느리지만 동작 보장)
    """

    CACHE_IMG_SUBDIR  = "images"           # 캐시 이미지 저장
    CACHE_GT_FILENAME = "gt_cache.json"    # GT bbox + quality 저장
    PREFETCH_SIDE     = 320                # 캐시 이미지 크기 = 학습 해상도와 동일
                                           # (640→320 변경: resize 스텝 제거로 ~3배 속도 향상)
                                           # ※ 기존 640px 캐시 사용 시 rebuild_cache=True 필요
    JPEG_QUALITY      = 92                 # 캐시 JPEG 화질

    def __init__(
        self,
        image_paths:  List[str],
        input_size:   int  = 320,
        augment:      bool = True,
        cache_dir:    Optional[str] = None,  # None → on-the-fly (느림)
    ):
        self.input_size = input_size
        self.augment    = augment
        self.transform  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        # ── 캐시 로드 시도 ────────────────────────────────────────────
        self._cache_gt: Optional[Dict] = None
        self._img_dir:  Optional[Path] = None

        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            gt_path   = cache_dir / self.CACHE_GT_FILENAME
            img_dir   = cache_dir / self.CACHE_IMG_SUBDIR
            if gt_path.exists() and img_dir.exists():
                try:
                    with open(gt_path, "r") as f:
                        self._cache_gt = json.load(f)

                    # sanity check (중요)
                    if not isinstance(self._cache_gt, dict) or len(self._cache_gt) == 0:
                        raise ValueError("empty or invalid cache")

                except Exception as e:
                    print(f"[StageA] ⚠ 캐시 손상 감지 → 자동 무시: {e}")
                    self._cache_gt = None
                    self._img_dir = None
                self._img_dir = img_dir
                print(f"[StageA] 캐시 로드: {len(self._cache_gt)} GT, "
                      f"이미지 디렉터리: {img_dir}")
            else:
                print(f"[StageA] ⚠ 캐시 없음 ({cache_dir}). "
                      f"on-the-fly GT 사용 (느림). "
                      f"먼저 build_training_cache() 를 실행하세요.")

        # ── 유효 경로 필터링 ──────────────────────────────────────────
        if self._cache_gt is not None:
            # 캐시에 GT가 있는 것만 사용
            self.paths = [p for p in image_paths
                          if self._cache_gt.get(str(p)) is not None]
        else:
            self.paths = [p for p in image_paths if os.path.exists(str(p))]
            self._init_online_gt()   # MediaPipe / skin-color 초기화

        self.estimator = QualityAwareROIEstimator(input_size)
        print(f"[StageA] {len(self.paths)} images  "
              f"(cache={'ON' if self._cache_gt else 'OFF'}, augment={augment})")

    # ── 캐시 없을 때 GT 생성기 초기화 ────────────────────────────────
    def _init_online_gt(self):
        self._mp = False
        try:
            import mediapipe as mp
            self._mp_hands = mp.solutions.hands.Hands(
                static_image_mode=True, max_num_hands=1,
                min_detection_confidence=0.35,
            )
            self._mp = True
        except Exception:
            self._mp = False

    # ─────────────────────────────────────────────────────────────────
    # ★ 핵심 정적 메서드: 오프라인 캐시 빌드 (학습 전 1회 실행)
    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def build_training_cache(
        image_paths:  List[str],
        cache_dir:    str,
        prefetch_side: int = 320,   # 학습 해상도와 동일하게 저장 → __getitem__에서 resize 불필요
        jpeg_quality:  int = 92,
    ) -> str:
        """
        학습 전 1회 실행. 이후 모든 epoch은 캐시에서 로드.

        수행 내용:
          1. 각 이미지를 prefetch_side(640)px으로 리사이즈 → JPEG 저장
          2. MediaPipe 또는 skin-color로 GT bbox 계산 (원본 크기 기준)
          3. quality pseudo-label 계산 (640px 이미지 기준)
          4. gt_cache.json 저장

        반환: cache_dir (str)
        """
        cache_dir = Path(cache_dir)
        img_dir   = cache_dir / StageADataset.CACHE_IMG_SUBDIR
        gt_path   = cache_dir / StageADataset.CACHE_GT_FILENAME
        img_dir.mkdir(parents=True, exist_ok=True)

        # 이미 완성된 캐시면 재사용
        if gt_path.exists():
            with open(gt_path) as f:
                existing = json.load(f)
            if len(existing) >= len(image_paths) * 0.95:
                print(f"[Cache] 기존 캐시 재사용 ({len(existing)} GT) → {cache_dir}")
                return str(cache_dir)
            print(f"[Cache] 기존 캐시 불완전 ({len(existing)}/{len(image_paths)}). 재생성...")

        # MediaPipe 초기화 (가능하면 사용)
        mp_hands = None
        try:
            import mediapipe as mp
            mp_hands = mp.solutions.hands.Hands(
                static_image_mode=True, max_num_hands=1,
                min_detection_confidence=0.35,
            )
            print("[Cache] MediaPipe 사용 (정확한 GT)")
        except Exception:
            print("[Cache] MediaPipe 없음 → skin-color GT 사용")

        gt_cache    = {}
        fail_count  = 0
        total       = len(image_paths)

        for i, raw_path in enumerate(image_paths):
            img_bgr = cv2.imread(str(raw_path))
            if img_bgr is None:
                fail_count += 1
                continue

            H_orig, W_orig = img_bgr.shape[:2]

            # ① 다운샘플 (MediaPipe/skin-color 처리 속도 대폭 향상)
            scale  = min(prefetch_side / max(H_orig, W_orig), 1.0)
            W_sm   = max(1, int(W_orig * scale))
            H_sm   = max(1, int(H_orig * scale))
            img_sm = cv2.resize(img_bgr, (W_sm, H_sm), interpolation=cv2.INTER_AREA)

            # ② GT bbox 계산 (소형 이미지 기준)
            bbox_sm: Optional[Tuple] = None

            if mp_hands is not None:
                rgb = cv2.cvtColor(img_sm, cv2.COLOR_BGR2RGB)
                res = mp_hands.process(rgb)
                if res.multi_hand_landmarks:
                    lms  = res.multi_hand_landmarks[0]
                    pts  = np.array([[lm.x * W_sm, lm.y * H_sm]
                                     for lm in lms.landmark])
                    # ★ numpy int64 → Python int (JSON 직렬화 호환)
                    x1, y1 = (int(v) for v in pts.min(axis=0))
                    x2, y2 = (int(v) for v in pts.max(axis=0))
                    pad    = int(max(x2 - x1, y2 - y1) * 0.08)
                    bbox_sm = (max(0, x1-pad), max(0, y1-pad),
                               min(W_sm, x2+pad), min(H_sm, y2+pad))

            if bbox_sm is None:
                # skin-color fallback
                hsv  = cv2.cvtColor(img_sm, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv,
                                   np.array([0, 15, 50], np.uint8),
                                   np.array([25, 255, 255], np.uint8))
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
                cnts, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    c = max(cnts, key=cv2.contourArea)
                    if cv2.contourArea(c) > 300:
                        x, y, bw, bh = cv2.boundingRect(c)
                        bbox_sm = (max(0, x-4), max(0, y-4),
                                   min(W_sm, x+bw+4), min(H_sm, y+bh+4))

            if bbox_sm is None:
                fail_count += 1
                continue

            # ③ quality pseudo-label (소형 이미지 기준, 빠름)
            q = QualityAwareROIEstimator.compute_rule_quality(img_sm, bbox_sm)

            # ④ bbox를 소형 이미지 기준 그대로 저장
            #    (학습 시 640px 캐시 이미지를 읽으므로 좌표계 일치)
            #    ★ int() 명시적 변환: numpy int64 → Python int (JSON 직렬화 필수)
            bbox_for_cache = [int(v) for v in bbox_sm]

            # ⑤ 640px 이미지 JPEG 저장 (이미 있으면 스킵)
            safe_name = Path(raw_path).name.replace(" ", "_")
            cache_img_path = img_dir / safe_name
            if not cache_img_path.exists():
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                cv2.imwrite(str(cache_img_path), img_sm, encode_params)

            gt_cache[str(raw_path)] = {
                "cache_img": str(cache_img_path),   # 640px JPEG 경로
                "bbox":      bbox_for_cache,          # [x1, y1, x2, y2] Python int
                "quality":   [float(q.blur), float(q.scale),
                              float(q.alignment), float(q.occlusion)],
            }

            if (i + 1) % 500 == 0 or (i + 1) == total:
                print(f"  [{i+1:>5}/{total}] cached={len(gt_cache)} fail={fail_count}")

        if mp_hands is not None:
            mp_hands.close()

        # ⑥ JSON 저장
        # NumpyEncoder: 만일의 numpy 타입 잔재 방어
        class _NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(gt_path, "w") as f:
            json.dump(gt_cache, f, default=lambda x: x.item() if hasattr(x, "item") else x.tolist())

        print(f"\n[Cache] 완료: {len(gt_cache)} GT / {total} 이미지 "
              f"({fail_count} 실패) → {cache_dir}")
        return str(cache_dir)

    # ─────────────────────────────────────────────────────────────────
    # __len__ / __getitem__
    # ─────────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path  = str(self.paths[idx])

        # ── 캐시 경로 (빠름) ─────────────────────────────────────────
        if self._cache_gt is not None:
            entry = self._cache_gt.get(path)
            if entry is None:
                return self._empty()
            cache_img_path = entry["cache_img"]
            img_bgr = cv2.imread(cache_img_path)   # 640px JPEG → ~3ms
            if img_bgr is None:
                return self._empty()
            bbox    = tuple(entry["bbox"])
            q_vals  = entry["quality"]

        # ── on-the-fly 경로 (폴백, 느림) ─────────────────────────────
        else:
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                return self._empty()
            H_orig, W_orig = img_bgr.shape[:2]

            # 큰 이미지 먼저 다운샘플 (augment/resize 속도 향상)
            scale = min(self.PREFETCH_SIDE / max(H_orig, W_orig), 1.0)
            if scale < 1.0:
                img_bgr = cv2.resize(img_bgr,
                                     (int(W_orig * scale), int(H_orig * scale)),
                                     interpolation=cv2.INTER_AREA)
            bbox = self._get_gt_online(img_bgr)
            if bbox is None:
                return self._empty()
            q_vals = None

        # ── Augmentation (640px 이미지 기준 → 빠름) ─────────────────
        H, W = img_bgr.shape[:2]
        if self.augment:
            img_bgr, bbox = self._augment(img_bgr, bbox, H, W)
            H, W = img_bgr.shape[:2]

        # ── quality tensor ────────────────────────────────────────────
        if q_vals is not None:
            q_tensor = torch.tensor(q_vals, dtype=torch.float32)
        else:
            q  = self.estimator.compute_rule_quality(img_bgr, bbox)
            q_tensor = q.as_tensor()

        # ── 최종 리사이즈 + bbox 정규화 ──────────────────────────────────
        # 캐시를 input_size=320으로 저장한 경우 resize 불필요 (3배 속도 향상)
        if H == self.input_size and W == self.input_size:
            img_res = img_bgr   # resize 스킵
        else:
            img_res = cv2.resize(img_bgr, (self.input_size, self.input_size),
                                 interpolation=cv2.INTER_AREA)
        sx  = self.input_size / W
        sy  = self.input_size / H
        gt_norm = torch.tensor(
            [bbox[0]*sx, bbox[1]*sy, bbox[2]*sx, bbox[3]*sy],
            dtype=torch.float32,
        )
        img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        return self.transform(img_rgb), gt_norm, q_tensor

    # ── 헬퍼 ─────────────────────────────────────────────────────────
    def _get_gt_online(self, img_bgr: np.ndarray) -> Optional[Tuple]:
        """on-the-fly GT (캐시 없을 때 폴백)."""
        H, W = img_bgr.shape[:2]
        if self._mp:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            res = self._mp_hands.process(rgb)
            if res.multi_hand_landmarks:
                lms = res.multi_hand_landmarks[0]
                pts = np.array([[lm.x*W, lm.y*H] for lm in lms.landmark])
                x1, y1 = pts.min(axis=0).astype(int)
                x2, y2 = pts.max(axis=0).astype(int)
                pad = int(max(x2-x1, y2-y1) * 0.08)
                return (max(0,x1-pad), max(0,y1-pad),
                        min(W,x2+pad), min(H,y2+pad))
        # skin-color fallback
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        m   = cv2.inRange(hsv, np.array([0,15,50], np.uint8),
                          np.array([25,255,255], np.uint8))
        m   = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 300:
            return None
        x, y, bw, bh = cv2.boundingRect(c)
        return (max(0,x-4), max(0,y-4), min(W,x+bw+4), min(H,y+bh+4))

    @staticmethod
    def _augment(img, bbox, H, W):
        x1, y1, x2, y2 = bbox
        # 좌우 플립
        if np.random.rand() < 0.3:
            img = cv2.flip(img, 1)
            x1, x2 = W - x2, W - x1
        # 밝기/대비 (numpy 연산, 빠름)
        alpha = np.random.uniform(0.75, 1.25)
        beta  = np.random.randint(-25, 25)
        img   = np.clip(img.astype(np.float32) * alpha + beta,
                        0, 255).astype(np.uint8)
        # 회전 (640px 기준 → 원본 대비 17배 빠름)
        angle = np.random.uniform(-12, 12)
        M = cv2.getRotationMatrix2D((W // 2, H // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (W, H),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        pts = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
        pts = cv2.transform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)
        x1n, y1n = pts.min(axis=0).astype(int)
        x2n, y2n = pts.max(axis=0).astype(int)
        return img, (max(0,x1n), max(0,y1n), min(W,x2n), min(H,y2n))

    def _empty(self):
        s = self.input_size
        return (torch.zeros(3, s, s),
                torch.tensor([-1., -1., -1., -1.]),
                torch.zeros(4))


class StageALoss(nn.Module):
    """
    Detection + Quality 통합 손실 — 벡터화 최적화 버전

    [변경사항]
    - Focal Loss : B 루프 제거 → 배치 전체 한 번에 계산 (B×H×W 텐서 연산)
    - CIoU Loss  : positive cell 마스크로 배치 내 유효 샘플만 선택 후 계산
    - Quality    : positive cell + 배치 선택 후 계산
    - valid_mask : gt[0] < 0 인 빈 샘플 자동 제외

    [손실 구성]
    L = L_focal + λ_bbox × L_CIoU + λ_quality × L_BCE_quality
    """

    def __init__(self, lambda_bbox=5.0, lambda_quality=0.05,
                 focal_alpha=0.25, focal_gamma=2.0):
        """
        [lambda_quality 변경: 1.0 → 0.05]
        quality는 보조 손실. conf/bbox가 학습된 뒤 자연스럽게 따라옴.
        lambda_quality=1.0이면 quality 그래디언트가 conf(0.06)보다 수십 배 커져
        backbone을 파괴하고 전체 수렴 실패를 유발.
        """
        super().__init__()
        self.lambda_bbox    = lambda_bbox
        self.lambda_quality = lambda_quality
        self.focal_alpha    = focal_alpha
        self.focal_gamma    = focal_gamma

    def forward(self, det_maps, qual_maps, gt_bboxes, gt_quality,
                input_size, strides):
        device     = gt_bboxes.device
        B          = gt_bboxes.shape[0]
        valid_mask = (gt_bboxes[:, 0] >= 0)   # (B,) — 유효 GT 샘플

        total_conf = torch.zeros(1, device=device)
        total_bbox = torch.zeros(1, device=device)
        total_qual = torch.zeros(1, device=device)
        n_pos      = 0

        for det_map, qual_map, stride in zip(det_maps, qual_maps, strides):
            _, _, H, W = det_map.shape

            # ── 셀 중심 좌표 (H, W) ────────────────────────────────────
            gy, gx = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing="ij",
            )
            cx = (gx + 0.5) * stride   # (H, W)
            cy = (gy + 0.5) * stride   # (H, W)

            # ── GT bbox 브로드캐스트: (B, 1, 1) ────────────────────────
            x1g = gt_bboxes[:, 0].view(B, 1, 1)
            y1g = gt_bboxes[:, 1].view(B, 1, 1)
            x2g = gt_bboxes[:, 2].view(B, 1, 1)
            y2g = gt_bboxes[:, 3].view(B, 1, 1)

            # in_box: (B, H, W) — positive cell 여부
            in_box = (
                (cx > x1g) & (cx < x2g) &
                (cy > y1g) & (cy < y2g) &
                valid_mask.view(B, 1, 1)
            )

            # ══════════════════════════════════════════════════════════
            # ① Focal Loss — 배치 전체 벡터 연산 (B×H×W)
            # ══════════════════════════════════════════════════════════
            conf_pred = det_map[:, 0]          # (B, H, W)
            conf_tgt  = in_box.float()          # (B, H, W)
            p         = torch.sigmoid(conf_pred)
            ce        = F.binary_cross_entropy_with_logits(
                conf_pred, conf_tgt, reduction="none")   # (B, H, W)
            pt  = torch.where(conf_tgt == 1, p, 1 - p)
            a_t = torch.where(conf_tgt == 1,
                              torch.full_like(conf_tgt, self.focal_alpha),
                              torch.full_like(conf_tgt, 1 - self.focal_alpha))
            focal = a_t * (1 - pt) ** self.focal_gamma * ce  # (B, H, W)
            # 유효 샘플만 포함 (invalid GT 제외)
            total_conf = total_conf + (focal * valid_mask.view(B, 1, 1)).mean()

            # ══════════════════════════════════════════════════════════
            # ② CIoU + Quality — positive cell 있는 샘플만 처리
            #    (Python loop이지만 유효 샘플로만 한정 → 실질적 부담 작음)
            # ══════════════════════════════════════════════════════════
            eps = 1e-6
            for b in range(B):
                if not valid_mask[b] or not in_box[b].any():
                    continue
                mask_b = in_box[b]     # (H, W)
                x1, y1, x2, y2 = gt_bboxes[b]

                # predicted LTRB distances at positive cells
                pl  = torch.exp(det_map[b, 1].clamp(-6, 6)) * stride
                pt_ = torch.exp(det_map[b, 2].clamp(-6, 6)) * stride
                pr  = torch.exp(det_map[b, 3].clamp(-6, 6)) * stride
                pb  = torch.exp(det_map[b, 4].clamp(-6, 6)) * stride

                # GT LTRB at positive cells
                gl = (cx - x1)[mask_b];  gt_l = (cy - y1)[mask_b]
                gr = (x2 - cx)[mask_b];  gb   = (y2 - cy)[mask_b]
                pl = pl[mask_b]; pt_ = pt_[mask_b]
                pr = pr[mask_b]; pb  = pb[mask_b]

                # CIoU
                pw   = pl + pr;  ph  = pt_ + pb
                gw   = gl + gr;  gh  = gt_l + gb
                iw   = (torch.minimum(pl, gl) + torch.minimum(pr, gr)).clamp(0)
                ih   = (torch.minimum(pt_, gt_l) + torch.minimum(pb, gb)).clamp(0)
                inter = iw * ih
                union = pw*ph + gw*gh - inter + eps
                iou   = inter / union
                c_w   = torch.maximum(pl, gl) + torch.maximum(pr, gr)
                c_h   = torch.maximum(pt_, gt_l) + torch.maximum(pb, gb)
                c2    = c_w**2 + c_h**2 + eps
                dcx   = (pr - pl) / 2 - (gr - gl) / 2
                dcy   = (pb - pt_) / 2 - (gb - gt_l) / 2
                rho2  = dcx**2 + dcy**2
                v     = (4 / math.pi**2) * (
                    torch.atan(gw / (gh + eps)) - torch.atan(pw / (ph + eps))
                ) ** 2
                with torch.no_grad():
                    alpha_v = v / (1 - iou + v + eps)
                ciou = (1 - iou + rho2 / c2 + alpha_v * v).mean()
                total_bbox = total_bbox + ciou

                # Quality BCE — bbox 중심에 가장 가까운 단일 셀만 감독
                # [이유] 모든 positive cell에 같은 per-image quality target을 주면
                #        공간 위치별로 다른 예측을 일관되지 않게 학습 → 발산
                #        가장 centerness가 높은 1개 셀만 사용해 안정화.
                bcx = (x1 + x2) / 2.0
                bcy = (y1 + y2) / 2.0
                dist2_hw = (cx - bcx) ** 2 + (cy - bcy) ** 2  # (H, W)

                # positive cells 중 중심과 가장 가까운 셀 인덱스
                dist2_pos = dist2_hw[mask_b]                   # (n_pos_cells,)
                best_idx  = dist2_pos.argmin()

                # 안전한 flatten 인덱싱 (2D mask → 1D)
                C_q = qual_map.shape[1]
                qual_flat = qual_map[b].reshape(C_q, -1)       # (4, H*W)
                mask_flat = mask_b.reshape(-1)                  # (H*W,) bool
                best_cell = qual_flat[:, mask_flat][:, best_idx]  # (4,)

                q_pred = torch.sigmoid(best_cell)               # (4,)  ← 단일 셀
                # target clamp [0.05, 0.95]: BCE 포화 방지
                q_tgt  = gt_quality[b].clamp(0.05, 0.95)       # (4,)
                total_qual = total_qual + F.binary_cross_entropy(
                    q_pred.unsqueeze(0), q_tgt.unsqueeze(0))    # scalar
                n_pos += 1

        if n_pos > 0:
            total_bbox = total_bbox / n_pos
            total_qual = total_qual / n_pos

        loss = (total_conf
                + self.lambda_bbox    * total_bbox
                + self.lambda_quality * total_qual)
        return {
            "total":   loss,
            "conf":    total_conf.detach(),
            "bbox":    total_bbox.detach(),
            "quality": total_qual.detach(),
        }


class StageATrainer:
    """
    LandmarkFreePalmDetector 학습 루프 — 최적화 버전

    [최적화]
    - DataLoader num_workers=2 (캐시 기반이면 멀티프로세싱 안전)
    - persistent_workers=True (worker 재생성 오버헤드 제거)
    - pin_memory=False (MPS 비호환)
    - build_training_cache() 사전 호출 + cache_dir 연결

    [스케줄러]
    - Linear warmup (warmup_epochs)
    - CosineAnnealingLR (나머지 epochs)
    """

    SAVE_PATH  = Path(__file__).parent.parent / "checkpoints" / "stage_a" / "detector.pt"
    CACHE_DIR  = Path(__file__).parent.parent / "checkpoints" / "stage_a" / "data_cache"

    def __init__(self, device="cpu", input_size=320, lr=3e-4, weight_decay=1e-4):
        self.device     = device
        self.input_size = input_size
        self.model      = LandmarkFreePalmDetector(input_size).to(device)
        self.loss_fn    = StageALoss()
        self.optimizer  = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = self.SAVE_PATH.parent / "train.log"

    def _log(self, msg: str):
        print(msg)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def prepare_dataset(
        self,
        image_paths: List[str],
        augment:     bool = True,
        rebuild_cache: bool = False,
    ) -> "StageADataset":
        """
        GT 캐시 빌드 (없거나 rebuild_cache=True일 때) 후 Dataset 반환.

        [속도 예상]
        캐시 빌드: 이미지 수 × ~100ms (1회 only)
        학습 epoch: 이미지 수 × ~4ms  (vs 기존 ~100ms+)
        """
        cache_dir = str(self.CACHE_DIR)

        # 캐시 빌드 여부 판단
        gt_path = self.CACHE_DIR / StageADataset.CACHE_GT_FILENAME
        if rebuild_cache or not gt_path.exists():
            self._log(f"[Trainer] GT 캐시 빌드 시작 ({len(image_paths)} 이미지)...")
            t0 = time.perf_counter()
            StageADataset.build_training_cache(image_paths, cache_dir)
            elapsed = time.perf_counter() - t0
            self._log(f"[Trainer] 캐시 빌드 완료: {elapsed:.0f}초")
        else:
            self._log(f"[Trainer] 기존 캐시 재사용: {gt_path}")

        return StageADataset(
            image_paths,
            input_size=self.input_size,
            augment=augment,
            cache_dir=cache_dir,
        )

    def fit(
        self,
        image_paths:    List[str],      # 이미지 경로 목록 (DataLoader 내부 생성)
        batch_size:     int  = 16,
        epochs:         int  = 50,
        warmup_epochs:  int  = 3,
        loss_patience:  int  = 10,
        min_delta:      float = 1e-3,
        num_workers:    int  = 2,       # 캐시 기반이면 멀티프로세싱 안전
        rebuild_cache:  bool = False,
    ) -> str:
        """
        [변경된 인터페이스]
        image_paths 를 직접 받아 내부에서 캐시 빌드 + DataLoader 생성.
        (기존: DataLoader 외부 생성 → num_workers 조정 불가)
        """
        # ── 캐시 빌드 + Dataset ────────────────────────────────────────
        train_ds = self.prepare_dataset(image_paths, augment=True,
                                        rebuild_cache=rebuild_cache)
        if len(train_ds) == 0:
            self._log("[Error] 유효 이미지 없음. 경로를 확인하세요.")
            return str(self.SAVE_PATH)

        # ── DataLoader ─────────────────────────────────────────────────
        # macOS 주의: Python 3.8+ 기본 multiprocessing context = spawn
        # → 배치마다 worker 직렬화 오버헤드 ~0.7s (num_workers=2 오히려 느려짐)
        # 해결: macOS에서 num_workers=0 (단일 프로세스, IPC 없음)
        #       또는 multiprocessing_context='fork' (일부 라이브러리 비호환 주의)
        import platform
        is_macos = platform.system() == "Darwin"

        if is_macos:
            # macOS: spawn 오버헤드 회피 → 단일 프로세스 or fork
            use_workers = 0
            mp_ctx = None
        else:
            use_workers = num_workers if train_ds._cache_gt is not None else 0
            mp_ctx = None

        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=use_workers,
            pin_memory=False,   # MPS 비호환
            drop_last=True,
        )
        if use_workers > 0:
            loader_kwargs["persistent_workers"] = True
            if mp_ctx:
                loader_kwargs["multiprocessing_context"] = mp_ctx
            loader_kwargs["prefetch_factor"] = 2

        loader = DataLoader(train_ds, **loader_kwargs)
        self._log(f"  DataLoader: workers={use_workers}"
                  + (" (macOS → 단일프로세스)" if is_macos else ""))

        # ── 스케줄러 ───────────────────────────────────────────────────
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(epochs - warmup_epochs, 1),
        )
        strides    = LandmarkFreePalmDetector.STRIDES
        best_loss  = float("inf")
        no_improve = 0
        best_path  = str(self.SAVE_PATH)

        self._log("=" * 60)
        self._log(f"  Stage A Training")
        self._log(f"  Model: {self.model.param_size_mb:.1f} MB | device={self.device}")
        self._log(f"  Dataset: {len(train_ds)} images | batch={batch_size} "
                  f"| workers={use_workers}")
        self._log(f"  Epochs: {epochs} | warmup={warmup_epochs} "
                  f"| patience={loss_patience}")
        self._log("=" * 60)

        for ep in range(1, epochs + 1):
            self.model.train()
            ep_vals  = {"total": 0., "conf": 0., "bbox": 0., "quality": 0.}
            n_batches = 0
            t_ep = time.perf_counter()

            for imgs, gt_bboxes, gt_quality in loader:
                imgs       = imgs.to(self.device)
                gt_bboxes  = gt_bboxes.to(self.device)
                gt_quality = gt_quality.to(self.device)

                det_maps, qual_maps = self.model(imgs)
                losses = self.loss_fn(
                    det_maps, qual_maps, gt_bboxes, gt_quality,
                    self.input_size, strides,
                )

                self.optimizer.zero_grad()
                losses["total"].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                for k in ep_vals:
                    ep_vals[k] += losses[k].item()
                n_batches += 1

            for k in ep_vals:
                ep_vals[k] /= max(n_batches, 1)

            # 스케줄러 스텝
            if ep <= warmup_epochs:
                warmup.step()
            else:
                cosine.step()

            lr_now  = self.optimizer.param_groups[0]["lr"]
            t_ep_ms = (time.perf_counter() - t_ep) * 1000

            # Early stopping
            imp = ""
            if ep_vals["total"] < best_loss - min_delta:
                best_loss  = ep_vals["total"]
                no_improve = 0
                torch.save(self.model.state_dict(), best_path)
                imp = "  ✓ best"
            else:
                no_improve += 1
                imp = f"  ({no_improve}/{loss_patience})"

            self._log(
                f"[Ep {ep:03d}/{epochs}] "
                f"total={ep_vals['total']:.4f}  "
                f"conf={ep_vals['conf']:.4f}  "
                f"bbox={ep_vals['bbox']:.4f}  "
                f"qual={ep_vals['quality']:.4f}  "
                f"lr={lr_now:.2e}  "
                f"{t_ep_ms/1000:.1f}s{imp}"
            )

            if no_improve >= loss_patience:
                self._log(f"[Early Stop] {loss_patience}ep 미개선 → 중단")
                break

        self._log(f"[Done] best loss={best_loss:.4f} | saved: {best_path}")
        return best_path
