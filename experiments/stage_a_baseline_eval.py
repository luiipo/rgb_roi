"""
Stage A: Lightweight Palm Localization — Baseline Comparison
=============================================================
세 가지 Palm Detection 방법을 동일 데이터셋 위에서 비교 평가합니다.

[비교 대상]
  Method 1: YOLOv5-lite        — ultralytics YOLOv5n (or skin-color fallback)
  Method 2: MediaPipe Hands    — Google MediaPipe Hand Detector
  Method 3: Proposed           — LightweightPalmDetector + MediaPipe fallback

[평가 지표]
  1. IoU@0.5  : 검출 bbox와 GT bbox의 IoU ≥ 0.5 비율
  2. IoU@0.75 : 검출 bbox와 GT bbox의 IoU ≥ 0.75 비율
  3. Detection Accuracy : correct / total (기준: IoU > 0.5)
  4. FPS : total_images / total_inference_time
  5. Model Size (MB) : os.path.getsize(weights) / 1024 / 1024

[GT 전략 — 어노테이션 없는 경우]
  BMPD / Tongji : MediaPipe Hands 검출 결과를 Pseudo-GT로 사용.
                  (MediaPipe가 실패한 이미지는 평가에서 제외)
  MPDv2         : .mat 파일에 bbox 어노테이션이 있으면 우선 사용,
                  없으면 MediaPipe Pseudo-GT 사용.

  ※ 실제 수동 어노테이션이 준비된 경우, generate_gt() 함수를
    load_gt_from_annotation() 으로 교체하면 됩니다.

[데이터셋 경로] (data/data/ 하위)
  BMPD  : Birjand University Mobile Palmprint Database (BMPD)/<session>/*.JPG
  Tongji: Tongji/session1|session2/*.tiff
  MPDv2 : MPDv2/PalmSet01-05/*.jpg

[실행]
  python experiments/stage_a_baseline_eval.py
  python experiments/stage_a_baseline_eval.py --dataset BMPD --max_images 200
  python experiments/stage_a_baseline_eval.py --save_csv results/stage_a.csv
"""

import os
import sys
import time
import glob
import argparse
import csv
import json
import struct
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn

# 프로젝트 루트 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ─── MPS / CUDA / CPU 자동 선택 ──────────────────────────────────────────────
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()
print(f"[Device] Using: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# 0. 공통 유틸
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BBox:
    x1: int; y1: int; x2: int; y2: int

    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @staticmethod
    def from_tuple(t) -> "BBox":
        return BBox(int(t[0]), int(t[1]), int(t[2]), int(t[3]))


def compute_iou(b1: BBox, b2: BBox) -> float:
    ix1 = max(b1.x1, b2.x1); iy1 = max(b1.y1, b2.y1)
    ix2 = min(b1.x2, b2.x2); iy2 = min(b1.y2, b2.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = b1.area() + b2.area() - inter
    return inter / union if union > 0 else 0.0


@dataclass
class DetectionResult:
    bbox: Optional[BBox] = None          # None = 검출 실패
    confidence: float = 0.0
    latency_ms: float = 0.0             # 단일 이미지 처리 시간


# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터셋 로더
# ══════════════════════════════════════════════════════════════════════════════

DATA_ROOT = ROOT / "data" / "data"

DATASET_PATHS = {
    "BMPD": {
        "root": DATA_ROOT / "Birjand University Mobile Palmprint Database (BMPD)",
        "pattern": "**/*.JPG",
    },
    "Tongji": {
        "root": DATA_ROOT / "Tongji",
        "pattern": "**/*.tiff",
    },
    "MPDv2": {
        "root": DATA_ROOT / "MPDv2",
        "pattern": "**/*.jpg",
    },
}


def load_image_paths(dataset: str, max_images: Optional[int] = None) -> List[Path]:
    cfg = DATASET_PATHS.get(dataset)
    if cfg is None:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(DATASET_PATHS.keys())}")
    root = cfg["root"]
    pattern = cfg["pattern"]
    paths = sorted(root.glob(pattern))
    if not paths:
        print(f"[Warning] No images found for {dataset} at {root}")
    if max_images:
        paths = paths[:max_images]
    print(f"[Dataset] {dataset}: {len(paths)} images")
    return paths


def load_all_image_paths(datasets: List[str], max_images: Optional[int] = None) -> Dict[str, List[Path]]:
    result = {}
    for ds in datasets:
        result[ds] = load_image_paths(ds, max_images)
    return result


def read_image(path: Path) -> Optional[np.ndarray]:
    """BGR 이미지 반환. TIFF / 16bit / grayscale 모두 지원."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        return None

    # 16bit → 8bit 변환 (Tongji TIFF 대응)
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)

    # grayscale → BGR
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 4채널 → 3채널
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


# ══════════════════════════════════════════════════════════════════════════════
# 2. Ground Truth 생성 (Pseudo-GT via MediaPipe)
# ══════════════════════════════════════════════════════════════════════════════

class PseudoGTGenerator:
    """
    MediaPipe Hands를 사용해 Pseudo-GT bounding box를 생성합니다.
    검출 성공 이미지만 평가에 포함됩니다.
    """

    def __init__(self):
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.4,
            )
            self.available = True
            print("[GT] MediaPipe Pseudo-GT generator ready.")
        except ImportError:
            self.available = False
            print("[GT] MediaPipe unavailable. Using full-image bbox as GT.")

    def generate(self, image_bgr: np.ndarray) -> Optional[BBox]:
        if image_bgr is None:
            return None

        # 16bit → 8bit (Tongji TIFF 대응)
        if image_bgr.dtype == np.uint16:
            image_bgr = (image_bgr / 256).astype(np.uint8)

        # grayscale → BGR
        if len(image_bgr.shape) == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

        # 4채널 → 3채널
        if image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

        h, w = image_bgr.shape[:2]

        if not self.available:
            return BBox(0, 0, w, h)

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        if not res.multi_hand_landmarks:
            return None

        lms = res.multi_hand_landmarks[0]
        pts = np.array([[lm.x * w, lm.y * h] for lm in lms.landmark])

        x1, y1 = pts.min(axis=0).astype(int)
        x2, y2 = pts.max(axis=0).astype(int)

        pad = int(max(x2 - x1, y2 - y1) * 0.08)

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        return BBox(x1, y1, x2, y2)

    def try_load_mat_gt(self, mat_path: Path) -> Optional[BBox]:
        """MPDv2 .mat 어노테이션 파싱 시도."""
        try:
            import scipy.io as sio
            data = sio.loadmat(str(mat_path))
            for key in ["bbox", "roi", "hand_bbox", "boundingBox"]:
                if key in data:
                    arr = data[key].flatten()
                    if len(arr) >= 4:
                        return BBox(int(arr[0]), int(arr[1]),
                                    int(arr[0]) + int(arr[2]),
                                    int(arr[1]) + int(arr[3]))
        except Exception:
            pass
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 3. Method 1 — YOLOv5-lite Detector
# ══════════════════════════════════════════════════════════════════════════════

class YOLOv5LiteDetector:
    """
    YOLOv5-lite Palm Detector.

    우선순위:
      1. ultralytics 패키지 + yolov5n 모델 (COCO pretrained, "person" 클래스)
      2. 피부색(Skin-Color) 기반 경량 검출 fallback

    ※ 실제 Palm Detection 전용 YOLO 가중치가 있으면
      load_custom_weights(path) 로 교체하세요.
    """

    MODEL_NAME = "yolov5n"   # 가장 작은 YOLOv5 모델

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.model_path: Optional[str] = None
        self._try_load_ultralytics()

    def _try_load_ultralytics(self):
        try:
            import torch
            # yolov5 패키지 (pip install yolov5)
            self.model = torch.hub.load(
                "ultralytics/yolov5", self.MODEL_NAME,
                pretrained=True, device=self.device, verbose=False
            )
            self.model.conf = 0.25
            self.model.iou  = 0.45
            self.model.classes = [0]   # person 클래스만 (COCO)
            # 임시 가중치 경로 추정
            cache = Path.home() / ".cache" / "torch" / "hub"
            w = list(cache.glob(f"**/{self.MODEL_NAME}.pt"))
            self.model_path = str(w[0]) if w else None
            print(f"[YOLOv5-lite] ultralytics YOLOv5n loaded (device={self.device})")
        except Exception as e:
            print(f"[YOLOv5-lite] ultralytics unavailable ({e}). Using skin-color fallback.")
            self.model = None

    def load_custom_weights(self, path: str):
        """Palm Detection 전용 가중치 로드."""
        try:
            import torch
            self.model = torch.hub.load(
                "ultralytics/yolov5", "custom",
                path=path, device=self.device, verbose=False
            )
            self.model_path = path
            print(f"[YOLOv5-lite] Custom weights loaded: {path}")
        except Exception as e:
            print(f"[YOLOv5-lite] Custom weights load failed: {e}")

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        t0 = time.perf_counter()
        result = self._detect_impl(image_bgr)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def _detect_impl(self, image_bgr: np.ndarray) -> DetectionResult:
        if self.model is not None:
            return self._yolo_detect(image_bgr)
        return self._skin_color_detect(image_bgr)

    def _yolo_detect(self, image_bgr: np.ndarray) -> DetectionResult:
        """ultralytics YOLOv5 추론."""
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.model(rgb, size=320)
        dets = results.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2,conf,cls]
        if len(dets) == 0:
            return DetectionResult()
        # 가장 큰 bbox 선택
        areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        best = dets[areas.argmax()]
        bbox = BBox(int(best[0]), int(best[1]), int(best[2]), int(best[3]))
        return DetectionResult(bbox=bbox, confidence=float(best[4]))

    def _skin_color_detect(self, image_bgr: np.ndarray) -> DetectionResult:
        """피부색 기반 경량 검출 (YOLO 대체 fallback)."""
        h, w = image_bgr.shape[:2]
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 15, 60],  dtype=np.uint8)
        upper = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        # Ycbcr 추가 마스크로 정확도 향상
        ycbcr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        mask2 = cv2.inRange(ycbcr,
                            np.array([0, 135, 85], dtype=np.uint8),
                            np.array([255, 180, 135], dtype=np.uint8))
        mask = cv2.bitwise_and(mask, mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return DetectionResult()
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 200:
            return DetectionResult()
        x, y, bw, bh = cv2.boundingRect(c)
        pad = int(max(bw, bh) * 0.05)
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad); y2 = min(h, y + bh + pad)
        return DetectionResult(bbox=BBox(x1, y1, x2, y2), confidence=0.65)

    @property
    def model_size_mb(self) -> float:
        if self.model_path and os.path.exists(self.model_path):
            return os.path.getsize(self.model_path) / 1024 / 1024
        # YOLO 모델 파라미터 수 기반 추정 (yolov5n ≈ 7.5 MB)
        if self.model is not None:
            try:
                params = sum(p.numel() for p in self.model.parameters())
                return params * 4 / 1024 / 1024  # float32 기준
            except Exception:
                return 7.5
        return 0.1   # skin-color: 가중치 없음


# ══════════════════════════════════════════════════════════════════════════════
# 4. Method 2 — MediaPipe Hands Detector
# ══════════════════════════════════════════════════════════════════════════════

class MediaPipeDetector:
    """
    Google MediaPipe Hands 기반 Palm Detector.
    21개 랜드마크의 bounding box를 Palm bbox로 사용.
    """

    MEDIAPIPE_PALM_SIZE_MB = 0.9   # palm_detection_lite.tflite ≈ 0.9 MB
    MEDIAPIPE_HAND_SIZE_MB = 8.3   # hand_landmark_lite.tflite ≈ 8.3 MB

    def __init__(self, min_confidence: float = 0.5):
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            self.available = True
            print("[MediaPipe] MediaPipe Hands loaded.")
        except ImportError:
            self.available = False
            print("[MediaPipe] MediaPipe not installed. Detector will always fail.")

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        t0 = time.perf_counter()
        result = self._detect_impl(image_bgr)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def _detect_impl(self, image_bgr: np.ndarray) -> DetectionResult:
        if not self.available:
            return DetectionResult()
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return DetectionResult()
        lms = res.multi_hand_landmarks[0]
        pts = np.array([[lm.x * w, lm.y * h] for lm in lms.landmark])
        x1, y1 = pts.min(axis=0).astype(int)
        x2, y2 = pts.max(axis=0).astype(int)
        pad = int(max(x2 - x1, y2 - y1) * 0.08)
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        return DetectionResult(bbox=BBox(x1, y1, x2, y2), confidence=1.0)

    @property
    def model_size_mb(self) -> float:
        # MediaPipe palm detector + landmark 모델 크기 합계
        return self.MEDIAPIPE_PALM_SIZE_MB + self.MEDIAPIPE_HAND_SIZE_MB


# ══════════════════════════════════════════════════════════════════════════════
# 5. Method 3 — Proposed: LandmarkFreePalmDetector (제안 모델)
# ══════════════════════════════════════════════════════════════════════════════

class ProposedDetector:
    """
    제안 방법: models/stage_a_localization.py 의 PalmROIExtractor 를 그대로 사용.

    파이프라인 (학습 가중치 유무에 따라 자동 전환):
      ┌─ 가중치 있음 ─────────────────────────────────────────────────────┐
      │  LandmarkFreePalmDetector (CNN, FCOS-style)                        │
      │    MobileDetectorBackbone (P3/P4) → FeaturePyramidNeck            │
      │    → AnchorFreeHead (conf + l,t,r,b) → QualityHead (blur/scale/…) │
      │  → QualityAwareROIEstimator (NMS + quality ranking)               │
      │  → AdaptiveROIRefiner (quality-adaptive margin + square crop)      │
      └───────────────────────────────────────────────────────────────────┘
      ┌─ 가중치 없음 ─────────────────────────────────────────────────────┐
      │  MediaPipe Hands (설치된 경우)                                    │
      │    → QualityAwareROIEstimator (rule-based quality)               │
      │    → AdaptiveROIRefiner                                           │
      └───────────────────────────────────────────────────────────────────┘
      ┌─ MediaPipe 없음 ───────────────────────────────────────────────────┐
      │  Skin-Color Detection (HSV + YCbCr)                                │
      │    → QualityAwareROIEstimator → AdaptiveROIRefiner                │
      └───────────────────────────────────────────────────────────────────┘

    학습: python main.py --mode train_stage_a
    가중치 저장: checkpoints/stage_a/detector.pt  (PalmROIExtractor가 자동 로드)
    """

    def __init__(self, device: str = "cpu"):
        from models.stage_a_localization import PalmROIExtractor
        self._extractor = PalmROIExtractor(device=device)

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        t0 = time.perf_counter()
        palm = self._extractor.detect(image_bgr)
        latency_ms = (time.perf_counter() - t0) * 1000

        if palm is None:
            return DetectionResult(latency_ms=latency_ms, method_used="failed")

        bbox = BBox(int(palm.bbox[0]), int(palm.bbox[1]),
                    int(palm.bbox[2]), int(palm.bbox[3]))
        # 사용된 내부 방법 추적
        if self._extractor._net_loaded:
            used = "landmark-free-cnn"
        elif self._extractor._mp:
            used = "mediapipe-fallback"
        else:
            used = "skincolor-fallback"

        return DetectionResult(
            bbox=bbox,
            confidence=palm.confidence,
            latency_ms=latency_ms,
            method_used=used,
        )

    @property
    def model_size_mb(self) -> float:
        return self._extractor.model_size_mb


# ══════════════════════════════════════════════════════════════════════════════
# 6. 평가 루프
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MethodMetrics:
    name: str
    iou_scores: List[float] = field(default_factory=list)
    correct_05: List[bool]  = field(default_factory=list)   # IoU > 0.5
    correct_075: List[bool] = field(default_factory=list)   # IoU > 0.75
    latencies_ms: List[float] = field(default_factory=list)
    n_failed: int = 0       # 검출 자체 실패 (bbox=None)
    model_size_mb: float = 0.0

    def add(self, pred: Optional[BBox], gt: BBox, latency_ms: float):
        self.latencies_ms.append(latency_ms)
        if pred is None:
            self.n_failed += 1
            self.iou_scores.append(0.0)
            self.correct_05.append(False)
            self.correct_075.append(False)
            return
        iou = compute_iou(pred, gt)
        self.iou_scores.append(iou)
        self.correct_05.append(iou >= 0.5)
        self.correct_075.append(iou >= 0.75)

    def summary(self) -> dict:
        n = len(self.iou_scores)
        mean_iou = float(np.mean(self.iou_scores)) if n > 0 else 0.0
        acc_05  = float(np.mean(self.correct_05))  if n > 0 else 0.0
        acc_075 = float(np.mean(self.correct_075)) if n > 0 else 0.0
        fps = 1000.0 / np.mean(self.latencies_ms) if self.latencies_ms else 0.0
        return {
            "method":          self.name,
            "n_total":         n,
            "n_failed":        self.n_failed,
            "mean_iou":        round(mean_iou, 4),
            "accuracy@0.5":    round(acc_05,  4),
            "accuracy@0.75":   round(acc_075, 4),
            "fps":             round(fps, 2),
            "model_size_mb":   round(self.model_size_mb, 2),
        }


def evaluate_on_dataset(
    dataset_name: str,
    image_paths: List[Path],
    detectors: Dict[str, object],
    gt_generator: PseudoGTGenerator,
) -> Dict[str, MethodMetrics]:
    """
    단일 데이터셋에 대해 모든 방법을 평가하고 MethodMetrics 반환.
    """
    metrics = {
        name: MethodMetrics(name=name, model_size_mb=det.model_size_mb)
        for name, det in detectors.items()
    }

    n_skipped = 0
    for i, path in enumerate(image_paths):
        img = read_image(path)
        if img is None:
            n_skipped += 1
            continue

        # GT 생성 (MPDv2: .mat 시도 → MediaPipe)
        gt_bbox = None
        if dataset_name == "MPDv2":
            mat_path = path.with_suffix(".mat")
            if mat_path.exists():
                gt_bbox = gt_generator.try_load_mat_gt(mat_path)
        if gt_bbox is None:
            gt_bbox = gt_generator.generate(img)
        if gt_bbox is None:
            n_skipped += 1
            continue   # GT를 생성할 수 없는 이미지는 제외

        # 각 방법 실행
        for name, det in detectors.items():
            result: DetectionResult = det.detect(img)
            metrics[name].add(result.bbox, gt_bbox, result.latency_ms)

        if (i + 1) % 50 == 0:
            print(f"  [{dataset_name}] {i+1}/{len(image_paths)} processed ...")

    print(f"  [{dataset_name}] Done. Skipped: {n_skipped} images.")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 7. 결과 출력 & 저장
# ══════════════════════════════════════════════════════════════════════════════

def print_results_table(results: Dict[str, Dict[str, MethodMetrics]]):
    """
    데이터셋별 × 방법별 결과 테이블을 콘솔에 출력합니다.
    """
    all_methods = []
    for ds_metrics in results.values():
        all_methods = list(ds_metrics.keys())
        break

    COLS = ["method", "n_total", "n_failed",
            "mean_iou", "accuracy@0.5", "accuracy@0.75",
            "fps", "model_size_mb"]
    COL_W = [22, 8, 8, 10, 13, 14, 8, 14]
    HEADER = "".join(c.ljust(w) for c, w in zip(COLS, COL_W))

    for ds_name, ds_metrics in results.items():
        print(f"\n{'━'*len(HEADER)}")
        print(f"  Dataset: {ds_name}")
        print(f"{'━'*len(HEADER)}")
        print(HEADER)
        print("-" * len(HEADER))
        for method_name, met in ds_metrics.items():
            s = met.summary()
            row_vals = [str(s[c]) for c in COLS]
            print("".join(v.ljust(w) for v, w in zip(row_vals, COL_W)))
        print()


def save_results_csv(results: Dict[str, Dict[str, MethodMetrics]], save_path: str):
    rows = []
    for ds_name, ds_metrics in results.items():
        for method_name, met in ds_metrics.items():
            row = {"dataset": ds_name}
            row.update(met.summary())
            rows.append(row)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fieldnames = ["dataset"] + list(rows[0].keys()) if rows else []
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Saved] CSV → {save_path}")


def save_results_json(results: Dict[str, Dict[str, MethodMetrics]], save_path: str):
    data = {}
    for ds_name, ds_metrics in results.items():
        data[ds_name] = {m: met.summary() for m, met in ds_metrics.items()}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[Saved] JSON → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. 통합 시각화 (선택)
# ══════════════════════════════════════════════════════════════════════════════

def visualize_sample(
    image_bgr: np.ndarray,
    gt_bbox: BBox,
    predictions: Dict[str, Optional[BBox]],
    save_path: Optional[str] = None,
):
    """
    단일 이미지에 GT + 각 방법의 bbox를 시각화합니다.
    """
    vis = image_bgr.copy()
    h, w = vis.shape[:2]
    # GT — 흰색
    cv2.rectangle(vis, (gt_bbox.x1, gt_bbox.y1), (gt_bbox.x2, gt_bbox.y2), (255, 255, 255), 2)
    cv2.putText(vis, "GT", (gt_bbox.x1, gt_bbox.y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    COLORS = {"YOLOv5-lite": (0, 200, 0),
              "MediaPipe":   (0, 128, 255),
              "Proposed":    (0, 0, 255)}
    for name, bbox in predictions.items():
        if bbox is None:
            continue
        color = COLORS.get(name, (200, 200, 0))
        cv2.rectangle(vis, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)
        iou = compute_iou(bbox, gt_bbox)
        label = f"{name} IoU={iou:.2f}"
        cv2.putText(vis, label, (bbox.x1, bbox.y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, vis)
    return vis


# ══════════════════════════════════════════════════════════════════════════════
# 9. Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Stage A Palm Localization Baseline Eval")
    p.add_argument("--dataset", nargs="+",
                   default=["BMPD", "Tongji", "MPDv2"],
                   choices=["BMPD", "Tongji", "MPDv2"],
                   help="평가할 데이터셋 (복수 선택 가능)")
    p.add_argument("--max_images", type=int, default=None,
                   help="데이터셋당 최대 이미지 수 (None = 전체)")
    p.add_argument("--save_csv",  default="results/stage_a_baseline.csv",
                   help="결과 CSV 저장 경로")
    p.add_argument("--save_json", default="results/stage_a_baseline.json",
                   help="결과 JSON 저장 경로")
    p.add_argument("--visualize", action="store_true",
                   help="각 데이터셋 첫 이미지 시각화 저장")
    p.add_argument("--device", default=None,
                   help="강제 디바이스 지정 (mps/cuda/cpu). 기본: 자동")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or DEVICE
    print(f"\n{'═'*60}")
    print("  Stage A — Lightweight Palm Localization Baseline Eval")
    print(f"{'═'*60}")
    print(f"  Device   : {device}")
    print(f"  Datasets : {args.dataset}")
    print(f"  Max imgs : {args.max_images or 'all'}")
    print(f"{'═'*60}\n")

    # ── 방법 초기화 ────────────────────────────────────────────────
    detectors = {
        "YOLOv5-lite": YOLOv5LiteDetector(device=device),
        "MediaPipe":   MediaPipeDetector(),
        "Proposed":    ProposedDetector(device=device),
    }
    gt_gen = PseudoGTGenerator()

    # ── 데이터셋별 평가 ─────────────────────────────────────────────
    all_results: Dict[str, Dict[str, MethodMetrics]] = {}
    for ds in args.dataset:
        paths = load_image_paths(ds, max_images=args.max_images)
        if not paths:
            print(f"[Skip] {ds}: no images found.")
            continue
        print(f"\n[Evaluating] {ds} ({len(paths)} images) ...")
        ds_metrics = evaluate_on_dataset(ds, paths, detectors, gt_gen)
        all_results[ds] = ds_metrics

        # 선택적 시각화 (첫 이미지)
        if args.visualize and paths:
            img = read_image(paths[0])
            if img is not None:
                gt = gt_gen.generate(img)
                if gt is not None:
                    preds = {}
                    for name, det in detectors.items():
                        r = det.detect(img)
                        preds[name] = r.bbox
                    vis_path = f"results/vis_{ds}_sample.jpg"
                    visualize_sample(img, gt, preds, save_path=vis_path)
                    print(f"  [Visualize] {vis_path}")

    # ── 결과 출력 & 저장 ────────────────────────────────────────────
    if not all_results:
        print("[Error] No results to display.")
        return

    print_results_table(all_results)
    save_results_csv(all_results, args.save_csv)
    save_results_json(all_results, args.save_json)
    print("\n[Done] Evaluation complete.")


if __name__ == "__main__":
    main()
