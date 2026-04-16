"""
Dataset Loaders
지원 데이터셋:
  - SMPD  (Sapienza University Mobile Palmprint Database)
  - MPDv2 (Mobile Palmprint Database v2)
  - BMPD  (Birjand University Mobile Palmprint Database)
  - Tongji (Tongji Mobile Palmprint Database — contact ROI, 인식 평가용)
  - IITD  (IIT Delhi Touchless Palmprint)

각 데이터셋의 디렉터리 구조가 다를 수 있으므로
DatasetConfig로 경로 패턴을 추상화.

[경로 전략]
  - DATA_ROOT = 이 파일(data/datasets.py) 기준 ../data/data/
  - 절대 경로 하드코딩 없음 → 어느 환경에서도 동작
"""

import os
import glob
import random
import numpy as np
import cv2
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
from collections import defaultdict

# ── 프로젝트 상대 경로 (환경 무관) ───────────────────────────────────────────
# data/datasets.py → 한 칸 위(data/) → 한 칸 위(project root) → data/data/
_DATA_ROOT = Path(__file__).resolve().parent / "data"


# ==================================================================
# 데이터셋 설정 (경로 패턴)
# ==================================================================

@dataclass
class DatasetConfig:
    name: str
    root: str
    img_pattern: str = "**/*.jpg"
    id_level: int = 1
    id_parser: Optional[Callable[[str], str]] = None
    split_ratio: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    left_right_split: bool = False
    csv_path: Optional[str] = None
    min_samples: int = 3   # 클래스당 최소 샘플 수


def _make_configs() -> Dict[str, "DatasetConfig"]:
    """
    DATA_ROOT 기반으로 DATASET_CONFIGS를 동적 생성.
    절대 경로 하드코딩을 제거하고 실행 환경에 자동 적응.
    """
    d = _DATA_ROOT

    configs = {
        "SMPD": DatasetConfig(
            name="SMPD",
            root=str(d / "Sapienza University Mobile Palmprint Database(SMPD)"),
            img_pattern="**/*.JPG",   # SMPD 이미지는 .JPG (대문자)
            id_level=1,
            min_samples=3,
        ),
        "MPDv2": DatasetConfig(
            name="MPDv2",
            root=str(d / "MPDv2"),
            img_pattern="**/*.jpg",
            id_level=1,
            # 파일명 앞 3자리 숫자 = subject ID
            # e.g. 001_1_h_l_01.jpg → "001"
            id_parser=lambda p: os.path.basename(p).split("_")[0],
            min_samples=3,
        ),
        "BMPD": DatasetConfig(
            name="BMPD",
            root=str(d / "Birjand University Mobile Palmprint Database (BMPD)"),
            # BMPD 이미지는 .JPG (대문자) — glob은 대소문자 구분 없이 처리
            img_pattern="**/*.JPG",
            id_level=1,
            min_samples=3,
        ),
        "Tongji": DatasetConfig(
            name="Tongji",
            root=str(d / "Tongji"),
            img_pattern="**/*.tiff",
            id_level=1,
            min_samples=3,
        ),
        "IITD": DatasetConfig(
            name="IITD",
            root=str(d / "IITD"),
            img_pattern="**/*.bmp",
            id_level=1,
            id_parser=lambda p: os.path.basename(p).split("_")[0],
            min_samples=3,
        ),
    }

    # 실제로 존재하는 데이터셋만 활성화 (경고 출력)
    active = {}
    for name, cfg in configs.items():
        if os.path.isdir(cfg.root):
            active[name] = cfg
        else:
            print(f"[Dataset] ⚠ {name}: 디렉터리 없음 → 비활성 ({cfg.root})")
    return active


DATASET_CONFIGS: Dict[str, DatasetConfig] = _make_configs()


# ==================================================================
# 유틸리티
# ==================================================================

def _extract_id(path: str, id_level: int, parser=None) -> str:
    if parser is not None:
        return parser(path)

    parts = path.replace("\\", "/").split("/")
    if len(parts) <= id_level:
        return parts[0]
    return parts[-(id_level + 1)]


def _build_index(root: str, pattern: str, id_level: int, parser=None, csv_path=None):
    
    index = defaultdict(list)

    # CSV 사용하는 경우
    if csv_path is not None and os.path.exists(csv_path):
        print(f"[Dataset] Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)

        # CSV에 path 컬럼 필요
        paths = df["full_path"].tolist()

        for p in paths:
            full_path = p if os.path.isabs(p) else os.path.join(root, p)
            rel = os.path.relpath(full_path, root)
            class_id = _extract_id(rel, id_level, parser)
            index[class_id].append(full_path)

    else:
        # 기존 glob 방식
        all_paths = glob.glob(os.path.join(root, pattern), recursive=True)

        for p in all_paths:
            rel = os.path.relpath(p, root)
            class_id = _extract_id(rel, id_level, parser)
            index[class_id].append(p)

    return dict(index)

# ==================================================================
# 기본 변환
# ==================================================================

def default_transform(roi_size: int = 128) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((roi_size, roi_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# data/datasets.py - augment_transform 수정
def augment_transform(roi_size=128):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((roi_size, roi_size)),
        transforms.RandomAffine(degrees=20,
                                translate=(0.08, 0.08),
                                scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomGrayscale(p=0.1),       # 추가
        transforms.GaussianBlur(kernel_size=3),   # 추가
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# ==================================================================
# PalmprintDataset
# ==================================================================

class PalmprintDataset(Dataset):
    """
    단일 데이터셋 로더.
    split='train'|'val'|'test' 로 분리.
    """

    def __init__(
        self,
        config: DatasetConfig,
        split: str = "train",
        roi_size: int = 128,
        augment: bool = True,
        seed: int = 42,
    ):
        self.config = config
        self.split = split
        self.roi_size = roi_size

        # 인덱스 구축
        raw_index = _build_index(
            config.root,
            config.img_pattern,
            config.id_level,
            config.id_parser,
            config.csv_path,
        )
        # 충분한 샘플이 있는 클래스만 사용 (최소 3장)
        min_s = getattr(config, 'min_samples', 3)
        raw_index = {k: v for k, v in raw_index.items() if len(v) >= min_s}

        # 클래스 정렬 후 재현성 있는 분할
        classes = sorted(raw_index.keys())
        rng = random.Random(seed)
        rng.shuffle(classes)

        n = len(classes)
        tr, va, te = config.split_ratio
        n_tr = int(n * tr)
        n_va = int(n * va)

        if split == "train":
            selected = classes[:n_tr]
        elif split == "val":
            selected = classes[n_tr:n_tr + n_va]
        else:
            selected = classes[n_tr + n_va:]

        self.class_to_idx = {c: i for i, c in enumerate(selected)}
        self.samples: List[Tuple[str, int]] = []
        for c in selected:
            for p in raw_index[c]:
                self.samples.append((p, self.class_to_idx[c]))

        self.transform = augment_transform(roi_size) if (augment and split == "train") \
            else default_transform(roi_size)

        print(
            f"[Dataset] {config.name}/{split}: "
            f"{len(selected)} classes, {len(self.samples)} samples"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.roi_size, self.roi_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = self.transform(img)
        return img_t, torch.tensor(label, dtype=torch.long)

    @property
    def num_classes(self):
        return len(self.class_to_idx)


# ==================================================================
# Cross-Dataset Loader (Generalization 평가용)
# ==================================================================

class CrossDatasetLoader:
    """
    train on A+B, test on C 형태의 cross-dataset 실험 지원.
    """

    def __init__(
        self,
        train_datasets: List[str],
        test_dataset: str,
        roi_size: int = 128,
        batch_size: int = 32,
        num_workers: int = 0,   # Windows freeze 방지: 기본 0
    ):
        self.roi_size = roi_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        # 학습 데이터셋 합치기
        train_sets = []
        self.num_train_classes = 0
        for name in train_datasets:
            cfg = DATASET_CONFIGS[name]
            ds = PalmprintDataset(cfg, split="train", roi_size=roi_size, augment=True)
            # 클래스 ID offset 적용 (데이터셋 간 충돌 방지)
            ds.samples = [(p, lbl + self.num_train_classes) for p, lbl in ds.samples]
            self.num_train_classes += ds.num_classes
            train_sets.append(ds)

        self.train_dataset = torch.utils.data.ConcatDataset(train_sets)

        # 테스트 데이터셋
        cfg = DATASET_CONFIGS[test_dataset]
        self.test_dataset = PalmprintDataset(cfg, split="test", roi_size=roi_size, augment=False)

    def get_train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=False,   # MPS(Apple Silicon) 비호환 → False
        )

    def get_test_loader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=False,   # MPS 비호환 → False
        )


# ==================================================================
# Video Burst Loader (비디오 burst ROI 융합 평가용)
# ==================================================================

class VideoFrameDataset(Dataset):
    """
    스마트폰으로 촬영한 짧은 팜 비디오 burst 로더.
    디렉터리 구조: root/subject_id/session_id/frame_*.jpg
    """

    def __init__(self, root: str, roi_size: int = 128):
        self.sequences: List[Tuple[str, List[str]]] = []
        self.roi_size = roi_size
        self.transform = default_transform(roi_size)

        for subj in sorted(os.listdir(root)):
            subj_dir = os.path.join(root, subj)
            if not os.path.isdir(subj_dir):
                continue
            for sess in sorted(os.listdir(subj_dir)):
                sess_dir = os.path.join(subj_dir, sess)
                frames = sorted(glob.glob(os.path.join(sess_dir, "*.jpg")))
                if frames:
                    self.sequences.append((subj, frames))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        subject_id, frames = self.sequences[idx]
        imgs = []
        for fp in frames[:10]:   # 최대 10 프레임
            img = cv2.imread(fp)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(self.transform(img))
        if not imgs:
            imgs.append(torch.zeros(3, self.roi_size, self.roi_size))
        return torch.stack(imgs), subject_id
