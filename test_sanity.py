"""
Quick Sanity Check — 실제 데이터 없이 모든 모듈이 올바르게 임포트되고
더미 데이터로 전방 전달(forward pass)이 통과하는지 확인.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import cv2


def test_stage_a():
    from models.stage_a_localization import PalmLocalizer
    loc = PalmLocalizer()
    dummy_img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    # MediaPipe 없는 환경: skin-color fallback 사용
    result = loc(dummy_img)
    print(f"[Stage A] Detection result: {result}")
    return True


def test_stage_b():
    from models.stage_a_localization import PalmDetection
    from models.stage_b_alignment import TopologyGuidedROIAligner
    aligner = TopologyGuidedROIAligner(roi_size=128)

    dummy_img = (np.random.rand(480, 640, 3) * 200 + 30).astype(np.uint8)

    # 가짜 손 마스크 생성 (중앙에 원형)
    mask = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(mask, (320, 240), 120, 255, -1)

    detection = PalmDetection(
        bbox=(200, 120, 440, 360),
        confidence=0.9,
        hand_mask=mask,
    )
    roi_result = aligner.align(dummy_img, detection)
    assert roi_result is not None, "Stage B returned None"
    assert roi_result.roi_image.shape == (128, 128, 3)
    print(f"[Stage B] ROI shape: {roi_result.roi_image.shape}, "
          f"angle: {roi_result.angle_deg:.1f}°, "
          f"conf: {roi_result.alignment_confidence:.3f}")
    return True


def test_stage_c():
    from models.stage_c_quality import QualityAwareROISelector
    selector = QualityAwareROISelector(roi_size=128, top_k=3)

    rois = [(np.random.rand(128, 128, 3) * 255).astype(np.uint8) for _ in range(5)]
    scores = [
        selector.score_roi(roi, alignment_conf=np.random.uniform(0.5, 1.0),
                           inscribed_radius=np.random.uniform(40, 70))
        for roi in rois
    ]
    best_roi, best_score = selector.select_best(rois, scores)
    fused = selector.fuse_burst(rois, scores)
    lw, gw = selector.compute_branch_weights(best_score)
    print(f"[Stage C] Best score: {best_score.total:.3f}  local_w={lw:.3f}  global_w={gw:.3f}")
    print(f"[Stage C] Fused ROI shape: {fused.shape}")
    return True


def test_stage_d():
    from models.stage_d_security import SecurityChecker
    checker = SecurityChecker()

    roi = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    full_hand = (np.random.rand(300, 300, 3) * 255).astype(np.uint8)
    mask = np.ones((300, 300), dtype=np.uint8) * 255

    result = checker.check(roi, full_hand, mask)
    print(f"[Stage D] Security check: {result}")
    return True


def test_feature_extraction():
    from models.feature_extraction import HybridPalmprintEncoder, PalmprintRecognitionModel

    # Encoder 단독 테스트
    encoder = HybridPalmprintEncoder(local_dim=256, global_dim=256, embed_dim=512)
    x = torch.randn(2, 3, 128, 128)
    feat = encoder(x)
    assert feat.shape == (2, 512), f"Expected (2, 512), got {feat.shape}"
    print(f"[Feature] Encoder output shape: {feat.shape}")

    # 전체 recognition model 테스트
    model = PalmprintRecognitionModel(num_classes=100, embed_dim=512)
    labels = torch.randint(0, 100, (2,))
    out = model(x, labels=labels)
    print(f"[Feature] Total loss: {out['total_loss'].item():.4f}")
    print(f"[Feature] Embedding shape: {out['embedding'].shape}")
    return True


def test_pretraining():
    from models.pretraining import SyntheticPalmGenerator, SyntheticPalmDataset

    gen = SyntheticPalmGenerator(size=128)
    samples = gen.generate(3)
    assert len(samples) == 3
    assert samples[0].shape == (128, 128, 3)
    print(f"[Pretrain] Synthetic palm shape: {samples[0].shape}")

    ds = SyntheticPalmDataset(n_classes=10, n_per_class=5, roi_size=128)
    img, label = ds[0]
    assert img.shape == (3, 128, 128)
    print(f"[Pretrain] Dataset len={len(ds)}, sample shape={img.shape}, label={label}")
    return True


def test_evaluation():
    from experiments.evaluation import (
        compute_iou, compute_eer, compute_tar_at_far,
        compute_rank1_identification, evaluate_efficiency
    )
    from models.feature_extraction import PalmprintRecognitionModel

    # IoU test
    iou = compute_iou((0, 0, 100, 100), (50, 50, 150, 150))
    print(f"[Eval] IoU test: {iou:.3f} (expected ~0.143)")

    # EER test
    genuine = np.random.normal(0.8, 0.05, 500)
    impostor = np.random.normal(0.3, 0.1, 5000)
    eer, thresh = compute_eer(genuine, impostor)
    print(f"[Eval] EER: {eer*100:.2f}%  threshold: {thresh:.3f}")

    # TAR@FAR
    tar = compute_tar_at_far(genuine, impostor, far_target=1e-2)
    print(f"[Eval] TAR@FAR=1e-2: {tar*100:.2f}%")

    # Rank-1
    n_classes, n_per_class = 20, 10
    embs = np.random.randn(n_classes * n_per_class, 512)
    # 같은 클래스끼리 유사하게 만들기
    for i in range(n_classes):
        center = np.random.randn(512) * 5
        embs[i*n_per_class:(i+1)*n_per_class] += center
    labels = np.repeat(np.arange(n_classes), n_per_class)
    rank1 = compute_rank1_identification(embs, labels)
    print(f"[Eval] Rank-1: {rank1*100:.2f}%")

    # Efficiency
    model = PalmprintRecognitionModel(num_classes=100, embed_dim=512)
    eff = evaluate_efficiency(model, device="cpu")
    print(f"[Eval] Efficiency: {eff}")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("Running all module sanity checks ...")
    print("=" * 50)

    tests = [
        ("Stage A: Palm Localization", test_stage_a),
        ("Stage B: ROI Alignment", test_stage_b),
        ("Stage C: Quality Scoring", test_stage_c),
        ("Stage D: Security Check", test_stage_d),
        ("Feature Extraction", test_feature_extraction),
        ("Pretraining", test_pretraining),
        ("Evaluation Metrics", test_evaluation),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}\n")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}\n")
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
