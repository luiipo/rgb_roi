"""
Stage A — Baseline Comparison Runner & Result Aggregator
=========================================================
stage_a_baseline_eval.py 를 실행한 후 저장된 JSON/CSV를
읽어 크로스-데이터셋 요약 테이블을 출력하고
시각화(matplotlib) 차트를 생성합니다.

[사용법]
  # 1) 평가 실행 (결과 JSON/CSV 생성)
  python experiments/stage_a_baseline_eval.py \
      --dataset BMPD Tongji MPDv2 \
      --max_images 300 \
      --save_json results/stage_a_baseline.json \
      --save_csv  results/stage_a_baseline.csv \
      --visualize

  # 2) 결과 집계 & 차트 출력
  python experiments/run_stage_a_comparison.py

  # 또는 두 단계를 한 번에:
  python experiments/run_stage_a_comparison.py --run_eval --max_images 300
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULT_JSON = ROOT / "results" / "stage_a_baseline.json"
RESULT_CSV  = ROOT / "results" / "stage_a_baseline.csv"
CHART_DIR   = ROOT / "results"


# ══════════════════════════════════════════════════════════════════
# 1. 결과 로드
# ══════════════════════════════════════════════════════════════════

def load_results(json_path: Path) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(
            f"결과 파일이 없습니다: {json_path}\n"
            "먼저 stage_a_baseline_eval.py 를 실행하세요."
        )
    with open(json_path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════
# 2. 콘솔 요약 테이블
# ══════════════════════════════════════════════════════════════════

METRICS = [
    ("mean_iou",       "Mean IoU"),
    ("accuracy@0.5",   "Acc@IoU0.5"),
    ("accuracy@0.75",  "Acc@IoU0.75"),
    ("fps",            "FPS"),
    ("model_size_mb",  "Size(MB)"),
]

def print_summary(results: dict):
    datasets = list(results.keys())
    methods  = list(next(iter(results.values())).keys())

    print(f"\n{'═'*80}")
    print("  Stage A Palm Localization — Cross-Dataset Summary")
    print(f"{'═'*80}")

    for metric_key, metric_label in METRICS:
        print(f"\n  ▶ {metric_label}")
        # 헤더
        header = f"  {'Method':<20}" + "".join(f"{ds:>12}" for ds in datasets) + f"{'Average':>12}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for method in methods:
            vals = []
            for ds in datasets:
                v = results[ds][method].get(metric_key, 0.0)
                vals.append(v)
            avg = sum(vals) / len(vals) if vals else 0.0
            row = f"  {method:<20}" + "".join(f"{v:>12.3f}" for v in vals) + f"{avg:>12.3f}"
            print(row)
    print()


# ══════════════════════════════════════════════════════════════════
# 3. 비교 차트 생성
# ══════════════════════════════════════════════════════════════════

def plot_comparison(results: dict, save_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[Chart] matplotlib 미설치. 차트 생략.")
        return

    datasets = list(results.keys())
    methods  = list(next(iter(results.values())).keys())

    PLOT_METRICS = [
        ("accuracy@0.5",   "Detection Accuracy (IoU≥0.5)", "%"),
        ("accuracy@0.75",  "Detection Accuracy (IoU≥0.75)", "%"),
        ("mean_iou",       "Mean IoU", ""),
        ("fps",            "FPS (↑ better)", "fps"),
        ("model_size_mb",  "Model Size (↓ better)", "MB"),
    ]

    n_metrics = len(PLOT_METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = ["#4C72B0", "#DD8452", "#55A868"]  # 방법별 색상
    x = np.arange(len(datasets))
    bar_w = 0.25

    for ax, (mkey, mtitle, munit) in zip(axes, PLOT_METRICS):
        for i, (method, color) in enumerate(zip(methods, colors)):
            vals = []
            for ds in datasets:
                v = results[ds][method].get(mkey, 0.0)
                # accuracy를 % 단위로
                if munit == "%":
                    v = v * 100
                vals.append(v)
            offset = (i - len(methods) / 2 + 0.5) * bar_w
            bars = ax.bar(x + offset, vals, bar_w, label=method, color=color, alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01 * max(vals + [1]),
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(mtitle, fontsize=10, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(munit if munit else "", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Stage A: Lightweight Palm Localization Baseline Comparison",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    chart_path = save_dir / "stage_a_comparison.png"
    plt.savefig(str(chart_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Chart] Saved → {chart_path}")


# ══════════════════════════════════════════════════════════════════
# 4. Markdown 리포트 생성
# ══════════════════════════════════════════════════════════════════

def _flat_summary(s: dict) -> dict:
    """fallback_used(dict) 제거 후 반환."""
    out = dict(s)
    out.pop("fallback_used", None)
    return out


def save_markdown_report(results: dict, save_path: Path):
    datasets = list(results.keys())
    methods  = list(next(iter(results.values())).keys())

    lines = [
        "# Stage A — Palm Localization Baseline Comparison",
        "",
        "## 실험 설정",
        "- **비교 방법**: " + ", ".join(methods),
        "- **평가 데이터셋**: " + ", ".join(datasets),
        "- **GT 전략**: MediaPipe Hands Pseudo-GT (검출 실패 이미지 제외)",
        "",
        "## 평가 지표 정의",
        "| 지표 | 설명 |",
        "|------|------|",
        "| IoU@0.5 | 예측 bbox와 GT bbox의 IoU ≥ 0.5인 이미지 비율 |",
        "| IoU@0.75 | 예측 bbox와 GT bbox의 IoU ≥ 0.75인 이미지 비율 |",
        "| Mean IoU | 전체 이미지의 평균 IoU (실패 시 IoU=0 포함) |",
        "| FPS | total_images / total_inference_time |",
        "| Model Size | os.path.getsize(weights) / 1024 / 1024 (MB) |",
        "",
    ]

    for ds in datasets:
        lines += [
            f"## 데이터셋: {ds}",
            "",
            "| Method | N | Failed | Mean IoU | Acc@0.5 | Acc@0.75 | FPS | Size(MB) |",
            "|--------|---|--------|----------|---------|----------|-----|---------|",
        ]
        for method in methods:
            s = _flat_summary(results[ds][method])
            lines.append(
                f"| {method} "
                f"| {s['n_total']} "
                f"| {s['n_failed']} "
                f"| {s['mean_iou']:.3f} "
                f"| {s['accuracy@0.5']:.3f} "
                f"| {s['accuracy@0.75']:.3f} "
                f"| {s['fps']:.1f} "
                f"| {s['model_size_mb']:.2f} |"
            )
        lines.append("")

    lines += [
        "## 크로스-데이터셋 평균",
        "",
        "| Method | Mean IoU | Acc@0.5 | Acc@0.75 | FPS | Size(MB) |",
        "|--------|----------|---------|----------|-----|---------|",
    ]
    for method in methods:
        agg = {k: [] for k in ["mean_iou", "accuracy@0.5", "accuracy@0.75", "fps", "model_size_mb"]}
        for ds in datasets:
            s_flat = _flat_summary(results[ds][method])
            for k in agg:
                agg[k].append(s_flat.get(k, 0.0))
        avg = {k: sum(v) / len(v) for k, v in agg.items()}
        lines.append(
            f"| {method} "
            f"| {avg['mean_iou']:.3f} "
            f"| {avg['accuracy@0.5']:.3f} "
            f"| {avg['accuracy@0.75']:.3f} "
            f"| {avg['fps']:.1f} "
            f"| {avg['model_size_mb']:.2f} |"
        )

    lines += [
        "",
        "---",
        "_자동 생성: run_stage_a_comparison.py_",
    ]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Report] Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_eval", action="store_true",
                   help="평가를 먼저 실행 후 집계")
    p.add_argument("--dataset", nargs="+",
                   default=["BMPD", "Tongji", "MPDv2"],
                   help="평가 데이터셋 (run_eval 모드에서만 사용)")
    p.add_argument("--max_images", type=int, default=None,
                   help="데이터셋당 최대 이미지 수")
    p.add_argument("--json", default=str(RESULT_JSON),
                   help="결과 JSON 파일 경로")
    p.add_argument("--no_chart", action="store_true",
                   help="차트 생성 건너뜀")
    p.add_argument("--no_report", action="store_true",
                   help="마크다운 리포트 생성 건너뜀")
    return p.parse_args()


def main():
    args = parse_args()

    # 평가 실행 옵션
    if args.run_eval:
        cmd = [
            sys.executable,
            str(ROOT / "experiments" / "stage_a_baseline_eval.py"),
            "--dataset", *args.dataset,
            "--save_json", str(RESULT_JSON),
            "--save_csv",  str(RESULT_CSV),
            "--visualize",
        ]
        if args.max_images:
            cmd += ["--max_images", str(args.max_images)]
        print(f"[Running] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # 결과 로드
    results = load_results(Path(args.json))

    # 콘솔 출력
    print_summary(results)

    # 차트
    if not args.no_chart:
        plot_comparison(results, CHART_DIR)

    # 마크다운 리포트
    if not args.no_report:
        report_path = CHART_DIR / "stage_a_report.md"
        save_markdown_report(results, report_path)

    print("\n[All Done] Results aggregated.")


if __name__ == "__main__":
    main()
