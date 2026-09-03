"""
demo_dryrun_without_tf.py
---------------------------
이 스크립트는 tensorflow / shap이 설치되지 않은 환경(예: 이 샌드박스)에서도
'최종 산출물이 어떤 모습일지'를 실제 이미지로 미리 확인하기 위한 데모입니다.

- 마스크 생성(mask_utils), 특징 추출(feature_extraction), 오버레이 이미지 생성,
  그래프 생성(plot_per_image_shap, plot_overall_summary), CSV 저장 로직은
  explain_and_report.py의 실제 코드를 그대로 재사용합니다.
- 다만 "모델 예측 확률"과 "SHAP 값"은 실제 학습된 모델이 없으므로,
  데모 라벨(demo_labels.csv)을 흉내 낸 가짜 값으로 대체했습니다.
  (실제 tensorflow/shap 환경에서는 explain_and_report.run_report()를 그대로 쓰면 됩니다.)
"""

import os
import sys
import types

# --- tensorflow / shap이 없는 환경에서도 explain_and_report의 순수 함수들을
#     import할 수 있도록 더미 모듈을 끼워넣음 (실제 배포 환경에서는 필요 없음) ---
if "tensorflow" not in sys.modules:
    fake_tf = types.ModuleType("tensorflow")
    fake_tf.keras = types.SimpleNamespace(models=types.SimpleNamespace(load_model=lambda *a, **k: None))
    sys.modules["tensorflow"] = fake_tf
if "shap" not in sys.modules:
    fake_shap = types.ModuleType("shap")
    fake_shap.kmeans = lambda *a, **k: None
    fake_shap.KernelExplainer = object
    sys.modules["shap"] = fake_shap

import cv2
import numpy as np
import pandas as pd

from dataset_utils import load_dataset, multihot_to_label_string
from feature_extraction import DEFECT_CLASSES, extract_mask_features
from explain_and_report import overlay_mask_on_original, plot_per_image_shap, plot_overall_summary, group_shap_by_defect

IMAGES_DIR = "/mnt/user-data/uploads"
CSV_PATH = "demo_labels.csv"
OUTPUT_DIR = "./demo_outputs"


def fake_predict_prob(y_true: np.ndarray, seed: int) -> np.ndarray:
    """실제 모델이 없으므로, 정답 라벨에 노이즈를 살짝 섞어 '그럴듯한 예측 확률'을 흉내낸다."""
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, 0.12, size=y_true.shape)
    probs = y_true * 0.85 + (1 - y_true) * 0.10 + noise
    return np.clip(probs, 0.01, 0.99)


def fake_shap_matrix(features: np.ndarray, probs: np.ndarray, seed: int) -> np.ndarray:
    """실제 SHAP 대신, '각 결함 그룹의 자기 특징 크기'에 비례하는 그럴듯한 영향도 행렬을 만든다."""
    rng = np.random.RandomState(seed)
    n_groups = len(DEFECT_CLASSES)
    n_feat = features.shape[0]
    per_group = n_feat // n_groups
    shap_matrix = np.zeros((n_groups, n_feat))
    for out_i in range(n_groups):
        for g in range(n_groups):
            start, end = g * per_group, (g + 1) * per_group
            base = np.abs(features[start:end]).sum()
            weight = 1.0 if g == out_i else 0.15  # 자기 그룹이 대부분의 영향을 주도록
            shap_matrix[out_i, start:end] = (base * weight / per_group) + rng.normal(0, 0.01, per_group)
    return shap_matrix


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    masks_dir = os.path.join(OUTPUT_DIR, "masks")
    charts_dir = os.path.join(OUTPUT_DIR, "shap_charts")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    dataset = load_dataset(csv_path=CSV_PATH, images_dir=IMAGES_DIR, keep_masks=True, keep_originals=True)

    rows = []
    all_group_matrices = []

    for i, fname in enumerate(dataset.filenames):
        rgb = dataset.originals[i]
        colored_mask = dataset.masks[i]
        features = dataset.X[i]
        y_true = dataset.Y[i]

        probs = fake_predict_prob(y_true, seed=i)
        predicted_label_str = multihot_to_label_string(probs, threshold=0.5)

        overlay = overlay_mask_on_original(rgb, colored_mask, alpha=0.6)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(masks_dir, fname), overlay_bgr)

        shap_matrix = fake_shap_matrix(features, probs, seed=i)
        group_matrix = group_shap_by_defect(shap_matrix)
        all_group_matrices.append(group_matrix)

        positive_idx = [j for j, p in enumerate(probs) if p >= 0.5]
        if positive_idx:
            top_contributors = []
            for out_i in positive_idx:
                top_group = DEFECT_CLASSES[int(np.argmax(group_matrix[out_i]))]
                top_contributors.append(f"{DEFECT_CLASSES[out_i]}<-{top_group}")
            top_contributor_str = "; ".join(top_contributors)
        else:
            top_contributor_str = "N/A (정상)"

        chart_path = os.path.join(charts_dir, f"{os.path.splitext(fname)[0]}_shap.png")
        plot_per_image_shap(group_matrix, chart_path, title=f"{fname} - 예측: {predicted_label_str}")

        row = {"filename": fname, "predicted_labels": predicted_label_str, "true_labels(demo)": multihot_to_label_string(y_true, 0.5)}
        for k, cls in enumerate(DEFECT_CLASSES):
            row[f"{cls}_probability"] = float(probs[k])
        row["top_contributing_feature_group"] = top_contributor_str
        rows.append(row)

    report_df = pd.DataFrame(rows)
    csv_out_path = os.path.join(OUTPUT_DIR, "report.csv")
    report_df.to_csv(csv_out_path, index=False, encoding="utf-8-sig")

    summary_path = os.path.join(OUTPUT_DIR, "shap_summary_overall.png")
    plot_overall_summary(all_group_matrices, summary_path)

    print(report_df.to_string(index=False))
    print(f"\n저장 위치: {OUTPUT_DIR}/ (masks/, shap_charts/, report.csv, shap_summary_overall.png)")


if __name__ == "__main__":
    main()
