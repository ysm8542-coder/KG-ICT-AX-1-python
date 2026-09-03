"""
explain_and_report.py
-----------------------
학습된 모델로 각 이미지를 판별하고, SHAP으로 "어떤 결함 종류가 판별에 가장 큰 영향을
줬는지" 설명한 뒤, 아래 세 가지 결과물을 저장합니다.

  1. outputs/masks/<파일명>            : 마스크가 오버레이된 이미지
  2. outputs/shap_charts/<파일명>.png  : 이미지별 SHAP 기여도 막대그래프
  3. outputs/report.csv                : 전체 이미지에 대한 판별 결과 요약 표
  4. outputs/shap_summary_overall.png  : 전체 데이터셋 기준, 결함 종류별 평균 영향도 그래프

사용법:
    python explain_and_report.py --model_dir ./saved_model \
                                  --images_dir path/to/images \
                                  --csv path/to/labels_or_filelist.csv \
                                  --output_dir ./outputs
"""

import argparse
import json
import os
import pickle
from typing import List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import shap
import tensorflow as tf

from dataset_utils import multihot_to_label_string
from feature_extraction import DEFECT_CLASSES, FEATURES_PER_CLASS, extract_mask_features
from mask_utils import create_colored_defect_mask

# 💡 그래프에 한글 라벨을 쓰므로, 시스템에 있는 한글 지원 폰트(Noto Sans CJK 등)를 찾아 적용.
#    폰트가 없는 환경에서는 자동으로 기본 폰트로 대체되며(글자만 깨질 뿐 실행에는 문제 없음),
#    폰트를 찾으면 한글이 정상적으로 렌더링됨.
def _setup_korean_font():
    # 한글은 유니코드 상 지역 변이가 없으므로, CJK 폰트 패밀리 중 아무거나(JP/KR 등) 잡혀도
    # 한글 자체는 정상적으로 렌더링됨. matplotlib의 ttc 폰트 이름 인식이 배포판마다
    # 다르므로(예: "Noto Sans CJK KR" 대신 "Noto Sans CJK JP"로만 잡히는 경우가 있음)
    # 폰트 파일을 직접 등록해서 실제 등록된 이름을 사용한다.
    candidate_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
                name = fm.FontProperties(fname=path).get_name()
                matplotlib.rcParams["font.family"] = name
                matplotlib.rcParams["axes.unicode_minus"] = False
                return name
            except Exception:
                continue

    candidates = ["Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic", "AppleGothic"]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    return None


_setup_korean_font()


def overlay_mask_on_original(rgb: np.ndarray, colored_mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """원본 이미지 위에 결함 마스크를 반투명하게 겹쳐서 보여주기 좋은 시각화 이미지를 만든다."""
    overlay = rgb.copy()
    mask_nonzero = np.any(colored_mask > 0, axis=-1)
    blended = cv2.addWeighted(rgb, 1 - alpha, colored_mask, alpha, 0)
    overlay[mask_nonzero] = blended[mask_nonzero]
    return overlay


def _normalize_shap_values(raw_shap, num_outputs: int, num_features: int) -> np.ndarray:
    """
    shap 라이브러리 버전에 따라 shap_values() 반환 형태가 다름:
      - list of length num_outputs, 각 (1, num_features)
      - ndarray shape (1, num_features, num_outputs)
      - ndarray shape (1, num_features)  (단일 출력인 경우)
    이를 (num_outputs, num_features) 형태로 통일한다.
    """
    if isinstance(raw_shap, list):
        arr = np.stack([np.asarray(v).reshape(-1) for v in raw_shap], axis=0)
    else:
        arr = np.asarray(raw_shap)
        if arr.ndim == 3:
            # (1, num_features, num_outputs) -> (num_outputs, num_features)
            arr = arr[0].T
        elif arr.ndim == 2:
            if arr.shape[0] == num_outputs:
                pass
            else:
                arr = arr.reshape(num_outputs, num_features)
        else:
            arr = arr.reshape(num_outputs, num_features)
    return arr


def group_shap_by_defect(shap_matrix: np.ndarray) -> np.ndarray:
    """
    shap_matrix: (num_outputs, num_features) - 15개 특징을 5개씩 3그룹(pinhole/peeling/crack)으로 묶어
    절대값 합을 구한다 -> (num_outputs, 3) 반환
    """
    n_groups = len(DEFECT_CLASSES)
    n_feat_per_group = len(FEATURES_PER_CLASS)
    grouped = np.zeros((shap_matrix.shape[0], n_groups))
    for g in range(n_groups):
        start, end = g * n_feat_per_group, (g + 1) * n_feat_per_group
        grouped[:, g] = np.abs(shap_matrix[:, start:end]).sum(axis=1)
    return grouped


def plot_per_image_shap(group_matrix: np.ndarray, save_path: str, title: str):
    """group_matrix: (num_outputs=3, num_groups=3). 출력 클래스별로 그룹 기여도 막대그래프."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(DEFECT_CLASSES))
    width = 0.25
    for g, group_name in enumerate(DEFECT_CLASSES):
        ax.bar(x + g * width, group_matrix[:, g], width, label=f"{group_name} 특징의 영향")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{c}\n예측" for c in DEFECT_CLASSES])
    ax.set_ylabel("|SHAP| 합 (영향도)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_overall_summary(all_group_matrices: List[np.ndarray], save_path: str):
    """전체 이미지 평균 기준, '예측 클래스 x 기여 특징그룹' 히트맵/막대그래프."""
    stacked = np.stack(all_group_matrices, axis=0)  # (N, 3, 3)
    mean_matrix = stacked.mean(axis=0)  # (3 outputs, 3 groups)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mean_matrix, cmap="Reds")
    ax.set_xticks(range(len(DEFECT_CLASSES)))
    ax.set_xticklabels([f"{c} 특징" for c in DEFECT_CLASSES])
    ax.set_yticks(range(len(DEFECT_CLASSES)))
    ax.set_yticklabels([f"{c} 예측" for c in DEFECT_CLASSES])
    for i in range(mean_matrix.shape[0]):
        for j in range(mean_matrix.shape[1]):
            ax.text(j, i, f"{mean_matrix[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_title("전체 데이터셋 평균: 어떤 특징 그룹이 각 결함 예측에 영향을 줬는가")
    fig.colorbar(im, ax=ax, label="평균 |SHAP| 영향도")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def run_report(
    model_dir: str,
    images_dir: str,
    csv_path: str,
    output_dir: str,
    filename_col: str = "filename",
    threshold: float = 0.5,
    background_size: int = 30,
    shap_nsamples: int = 100,
):
    os.makedirs(output_dir, exist_ok=True)
    masks_dir = os.path.join(output_dir, "masks")
    charts_dir = os.path.join(output_dir, "shap_charts")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    # --- 모델 / 스케일러 로드 ---
    model = tf.keras.models.load_model(os.path.join(model_dir, "defect_model.keras"))
    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    df = pd.read_csv(csv_path)
    filenames = df[filename_col].astype(str).tolist()

    # --- 1차 패스: 모든 이미지의 특징을 미리 뽑아 SHAP 배경 데이터로 사용 ---
    print("1) 전체 이미지 특징 추출 중 (SHAP 배경 데이터 구성) ...")
    all_features, valid_filenames, all_masks, all_originals = [], [], [], []
    for fname in filenames:
        img_path = os.path.join(images_dir, fname)
        if not os.path.exists(img_path):
            print(f"[경고] 없는 파일, 건너뜀: {img_path}")
            continue
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"[경고] 읽기 실패, 건너뜀: {img_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        colored_mask = create_colored_defect_mask(rgb)
        features = extract_mask_features(colored_mask)

        valid_filenames.append(fname)
        all_features.append(features)
        all_masks.append(colored_mask)
        all_originals.append(rgb)

    if len(all_features) == 0:
        raise RuntimeError("처리할 이미지가 없습니다.")

    X_all = np.stack(all_features, axis=0)
    X_all_scaled = scaler.transform(X_all)

    # --- SHAP 배경 데이터: 너무 크면 느리므로 kmeans로 요약 ---
    n_bg = min(background_size, X_all_scaled.shape[0])
    background = shap.kmeans(X_all_scaled, n_bg) if X_all_scaled.shape[0] > n_bg else X_all_scaled

    print("2) SHAP Explainer 준비 중 (모델 예측 함수를 감싸서 사용) ...")
    predict_fn = lambda x: model.predict(x, verbose=0)
    explainer = shap.KernelExplainer(predict_fn, background)

    print(f"3) 이미지별 예측 + SHAP 설명 계산 ({len(valid_filenames)}장) ...")
    rows = []
    all_group_matrices = []

    for idx, fname in enumerate(valid_filenames):
        rgb = all_originals[idx]
        colored_mask = all_masks[idx]
        x_scaled = X_all_scaled[idx : idx + 1]

        probs = model.predict(x_scaled, verbose=0)[0]
        predicted_label_str = multihot_to_label_string(probs, threshold=threshold)

        # 마스크 오버레이 이미지 저장
        overlay = overlay_mask_on_original(rgb, colored_mask, alpha=0.6)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(masks_dir, fname), overlay_bgr)

        # SHAP 계산
        raw_shap = explainer.shap_values(x_scaled, nsamples=shap_nsamples, silent=True)
        shap_matrix = _normalize_shap_values(raw_shap, num_outputs=len(DEFECT_CLASSES), num_features=X_all.shape[1])
        group_matrix = group_shap_by_defect(shap_matrix)  # (3 outputs, 3 groups)
        all_group_matrices.append(group_matrix)

        # 이 이미지에서 검출된(임계값 넘긴) 결함들에 대해, 가장 영향을 많이 준 특징 그룹 찾기
        positive_idx = [i for i, p in enumerate(probs) if p >= threshold]
        if positive_idx:
            top_contributors = []
            for out_i in positive_idx:
                top_group = DEFECT_CLASSES[int(np.argmax(group_matrix[out_i]))]
                top_contributors.append(f"{DEFECT_CLASSES[out_i]}<-{top_group}")
            top_contributor_str = "; ".join(top_contributors)
        else:
            top_contributor_str = "N/A (정상)"

        # 이미지별 SHAP 막대그래프 저장
        chart_path = os.path.join(charts_dir, f"{os.path.splitext(fname)[0]}_shap.png")
        plot_per_image_shap(group_matrix, chart_path, title=f"{fname} - 예측: {predicted_label_str}")

        row = {"filename": fname, "predicted_labels": predicted_label_str}
        for i, cls in enumerate(DEFECT_CLASSES):
            row[f"{cls}_probability"] = float(probs[i])
        row["top_contributing_feature_group"] = top_contributor_str
        rows.append(row)

        if (idx + 1) % 5 == 0 or idx == len(valid_filenames) - 1:
            print(f"   {idx + 1}/{len(valid_filenames)} 완료")

    # --- 결과 CSV 저장 ---
    report_df = pd.DataFrame(rows)
    csv_out_path = os.path.join(output_dir, "report.csv")
    report_df.to_csv(csv_out_path, index=False, encoding="utf-8-sig")

    # --- 전체 요약 그래프 저장 ---
    summary_path = os.path.join(output_dir, "shap_summary_overall.png")
    plot_overall_summary(all_group_matrices, summary_path)

    print("완료!")
    print(f" - 마스크 오버레이 이미지: {masks_dir}/")
    print(f" - 이미지별 SHAP 그래프: {charts_dir}/")
    print(f" - 결과 요약 CSV: {csv_out_path}")
    print(f" - 전체 요약 그래프: {summary_path}")

    return report_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--csv", required=True, help="filename 컬럼이 있는 CSV (라벨 컬럼은 없어도 됨)")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    run_report(
        model_dir=args.model_dir,
        images_dir=args.images_dir,
        csv_path=args.csv,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
