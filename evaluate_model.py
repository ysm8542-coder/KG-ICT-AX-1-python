import argparse
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score, multilabel_confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_utils import load_dataset
from feature_extraction import DEFECT_CLASSES

def _setup_korean_font():
    import matplotlib.font_manager as fm
    candidates = ["Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic", "AppleGothic"]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    return None

_setup_korean_font()

def main():
    parser = argparse.ArgumentParser(description="학습된 결함 판별 모델 평가 및 결과 저장 스크립트")
    parser.add_argument("--model_dir", default="./saved_model", help="모델과 스케일러가 저장된 폴더")
    parser.add_argument("--images_dir", required=True, help="평가할 원본 이미지 폴더")
    parser.add_argument("--csv", required=True, help="평가할 라벨 CSV 파일")
    parser.add_argument("--threshold", type=float, default=0.5, help="양성 판정 임계값")
    parser.add_argument("--output_dir", default="./outputs", help="평가 결과물을 저장할 폴더")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("1) 평가 데이터셋 로드 ...")
    dataset = load_dataset(csv_path=args.csv, images_dir=args.images_dir)

    print("2) 저장된 모델 및 스케일러 불러오기 ...")
    model_path = os.path.join(args.model_dir, "defect_model.keras")
    scaler_path = os.path.join(args.model_dir, "scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"모델이나 스케일러 파일을 찾을 수 없습니다: {args.model_dir}")

    model = tf.keras.models.load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    print("3) 모델 예측 수행 ...")
    X_scaled = scaler.transform(dataset.X)
    Y_true = dataset.Y
    Y_pred_prob = model.predict(X_scaled)
    Y_pred_binary = (Y_pred_prob >= args.threshold).astype(int)

    print("4) 평가 결과 파일 저장 중 ...")
    
    # --- [1] 성능 지표 CSV 저장 ---
    report_dict = classification_report(Y_true, Y_pred_binary, target_names=DEFECT_CLASSES, zero_division=0, output_dict=True)
    
    try:
        auc_scores = roc_auc_score(Y_true, Y_pred_prob, average=None)
    except ValueError:
        auc_scores = [0.0] * len(DEFECT_CLASSES)
        
    metrics_rows = []
    for i, cls in enumerate(DEFECT_CLASSES):
        row = report_dict[cls]
        row['class'] = cls
        row['roc_auc'] = auc_scores[i]
        metrics_rows.append(row)
        
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df[['class', 'precision', 'recall', 'f1-score', 'roc_auc', 'support']]
    
    csv_path = os.path.join(args.output_dir, "evaluation_metrics.csv")
    metrics_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # --- [2] F1-Score & AUC 시각화 차트 저장 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DEFECT_CLASSES))
    width = 0.35

    f1_scores = metrics_df['f1-score']
    aucs = metrics_df['roc_auc']

    ax.bar(x - width/2, f1_scores, width, label='F1-Score', color='royalblue')
    ax.bar(x + width/2, aucs, width, label='ROC-AUC', color='darkorange')

    ax.set_ylabel('Score (0.0 ~ 1.0)')
    ax.set_title('결함 종류별 모델 성능 (F1-Score & ROC-AUC)')
    ax.set_xticks(x)
    ax.set_xticklabels(DEFECT_CLASSES)
    ax.legend()
    ax.set_ylim(0, 1.15)
    
    for i, v in enumerate(f1_scores):
        ax.text(i - width/2, v + 0.02, f"{v:.2f}", ha='center', fontsize=9)
    for i, v in enumerate(aucs):
        ax.text(i + width/2, v + 0.02, f"{v:.2f}", ha='center', fontsize=9)

    chart_path = os.path.join(args.output_dir, "evaluation_chart.png")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)

    # --- [3] 혼동 행렬 CSV 저장 ---
    mcm = multilabel_confusion_matrix(Y_true, Y_pred_binary)
    cm_rows = []
    for i, cls in enumerate(DEFECT_CLASSES):
        tn, fp, fn, tp = mcm[i].ravel()
        cm_rows.append({
            "class": cls,
            "True_Positive(결함_맞춤)": tp,
            "True_Negative(정상_맞춤)": tn,
            "False_Positive(오탐)": fp,
            "False_Negative(놓침)": fn
        })
    cm_df = pd.DataFrame(cm_rows)
    cm_csv_path = os.path.join(args.output_dir, "evaluation_confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path, index=False, encoding='utf-8-sig')

    print("==================================================")
    print("                [ 모델 평가 완료 ]                ")
    print("==================================================")
    print(f" - 성능 요약 CSV: {csv_path}")
    print(f" - 혼동 행렬 CSV: {cm_csv_path}")
    print(f" - 성능 비교 차트: {chart_path}")

if __name__ == "__main__":
    main()