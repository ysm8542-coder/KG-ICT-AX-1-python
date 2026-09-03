"""
train.py
---------
사용법:
    python train.py --csv path/to/labels.csv --images_dir path/to/images \
                     --output_dir path/to/save_model

CSV 형식은 dataset_utils.py 상단 docstring 참고 (filename, labels 컬럼).
"""

import argparse
import json
import os
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset_utils import load_dataset
from feature_extraction import FEATURE_NAMES, DEFECT_CLASSES
from model_utils import build_model, train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="라벨 CSV 경로 (filename, labels 컬럼)")
    parser.add_argument("--images_dir", required=True, help="원본 이미지 폴더")
    parser.add_argument("--output_dir", default="./saved_model", help="모델/스케일러 저장 폴더")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("1) 데이터셋 로드 (이미지 -> 마스크 -> 특징 추출) ...")
    dataset = load_dataset(csv_path=args.csv, images_dir=args.images_dir)

    print("2) train/validation 분할 ...")
    X_train, X_val, Y_train, Y_val = train_test_split(
        dataset.X, dataset.Y, test_size=args.test_size, random_state=args.seed
    )

    print("3) 특징 스케일링 (StandardScaler) ...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("4) 모델 생성 및 학습 ...")
    model = build_model(input_dim=dataset.X.shape[1], num_classes=len(DEFECT_CLASSES))
    model.summary()

    # 데이터 비율을 역산하여 클래스별 가중치(class_weight) 자동 계산
    counts = Y_train.sum(axis=0)
    total =len(Y_train)
    class_weights = {}
    print('\n[클래스 가중치 적용]')
    for i in range(len(DEFECT_CLASSES)):
        # 정답 개수가 적을수록 가중치(weight)가 커지는 공식 적용
        # weight = total / (len(DEFECT_CLASSES)) * (counts[i] + 1)
        weight = total / (len(DEFECT_CLASSES)) * (counts[i] + 1)        # <=========가중치 수정함
        class_weights[i] = float(weight)
        print(f"  - {DEFECT_CLASSES[i]} 가중치: {class_weights[i]:.2f} (데이터 수: {int(counts[i])})")
    print("-" * 30)

    history = train_model(
        model, X_train_scaled, Y_train, X_val_scaled, Y_val,
        epochs=args.epochs, batch_size=args.batch_size,
    )

    print("5) 저장 ...")
    model_path = os.path.join(args.output_dir, "defect_model.keras")
    scaler_path = os.path.join(args.output_dir, "scaler.pkl")
    meta_path = os.path.join(args.output_dir, "meta.json")

    model.save(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_names": FEATURE_NAMES,
                "defect_classes": DEFECT_CLASSES,
                "final_val_loss": float(history.history["val_loss"][-1]),
                "final_val_auc": float(history.history["val_auc"][-1]),
            },
            f, ensure_ascii=False, indent=2,
        )

    print(f"완료. 모델: {model_path}\n스케일러: {scaler_path}\n메타정보: {meta_path}")


if __name__ == "__main__":
    main()
