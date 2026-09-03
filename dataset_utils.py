"""
dataset_utils.py
------------------
CSV 라벨 파일 + 원본 이미지 폴더를 읽어서, 학습에 쓸 (X, Y)를 만듭니다.

기대하는 CSV 형식 (헤더 포함, 컬럼명은 자유롭게 지정 가능 - 아래 인자로 지정):
    filename,labels
    img001.png,crack
    img002.png,crack;peeling
    img003.png,normal
    img004.png,pinhole;crack

- labels 컬럼은 세미콜론(;)으로 구분된 결함 종류 목록입니다. (한 이미지에 여러 결함 가능)
- 결함이 전혀 없는 이미지는 "normal" 이라고 적어주세요 (또는 labels 칸을 비워둬도 됩니다).
- 결함 종류 이름은 pinhole / peeling / crack 셋 중 하나를 사용해주세요.
  (한국어 라벨 "핀홀"/"박리"/"크랙"을 쓰셔도 자동으로 변환됩니다 - KOR_TO_EN 참고)
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from feature_extraction import DEFECT_CLASSES, extract_mask_features, FEATURE_NAMES
from mask_utils import create_colored_defect_mask

KOR_TO_EN = {
    "핀홀": "pinhole",
    "박리": "peeling",
    "크랙": "crack",
    "정상": "normal",
}


def _parse_labels(label_str: str) -> List[str]:
    """'crack;peeling' 또는 '크랙;박리' 같은 문자열을 ['crack','peeling'] 리스트로 변환"""
    if label_str is None or (isinstance(label_str, float) and np.isnan(label_str)):
        return []
    label_str = str(label_str).strip()
    if label_str == "" or label_str.lower() == "normal" or label_str == "정상":
        return []

    raw_tokens = [t.strip() for t in label_str.replace(",", ";").split(";") if t.strip()]
    tokens = []
    for t in raw_tokens:
        t_norm = KOR_TO_EN.get(t, t.lower())
        if t_norm == "normal":
            continue
        if t_norm not in DEFECT_CLASSES:
            raise ValueError(
                f"알 수 없는 라벨 '{t}' (허용값: {DEFECT_CLASSES} 또는 한글 핀홀/박리/크랙/정상)"
            )
        tokens.append(t_norm)
    return tokens


def labels_to_multihot(labels: List[str]) -> np.ndarray:
    """['crack','peeling'] -> [pinhole=0, peeling=1, crack=1] 순서의 (3,) 배열"""
    vec = np.zeros(len(DEFECT_CLASSES), dtype=np.float32)
    for l in labels:
        vec[DEFECT_CLASSES.index(l)] = 1.0
    return vec


def multihot_to_label_string(multihot: np.ndarray, threshold=0.3) -> str:
    """[0.9, 0.1, 0.8] 같은 예측 확률 -> 'pinhole;crack' 문자열. 전부 임계값 미만이면 'normal'.
    threshold: 모든 클래스에 공통으로 쓰는 float 값이거나,
               {'pinhole': 0.4, 'peeling': 0.5, 'crack': 0.5} 형태의 클래스별 dict일 수 있음
               (예: 온습도 조건에 따라 특정 클래스만 임계값을 낮춘 경우)."""
    if isinstance(threshold, dict):
        thresholds = [threshold.get(cls, 0.3) for cls in DEFECT_CLASSES]
    else:
        thresholds = [threshold] * len(DEFECT_CLASSES)

    active = [DEFECT_CLASSES[i] for i, p in enumerate(multihot) if p >= thresholds[i]]
    return ";".join(active) if active else "normal"


@dataclass
class DefectDataset:
    filenames: List[str]
    X: np.ndarray               # (N, 15) 특징 행렬
    Y: np.ndarray                # (N, 3) 멀티핫 라벨
    masks: List[np.ndarray] = field(default_factory=list)   # 각 이미지의 컬러 마스크 (원할 때만 채움)
    originals: List[np.ndarray] = field(default_factory=list)  # 각 이미지의 원본 (원할 때만 채움)


def load_dataset(
    csv_path: str,
    images_dir: str,
    filename_col: str = "filename",
    labels_col: str = "labels",
    keep_masks: bool = False,
    keep_originals: bool = False,
    verbose: bool = True,
) -> DefectDataset:
    """
    csv_path    : 라벨 CSV 경로
    images_dir  : 원본 이미지들이 들어있는 폴더 (csv의 filename과 매칭)
    keep_masks  : True면 각 이미지의 마스크 결과도 함께 메모리에 보관 (리포트 생성 시 유용)
    """
    df = pd.read_csv(csv_path)
    if filename_col not in df.columns or labels_col not in df.columns:
        raise ValueError(
            f"CSV에 '{filename_col}', '{labels_col}' 컬럼이 있어야 합니다. 실제 컬럼: {list(df.columns)}"
        )

    filenames, X_list, Y_list, masks, originals = [], [], [], [], []

    for _, row in df.iterrows():
        fname = str(row[filename_col])
        img_path = os.path.join(images_dir, fname)
        if not os.path.exists(img_path):
            if verbose:
                print(f"[경고] 이미지 없음, 건너뜀: {img_path}")
            continue

        bgr = cv2.imread(img_path)
        if bgr is None:
            if verbose:
                print(f"[경고] 이미지를 읽을 수 없음, 건너뜀: {img_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        colored_mask = create_colored_defect_mask(rgb)
        features = extract_mask_features(colored_mask)
        labels = _parse_labels(row[labels_col])
        y = labels_to_multihot(labels)

        filenames.append(fname)
        X_list.append(features)
        Y_list.append(y)
        if keep_masks:
            masks.append(colored_mask)
        if keep_originals:
            originals.append(rgb)

    if len(X_list) == 0:
        raise RuntimeError("로드된 이미지가 하나도 없습니다. 경로와 CSV 내용을 확인하세요.")

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)

    if verbose:
        print(f"총 {len(filenames)}개 이미지 로드 완료. 특징 shape={X.shape}, 라벨 shape={Y.shape}")
        for i, cls in enumerate(DEFECT_CLASSES):
            print(f"  - {cls}: 양성 {int(Y[:, i].sum())}개 / 전체 {len(Y)}개")

    return DefectDataset(filenames=filenames, X=X, Y=Y, masks=masks, originals=originals)
