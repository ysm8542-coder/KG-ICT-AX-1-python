"""
feature_extraction.py
----------------------
create_colored_defect_mask()가 만든 컬러 마스크(빨강=핀홀, 초록=박리, 파랑=크랙)에서
머신러닝 모델에 넣을 수 있는 '숫자 특징 벡터(feature vector)'를 뽑아냅니다.

왜 마스크를 그대로 CNN에 넣지 않고 특징을 뽑는가?
    - 이미 정교한 규칙 기반(cv2) 알고리즘으로 결함 위치/색을 검출해 놓았으므로,
      그 결과(마스크)를 요약한 소수의 해석 가능한 숫자로 변환하는 편이
      1) 학습 데이터가 적어도 안정적으로 학습되고
      2) SHAP으로 "어떤 결함이 얼마나 영향을 줬는지" 설명하기 쉽습니다.
      (SHAP이 각 특징에 대해 값을 매기는데, 특징 자체가 이미 결함별로 나뉘어 있으면
       "핀홀 관련 특징들의 SHAP 합계" = "핀홀이 예측에 준 영향" 으로 바로 해석 가능)

각 결함 클래스(pinhole/peeling/crack)마다 5개의 특징을 뽑습니다:
    1. area_ratio        : 전체 이미지 대비 해당 색 픽셀 비율 (0~1)
    2. pixel_count        : 해당 색 픽셀 개수 (log1p 스케일)
    3. num_components     : 분리된 결함 덩어리(connected component) 개수
    4. max_component_area  : 가장 큰 덩어리의 면적 (log1p 스케일)
    5. mean_component_area : 덩어리들의 평균 면적 (log1p 스케일)

따라서 총 특징 개수 = 3 (결함 종류) x 5 (특징) = 15개
"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

# 마스크에서 사용하는 색상 <-> 결함 이름 매핑 (mask_utils.create_colored_defect_mask 기준)
DEFECT_CLASSES: List[str] = ["pinhole", "peeling", "crack"]
DEFECT_COLORS = {
    "pinhole": (255, 0, 0),   # 빨강
    "peeling": (0, 255, 0),   # 초록
    "crack": (0, 0, 255),     # 파랑
}
FEATURES_PER_CLASS = ["area_ratio", "pixel_count_log", "num_components", "max_component_area_log", "mean_component_area_log"]

FEATURE_NAMES: List[str] = [
    f"{cls}_{feat}" for cls in DEFECT_CLASSES for feat in FEATURES_PER_CLASS
]  # 길이 15, SHAP 결과를 사람이 읽을 때 그대로 사용


def _extract_single_class_features(binary_mask: np.ndarray) -> List[float]:
    """binary_mask: 0/255 값의 단일 채널 마스크 (특정 결함 색상 하나에 대한 마스크)"""
    total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
    pixel_count = int(np.count_nonzero(binary_mask))
    area_ratio = pixel_count / total_pixels if total_pixels > 0 else 0.0

    if pixel_count == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])
    num_components = len(areas)
    max_area = float(areas.max()) if num_components > 0 else 0.0
    mean_area = float(areas.mean()) if num_components > 0 else 0.0

    return [
        float(area_ratio),
        float(np.log1p(pixel_count)),
        float(num_components),
        float(np.log1p(max_area)),
        float(np.log1p(mean_area)),
    ]


def extract_mask_features(colored_mask: np.ndarray) -> np.ndarray:
    """
    colored_mask: create_colored_defect_mask()의 반환값 (H, W, 3) RGB, 빨강/초록/파랑/검정만 존재

    반환: shape (15,) 짜리 float32 numpy 배열. 순서는 FEATURE_NAMES와 동일.
    """
    features: List[float] = []
    for cls in DEFECT_CLASSES:
        color = np.array(DEFECT_COLORS[cls], dtype=np.uint8)
        binary_mask = np.all(colored_mask == color, axis=-1).astype(np.uint8) * 255
        features.extend(_extract_single_class_features(binary_mask))
    return np.array(features, dtype=np.float32)


@dataclass
class MaskSummary:
    """이미지 한 장에 대한 마스크 요약 정보 (CSV 저장, 리포트용)"""
    detected_defects: List[str]     # 검출된 결함 종류 목록 (없으면 빈 리스트 -> "normal")
    area_ratio_by_class: dict       # {"pinhole": 0.01, "peeling": 0.0, "crack": 0.05}


def summarize_mask(colored_mask: np.ndarray, min_area_ratio: float = 1e-5) -> MaskSummary:
    """마스크만 보고 (모델 예측 없이) 어떤 결함이 '실제로 검출'되었는지 요약.
    학습 라벨 검증이나, 모델 예측과 별개로 순수 마스크 기반 판단이 필요할 때 사용."""
    detected = []
    ratios = {}
    for cls in DEFECT_CLASSES:
        color = np.array(DEFECT_COLORS[cls], dtype=np.uint8)
        binary_mask = np.all(colored_mask == color, axis=-1)
        ratio = float(np.count_nonzero(binary_mask)) / binary_mask.size
        ratios[cls] = ratio
        if ratio > min_area_ratio:
            detected.append(cls)
    return MaskSummary(detected_defects=detected, area_ratio_by_class=ratios)
