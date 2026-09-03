from typing import Union, Dict

import cv2
import pickle
import numpy as np
import tensorflow as tf

from mask_utils import create_colored_defect_mask
from feature_extraction import extract_mask_features, DEFECT_CLASSES
from dataset_utils import multihot_to_label_string

_MODEl_CACHE: dict = {}

def _load_model_and_scaler(model_path: str, scaler_path: str):
    key = (model_path, scaler_path)
    if key not in _MODEl_CACHE:
        model = tf.keras.models.load_model(model_path)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        _MODEl_CACHE[key] = (model, scaler)
        print(f"[predict_single_image] 모델 로드(캐시 저장): {model_path}")
    return _MODEl_CACHE[key]

def predict_single_image(
        image_path: str,
        model_path: str,
        scaler_path: str,
        threshold: Union[float, Dict[str, float]] = 0.5
) -> dict:
    """
    단일 이미지 한 장을 입력받아 결함 여부 및 각 결함별 확률을 판별합니다.

    threshold: 모든 클래스 공통 float 값이거나, 클래스별 dict(예: 온습도 조건에 따라
               일부 클래스만 임계값을 낮춘 dict)일 수 있습니다. 이 함수는 threshold를
               그대로 multihot_to_label_string에 전달하기만 하며 원본 값은 변경하지 않습니다.

    반환값 (dict):
        - result: 'normal' 또는 결함 이름 (예: 'crack', 'pinhole;peeling')
        - is_normal: 정상 여부 (True / False)
        - probabilities: 각 결함 클래스별 예측 확률 (dict)
    """
    #이미지 읽기 및 RGB 변환
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없거나 읽을 수 없습니다: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    #마스트 생성 및 15개 수치 특징 추출
    colored_mask = create_colored_defect_mask(rgb)
    features = extract_mask_features(colored_mask)

    #모델 및 스케일러 로드
    model = tf.keras.models.load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    #특징 스케일링 후 딥러닝 모델 예측
    x_scaled = scaler.transform([features])
    probs = model.predict(x_scaled, verbose=0)[0]

    #확률 기반 라벨 판별(normal 혹은 결함명)
    result_label = multihot_to_label_string(probs, threshold=threshold)

    #각 결함별 확률 딕셔너리 생성
    prob_dict = {cls: float(prob) for cls, prob in zip(DEFECT_CLASSES, probs)}

    return {
        "result" : result_label,
        "is_normal" : result_label == "normal",
        "probabilities" : prob_dict
    }

if __name__ == "__main__":
    test_img = "sample.png"
    test_model_path = "./saved_model/defect_model.keras"
    test_scaler_path = "./saved_model/scaler.pkl"
 
    res = predict_single_image(test_img, test_model_path, test_scaler_path)
    print(f"판별 결과 : {res['result']}")
    print(f"정상 여부 : {res['is_normal']}")
    print(f"결함 확률 : {res['probabilities']}")