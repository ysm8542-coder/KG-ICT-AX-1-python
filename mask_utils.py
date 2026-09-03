"""
mask_utils.py
-------------
지금까지 대화를 통해 튜닝한 create_colored_defect_mask 함수를 그대로 담고 있습니다.
원본 RGB 이미지를 입력받아, 결함 종류별로 색이 칠해진 마스크 이미지를 반환합니다.

색상 규칙:
    빨강 (255, 0, 0) : 핀홀 (pinhole)
    초록 (0, 255, 0) : 박리 (peeling)
    파랑 (0, 0, 255) : 크랙 (crack)
    아무 색도 없음    : 결함 없음
"""

import cv2
import numpy as np

def create_colored_defect_mask(rgb_image):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
 
    green_mask = np.zeros_like(gray)
    red_mask = np.zeros_like(gray)
    blue_mask = np.zeros_like(gray)
 
    # ---------------------------------------------------------
    # 0. [어두운 결함: 크랙 & 박리] 마스크를 먼저 계산
    # ---------------------------------------------------------
    background = cv2.GaussianBlur(gray_blur, (51, 51), 0)
    normalized = cv2.divide(gray_blur, background, scale=255)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)
 
    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, blackhat_kernel)
 
    mean_val = np.mean(blackhat)
    std_val = np.std(blackhat)
    dynamic_thresh = int(mean_val + 2.0 * std_val)
    threshold_value = max(15, min(dynamic_thresh, 35))
 
    _, dark_mask = cv2.threshold(blackhat, threshold_value, 255, cv2.THRESH_BINARY)
 
    def remove_small_components(mask, min_area):
        num_l, lbls, sts, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out_mask = np.zeros_like(mask)
        for j in range(1, num_l):
            if sts[j, cv2.CC_STAT_AREA] >= min_area:
                out_mask[lbls == j] = 255
        return out_mask
 
    def remove_small_and_round_components(binary, min_area=35):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        cleaned = np.zeros_like(binary)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
 
            if area < min_area:
                continue
 
            short_side = min(width, height)
            if short_side <= 2 and area < 60:
                continue
 
            aspect_ratio = max(width, height) / max(short_side, 1)
            if aspect_ratio < 1.15 and area < 150:
                continue
 
            cleaned[labels == i] = 255
        return cleaned
 
    dark_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, dark_open_kernel, iterations=2)
 
    dark_mask = remove_small_components(dark_mask, min_area=30)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, close_kernel)
    dark_mask = remove_small_and_round_components(dark_mask, min_area=35)
 
    adjacency_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    # (dark_mask_dilated는 크랙/박리 최종 분류가 끝난 뒤, 실제 확정된 결함만으로 아래에서 재계산합니다.
    #  최초 dark_mask에는 노이즈성 컴포넌트가 아직 섞여있어, 그걸 기준으로 인접성을 판단하면
    #  진짜 핀홀이 "노이즈 옆에 있다"는 이유만으로 잘못 버려지는 문제가 있었습니다 - 수정 10)
 
    num_labels_dark, labels_dark, stats_dark, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
 
    crack_labels = []
    peeling_candidate_labels = []
 
    for i in range(1, num_labels_dark):
        area = stats_dark[i, cv2.CC_STAT_AREA]
        if area < 20:
            continue
 
        width = stats_dark[i, cv2.CC_STAT_WIDTH]
        height = stats_dark[i, cv2.CC_STAT_HEIGHT]
        short_side = max(min(width, height), 1)
        long_side = max(width, height)
        elongation = long_side / short_side
 
        comp_mask_dark = (labels_dark == i).astype(np.uint8)
        dist = cv2.distanceTransform(comp_mask_dark, cv2.DIST_L2, 5)
        max_thickness = 2.0 * dist.max()
 
        # 💡 수정 16 (크랙 교차점 반사광 이미지에서 발견): elongation은 bounding-box
        #    기반이라 "대각선 방향" 크랙 조각에서 또다시 실패함(대각선은 가로/세로 폭이
        #    비슷해서 낮게 나옴 - Y분기 문제와 동일한 유형의 함정).
        #    실측: 대각선 크랙 조각 elongation=1.08~1.31(낮음)인데 circularity는
        #    0.095~0.180(매우 낮음, 진짜 얇은 선의 특징). 반면 수정9의 노이즈 블롭은
        #    뭉툭해서 circularity가 높음(~0.7~0.9). circularity는 회전에 영향을 받지
        #    않으므로 elongation 대신(또는 함께) 사용해 방향에 무관하게 "선인지 블롭인지"
        #    판별.
        contours, _ = cv2.findContours(comp_mask_dark * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dark_circularity = 0.0
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            if perimeter > 0:
                dark_circularity = 4 * np.pi * area / (perimeter ** 2)
 
        is_thin_line = max_thickness <= 12.0
        is_elongated = elongation >= 2.2
        # 💡 임계값 0.3 -> 0.2로 조정: image_80의 노이즈 블롭(circ=0.259)이 실수로
        #    "선 형태"로 인정되어 핀홀 근처 인접성 판정을 오염시키는 문제 발견.
        #    실제 대각선 크랙 조각들의 circularity(0.095~0.180)는 0.2 밑으로 충분히
        #    낮아 이 조정으로도 정상적으로 크랙으로 인정됨.
        is_line_like = dark_circularity < 0.2
 
        # 💡 수정 9 (실제 핀홀 샘플에서 발견): is_thin_line이 "길쭉함"을 전혀 요구하지 않아서,
        #    작고 동그란 노이즈 블롭도 "전체 크기가 작아 두께도 자연히 얇다"는 이유만으로
        #    크랙으로 오분류되는 문제 발견 (예: 12x10, 18x13 크기의 뭉툭한 노이즈 블롭이
        #    두께<=12를 만족해 크랙으로 잡힘 -> 하필 그 근처 진짜 핀홀까지 "크랙에 붙었다"고
        #    오판되어 통째로 버려짐).
        #    -> "얇다"는 것만으로는 부족하고, 선 형태(원형도가 낮음)여야 진짜 크랙 선으로 인정.
        #    (elongation 조건도 보조적으로 함께 유지: 축에 나란한 뚜렷한 직선은 그것만으로도 통과)
        is_real_crack_shape = (is_thin_line and (is_line_like or elongation >= 1.5)) or is_elongated
 
        if is_real_crack_shape:
            crack_labels.append(i)
        elif area >= 300:
            peeling_candidate_labels.append(i)
 
    for i in crack_labels:
        blue_mask[labels_dark == i] = 255
 
    crack_adjacency_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    crack_mask_dilated = cv2.dilate(blue_mask, crack_adjacency_kernel)
 
    for i in peeling_candidate_labels:
        comp = (labels_dark == i)
        touches_crack = np.any(crack_mask_dilated[comp] > 0)
        if touches_crack:
            blue_mask[comp] = 255
        else:
            green_mask[comp] = 255
 
    # 💡 수정 10: 노이즈가 이미 걸러진 "최종 확정 결함"(blue_mask + green_mask)만으로
    #    dark_mask_dilated를 다시 계산 -> 밝은 결함(핀홀)의 인접성 판정이
    #    더 이상 노이즈성 컴포넌트에 영향받지 않음
    confirmed_dark_defect = cv2.bitwise_or(blue_mask, green_mask)
    dark_mask_dilated = cv2.dilate(confirmed_dark_defect, adjacency_kernel)
 
    # ---------------------------------------------------------
    # 1. [밝은 결함: 핀홀 & 밝은 박리] 처리
    # ---------------------------------------------------------
    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    bright_tophat = cv2.morphologyEx(gray_blur, cv2.MORPH_TOPHAT, tophat_kernel)
 
    b_mean = np.mean(bright_tophat)
    b_std = np.std(bright_tophat)
    bright_thresh = int(b_mean + 3.0 * b_std)
    bright_thresh = max(20, min(bright_thresh, 60))
 
    _, bright_mask = cv2.threshold(bright_tophat, bright_thresh, 255, cv2.THRESH_BINARY)
 
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, noise_kernel)
 
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 15:
            continue
 
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        short_side = max(min(width, height), 1)
        aspect_ratio = max(width, height) / short_side
 
        comp_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circularity = 0.0
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
 
        touches_dark_defect = np.any(dark_mask_dilated[labels == i] > 0)
 
        # 💡 수정 8 (실제 핀홀 샘플로 보정): AND 조건이 너무 엄격해서 진짜 핀홀을 놓치는 사례 발견.
        #    예: area=232, aspect_ratio=1.533(기준 1.5 살짝 초과), circularity=0.796(기준 0.6 충분히 통과)
        #    -> AND라서 전체 탈락 -> 박리(초록)로 오분류됨.
        #    실제 핀홀은 완벽한 원이 아니라 약간 길쭉한 반짝임/스미어로 나타나는 경우가 많으므로,
        #    "종횡비가 크지 않거나(<3.0) 원형도가 충분히 높으면(>0.4)" 둘 중 하나만 만족해도 인정(OR).
        #    ⚠️ touches_dark_defect 처리는 그대로 유지 (크랙 인접 반사광 오검출 회귀 방지, 검증 완료)
        is_round_and_small = (aspect_ratio < 3.0 or circularity > 0.4) and area < 500
 
        # 💡 수정 14 (순수 크랙 이미지 8장으로 검증): 크랙 가장자리가 빛을 살짝
        #    반사하는 자연스러운 하이라이트가 최대 ~1100px까지 나타날 수 있음을 확인.
        #    기존 임계값(200)이 너무 낮아 이런 하이라이트가 "크지만 독립적인 박리"로
        #    오분류됨. 실제 독립 반사광 박리 사례(면적 2030px)와 크랙 하이라이트
        #    노이즈(최대 1119px) 사이에 여유가 있어 200 -> 1500으로 상향.
        LARGE_BRIGHT_DEFECT_AREA = 1500
 
        # 💡 수정 15: 대각선 방향의 얇고 긴 하이라이트는 bounding-box 종횡비로는
        #    안 걸러짐 (대각선은 가로/세로 폭이 비슷해서 elongation이 낮게 나옴).
        #    크랙 분류 때 썼던 것과 동일하게 distance transform 기반 "실제 두께"로 판별.
        #    실측: 크랙 하이라이트 두께 6.4px vs 진짜 반사광 박리 두께 38.4px (6배 차이).
        comp_mask_bright = (labels == i).astype(np.uint8)
        bright_dist = cv2.distanceTransform(comp_mask_bright, cv2.DIST_L2, 5)
        bright_thickness = 2.0 * bright_dist.max()
        MIN_PEELING_THICKNESS = 15.0
 
        if is_round_and_small and not touches_dark_defect:
            red_mask[labels == i] = 255
        elif touches_dark_defect and area < LARGE_BRIGHT_DEFECT_AREA:
            pass
        elif bright_thickness < MIN_PEELING_THICKNESS:
            pass  # 가늘고 긴 하이라이트 선 -> 독립 박리 아님, 버림
        else:
            green_mask[labels == i] = 255
 
    # ---------------------------------------------------------
    # 1-2. [넓게 퍼진 박리: 완전 포화(saturation) 기반 검출]
    # ---------------------------------------------------------
    # 💡 수정 13: tophat은 구조 요소보다 넓은 영역은 원천적으로 못 잡음(내부가 0에 가까움).
    #    커널을 키우거나 임계값을 낮추면 일반 표면 텍스처까지 오검출되는 회귀가 실측으로
    #    확인됨(크랙 전용 이미지 8장 검증). 대신 훨씬 안전한 신호를 사용:
    #    실측 결과, 순수 크랙 이미지 8장은 원본 밝기 최댓값이 151~198에 그치고 250 이상
    #    픽셀이 전혀 없었던 반면, 실제 넓은 박리 이미지는 최소 10%~30% 픽셀이 완전
    #    포화(255)에 도달함. 이는 이미지마다 다른 절대 밝기에 흔들리지 않는 물리적
    #    특성(반사가 심한 벗겨짐 표면은 카메라 센서를 포화시킴)이라 안전하게 쓸 수 있음.
    #    작은 핀홀 판정(위 섹션)과는 독립적으로, 넓은 영역에만 적용.
    _, saturation_mask = cv2.threshold(gray_blur, 245, 255, cv2.THRESH_BINARY)
    sat_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    saturation_mask = cv2.morphologyEx(saturation_mask, cv2.MORPH_OPEN, sat_open_kernel)
 
    num_labels_sat, labels_sat, stats_sat, _ = cv2.connectedComponentsWithStats(saturation_mask, connectivity=8)
    for i in range(1, num_labels_sat):
        area = stats_sat[i, cv2.CC_STAT_AREA]
        # 작은 반짝임(핀홀 등)은 위 tophat 섹션에서 이미 처리하므로, 여기서는
        # "넓은 영역"만 대상으로 함 (노이즈성 소규모 포화 픽셀 제외)
        if area < 300:
            continue
        # 💡 얇고 긴 포화 선(예: 강한 반사 줄) 오분류 방지 위해 동일하게 두께 체크
        comp_mask_sat = (labels_sat == i).astype(np.uint8)
        sat_dist = cv2.distanceTransform(comp_mask_sat, cv2.DIST_L2, 5)
        if 2.0 * sat_dist.max() < 15.0:
            continue
        green_mask[labels_sat == i] = 255
 
    # ---------------------------------------------------------
    # 3. 겹치는 부분 정리 및 색상 매핑
    # ---------------------------------------------------------
    combined_peeling_and_pinhole = cv2.bitwise_or(green_mask, red_mask)
    blue_mask = cv2.bitwise_and(blue_mask, cv2.bitwise_not(combined_peeling_and_pinhole))
 
    colored_mask[red_mask == 255] = [255, 0, 0]     # 핀홀
    colored_mask[green_mask == 255] = [0, 255, 0]   # 박리
    colored_mask[blue_mask == 255] = [0, 0, 255]    # 크랙
 
    return colored_mask