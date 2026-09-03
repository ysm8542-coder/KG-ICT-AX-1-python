import os
import csv
import json
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
import serial
import cv2
from flask import Flask, request, jsonify, send_from_directory, Response
from predict_single_image import predict_single_image
from mask_utils import create_colored_defect_mask
import Nucleo_bridge
 
 
app = Flask(__name__)
BASE_DIR = os.environ.get("PROJECT_BASE_DIR", r"D:\B-2 Spirit\Battery_dataset\DHT_wireless_update")
 
MODEL_PATH = os.environ.get("VISION_MODEL_PATH", os.path.join(BASE_DIR, "saved_model", "defect_model.keras"))
SCALER_PATH = os.environ.get("VISION_SCALER_PATH", os.path.join(BASE_DIR, "saved_model", "scaler.pkl"))
THRESHOLD = float(os.environ.get("VISION_THRESHOLD", "0.5"))
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("LOW_CONFIDENCE_THRESHOLD", "0.7"))
 
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        f"모델 파일을 찾을 수 없습니다: {MODEL_PATH} / {SCALER_PATH} \n"
        f"먼저 train.py로 모델을 학습하고 저장하세요"
    )
 
 
Nucleo_bridge.connect()
 
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MASK_DIR = os.path.join(BASE_DIR, "masks")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REPORT_DIR = os.path.join(BASE_DIR, "report")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
CSV_FIELDS = ["timestamp", "filename", "label", "confidence", "temperature", "humidity"]
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
 
 
# --- 참고용 모델 성능 지표 ---
# 주의: 아래 파일들은 이 서버가 실제로 쓰는 defect_model.keras(Keras 분류 모델)가 아니라,
# 별도 프로젝트의 세그멘테이션 모델(seg_model.pt)의 검증된 test 성능이다. 같은 결함 유형
# (크랙/박리/핀홀)을 다루는 자매 모델이라 "참고용" 감각 지표로만 화면에 보여주고,
# defect_model.keras 자체의 성능 파일이 확보되면 REFERENCE_PERF_DIR에 넣어 교체하면 된다.
REFERENCE_PERF_DIR = os.environ.get(
    "REFERENCE_PERF_DIR",
    os.path.join(BASE_DIR, "project", "project", "CoatingVision", "outputs"),
)
REFERENCE_PERF_FILES = {
    "deployed": "seg_deployed_final_metrics.json",
    "freeze": "seg_freeze_combined_results.json",
}
# 참고 모델의 클래스명(영문) -> 이 서버의 실제 판정 라벨(한글) 매핑
REFERENCE_LABEL_MAP = {
    "Surface_Crack": "크랙",
    "Delamination": "박리",
    "Pinhole": "핀홀",
    "unclassified": "미분류(참고)",
}
 
lock = threading.Lock()
 
session_rows = []
session_start_dt = datetime.now()
 
latest= {
    "filename" : None,
    "mask_filename" : None,
    "label" : None,
    "confidence" : None,
    "timestamp" : None,
    "temperature" : None,
    "humidity" : None,
    "class_breakdown" : [],
}
 
DEFECT_LABELS = ["정상", "핀홀", "박리", "크랙"]
 
EN_TO_KOR = {
    "pinhole": "핀홀",
    "peeling": "박리",
    "crack": "크랙",
}
 
 
 
def load_reference_perf():
    """참고용 성능 지표 로드. defect_model.keras 자체 지표가 아니라 자매 세그멘테이션 모델의
    검증된 test 성능이므로, 화면에는 반드시 '참고용/다른 모델' 표시와 함께 노출해야 한다."""
    data = {}
    for key, fname in REFERENCE_PERF_FILES.items():
        path = os.path.join(REFERENCE_PERF_DIR, fname)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data[key] = json.load(f)
            except Exception:
                data[key] = None
        else:
            data[key] = None
    return data
 
 
#-------------------------------------
#비전 AI 연결지점
def classify_defect(image_path: str):
    res = predict_single_image(
        image_path=image_path,
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        threshold=THRESHOLD,
    )
    probs = res["probabilities"]  #pinhole:0.1, peeling: 0.05, crack:0.92
 
    if res["is_normal"]:
        label = "정상"
        confidence = float(1.0 - max(probs.values())) if probs else 1.0
    else:
        triggered = [cls for cls, p in probs.items() if p >= THRESHOLD]
        if triggered:
            label = ";".join(EN_TO_KOR.get(cls, cls) for cls in triggered)
            confidence = float(max(probs[cls]for cls in triggered))
        else:
            label = res["result"]
            confidence = float(max(probs.values())) if probs else 0.0
 
    # 결함 유형별 확신도 breakdown (dashboard_app.py의 "판정 결과" 표와 같은 형식) —
    # 최종 라벨/신뢰도 하나로 뭉개기 전에, 클래스별 확률을 그대로 남겨서 화면에 같이 보여주기 위함
    breakdown = []
    for cls, p in probs.items():
        kor = EN_TO_KOR.get(cls, cls)
        breakdown.append({
            "결함유형": kor,
            "판정": "🔴 발견" if p >= THRESHOLD else "미발견",
            "신뢰도": round(float(p) * 100, 1),
        })
    breakdown.sort(key=lambda r: -r["신뢰도"])
 
    return label, confidence, breakdown
#-------------------------------------
 
def generate_mask_image(image_path: str, ts: str):
    """원본 이미지를 읽어 create_colored_defect_mask로 색칠된 결함 마스크를 만들고
    MASK_DIR에 저장한다. 대시보드 실시간 판정 화면에서 원본 사진 옆에 같이 보여주기 위함.
    실패해도 판정 자체(label/confidence)는 막지 않도록 예외를 흡수하고 None을 반환한다."""
    try:
        bgr_image = cv2.imread(image_path)
        if bgr_image is None:
            return None
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        colored_mask = create_colored_defect_mask(rgb_image)
        mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
 
        mask_filename = f"{ts}_mask.jpg"
        mask_path = os.path.join(MASK_DIR, mask_filename)
        cv2.imwrite(mask_path, mask_bgr)
        return mask_filename
    except Exception as e:
        print(f"마스크 생성 실패({e})")
        return None
 
def append_result(row: dict):
    file_exists = os.path.isfile(RESULTS_CSV)
    with lock:
        with open(RESULTS_CSV, "a", newline="", encoding = "utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
 
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        session_rows.append(row)
 
@app.route("/upload", methods=["POST"])
def uploade():
    image_bytes = request.get_data()
    if not image_bytes:
        return jsonify({"error": "empty body"}),400
 
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    filename = f"{ts}.jpg"
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)
 
    label, confidence, breakdown = classify_defect(filepath)
    #사진이 도착한 시점에 바로 Nucleo에 온습도 요청 -> 같은 timestamp로 묶임
    temperature, humidity = Nucleo_bridge.read_temp_humidity()
 
    #Nucleo로 판정 결과 전송 -> LED 구동
    Nucleo_bridge.send_result(label)
 
    # 결함 위치를 색으로 보여주는 마스크 이미지 생성 (원본 사진 옆에 같이 띄우기 위함)
    mask_filename = generate_mask_image(filepath, ts)
 
    row = {
        "timestamp" : ts,
        "filename" : filename,
        "label" : label,
        "confidence" : confidence,
        "temperature" : temperature,
        "humidity" : humidity
    }
    append_result(row)
    
    latest.update(row)
    latest["mask_filename"] = mask_filename
    latest["class_breakdown"] = breakdown  # 사진이 올라올 때마다 클래스별 확신도까지 같이 갱신
 
    time.sleep(3)
 
    return jsonify({"result" : label, "confidence" : confidence})
 
@app.route("/latest.jpg")
def latest_jpg():
    if not latest["filename"]:
        return "", 404
 
    return send_from_directory(UPLOAD_DIR, latest["filename"])
 
@app.route("/latest_mask.jpg")
def latest_mask_jpg():
    if not latest["mask_filename"]:
        return "", 404
 
    return send_from_directory(MASK_DIR, latest["mask_filename"])
 
@app.route("/reference_perf")
def reference_perf():
    data = load_reference_perf()
    data["_label_map"] = REFERENCE_LABEL_MAP
    data["_note"] = (
        "이 지표는 defect_model.keras가 아니라 자매 프로젝트의 세그멘테이션 모델(seg_model.pt) "
        "test 성능입니다. 같은 결함 유형을 다루지만 다른 모델이므로 참고용으로만 확인하세요."
    )
    # 파일을 못 찾았을 때 어디를 봤는지 화면에서 바로 알 수 있도록, 실제 검사 경로/파일명을 항상 같이 내려준다.
    data["_dir"] = REFERENCE_PERF_DIR
    data["_expected_files"] = {
        key: os.path.join(REFERENCE_PERF_DIR, fname) for key, fname in REFERENCE_PERF_FILES.items()
    }
    return jsonify(data)
 
@app.route("/latest_result")
def latest_result():
    return jsonify(latest)
 
@app.route("/stats")
def stats():
    with lock:
        rows_copy = list(session_rows)
    return jsonify(compute_session_stats(rows_copy))
 
@app.route("/")
def viewer():
    html = """
    <html>
    <head>
      <meta charset="utf-8">
      <title>배터리 코팅 결함 검사 대시보드</title>
      <style>
        body { font-family: sans-serif; margin: 20px; background:#f5f6f8; }
        h2 { margin-bottom: 4px; }
        .grid { display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start; }
        .card { background:#fff; border:1px solid #ddd; border-radius:8px; padding:16px; box-shadow:0 1px 3px rgba(0,0,0,0.06); }
        .live-card { display:flex; gap:32px; width:100%; max-width:1200px; padding:28px; align-items:flex-start; }
        .live-image { flex: 0 0 46%; display:flex; flex-direction:column; gap:10px; }
        .live-image img { width:100%; border-radius:8px; border:1px solid #ccc; display:block; background:#111; }
        .live-image .img-label { font-size:12px; color:#888; margin:0 0 -4px 2px; }
        .live-info { flex:1; min-width:0; text-align:left; }
        .live-info h3 { font-size:32px; margin:0 0 12px 0; }
        .live-info p { font-size:18px; margin:8px 0; }
        .live-info .section-title { font-size:19px; }
        .live-info table { font-size:16px; }
        .live-info th, .live-info td { padding:10px 12px; }
        .stat-card { flex:1; min-width:260px; }
        .stat-row { display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #f0f0f0; }
        table { width:100%; border-collapse: collapse; font-size: 13px; }
        th, td { text-align:left; padding:6px 8px; border-bottom:1px solid #eee; white-space:nowrap; }
        th { background:#fafafa; }
        .defect { color:#c0392b; font-weight:bold; }
        .normal { color:#2e7d32; }
        .low-conf { color:#d68910; }
        .section-title { margin-top:24px; font-size:16px; font-weight:bold; }
        .tabs { display:flex; gap:4px; margin-bottom:16px; border-bottom:2px solid #ddd; }
        .tab-btn { padding:10px 18px; border:none; background:none; cursor:pointer; font-size:14px; color:#888; border-bottom:3px solid transparent; }
        .tab-btn.active { color:#1a1a1a; font-weight:bold; border-bottom:3px solid #2e7d32; }
        .tab-panel { display:none; }
        .tab-panel.active { display:block; }
        .warn-banner { background:#fff8e1; border:1px solid #ffca28; border-radius:6px; padding:10px 14px; font-size:13px; margin-bottom:16px; color:#7a5c00; }
        .toggle-row { display:flex; align-items:center; gap:8px; margin-bottom:10px; }
        .small-btn { padding:6px 12px; border:1px solid #bbb; background:#fff; border-radius:5px; cursor:pointer; font-size:13px; }
        .small-btn:hover { background:#f0f0f0; }
        .perf-stat-row { font-size:16px; }
      </style>
      <script>
        function fmtPct(v) { return (v == null) ? '-' : v.toFixed(1) + '%'; }
 
        function showTab(name) {
          document.querySelectorAll('.tab-panel').forEach(el => el.classList.remove('active'));
          document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
          document.getElementById('tab-' + name).classList.add('active');
          document.getElementById('btn-' + name).classList.add('active');
        }
 
        function refreshLive() {
          fetch('/latest_result').then(r => r.json()).then(d => {
            document.getElementById('label').innerText = d.label || '대기 중...';
            document.getElementById('conf').innerText = d.confidence != null ? (d.confidence*100).toFixed(1) + '%' : '-';
            document.getElementById('ts').innerText = d.timestamp || '-';
            document.getElementById('temp').innerText = d.temperature != null ? d.temperature + ' °C' : '-';
            document.getElementById('hum').innerText = d.humidity != null ? d.humidity + ' %' : '-';
            if (d.filename) document.getElementById('img').src = '/latest.jpg?' + Date.now();
            if (d.mask_filename) document.getElementById('mask_img').src = '/latest_mask.jpg?' + Date.now();
 
            const breakdownBody = document.getElementById('class_breakdown_body');
            breakdownBody.innerHTML = '';
            if (d.class_breakdown && d.class_breakdown.length > 0) {
              d.class_breakdown.forEach(r => {
                const isDefect = r['판정'] !== '미발견';
                breakdownBody.innerHTML += `<tr>
                  <td class="${isDefect ? 'defect' : 'normal'}">${r['결함유형']}</td>
                  <td class="${isDefect ? 'defect' : 'normal'}">${r['판정']}</td>
                  <td>${r['신뢰도'].toFixed(1)}%</td>
                </tr>`;
              });
            } else {
              breakdownBody.innerHTML = '<tr><td colspan="3">아직 판정 결과가 없어요</td></tr>';
            }
          });
        }
 
        function rowClass(label) {
          return (label && label.startsWith('정상')) ? 'normal' : 'defect';
        }
 
        function refreshStats() {
          fetch('/stats').then(r => r.json()).then(s => {
            document.getElementById('session_start').innerText = s.session_start;
            document.getElementById('total').innerText = s.total;
            document.getElementById('normal_count').innerText = s.normal_count + ' (' + fmtPct(s.normal_pct) + ')';
            document.getElementById('defect_count').innerText = s.defect_count + ' (' + fmtPct(s.defect_pct) + ')';
            document.getElementById('avg_conf').innerText = s.avg_confidence != null ? s.avg_confidence + '%' : '-';
            document.getElementById('temp_stat').innerText = s.temp_avg != null ? ('평균 ' + s.temp_avg + '°C (' + s.temp_min + '~' + s.temp_max + ')') : '데이터 없음';
            document.getElementById('hum_stat').innerText = s.hum_avg != null ? ('평균 ' + s.hum_avg + '% (' + s.hum_min + '~' + s.hum_max + ')') : '데이터 없음';
            document.getElementById('missing_env').innerText = s.missing_env + '건 (' + fmtPct(s.missing_env_pct) + ')';
 
            const labelBody = document.getElementById('label_counts_body');
            labelBody.innerHTML = '';
            s.label_counts.forEach(([label, cnt]) => {
              labelBody.innerHTML += `<tr><td class="${rowClass(label)}">${label}</td><td>${cnt}</td></tr>`;
            });
 
            const recentBody = document.getElementById('recent_body');
            recentBody.innerHTML = '';
            s.recent_rows.forEach(r => {
              recentBody.innerHTML += `<tr>
                <td>${r.timestamp}</td>
                <td class="${rowClass(r.label)}">${r.label}</td>
                <td>${(r.confidence*100).toFixed(1)}%</td>
                <td>${r.temperature ?? '-'}</td>
                <td>${r.humidity ?? '-'}</td>
              </tr>`;
            });
 
            const lowConfBody = document.getElementById('low_conf_body');
            lowConfBody.innerHTML = '';
            if (s.low_confidence_rows.length === 0) {
              lowConfBody.innerHTML = `<tr><td colspan="3">신뢰도 ${s.low_confidence_threshold_pct}% 이하 판정 없음</td></tr>`;
            } else {
              s.low_confidence_rows.forEach(r => {
                lowConfBody.innerHTML += `<tr class="low-conf"><td>${r.timestamp}</td><td>${r.label}</td><td>${(r.confidence*100).toFixed(1)}%</td></tr>`;
              });
            }
 
            const defectBody = document.getElementById('defect_recent_body');
            defectBody.innerHTML = '';
            if (s.defect_recent_rows.length === 0) {
              defectBody.innerHTML = `<tr><td colspan="4">불량 없음</td></tr>`;
            } else {
              s.defect_recent_rows.forEach(r => {
                defectBody.innerHTML += `<tr>
                  <td>${r.timestamp}</td>
                  <td class="defect">${r.label}</td>
                  <td>${r.temperature ?? '-'}°C</td>
                  <td>${r.humidity ?? '-'}%</td>
                </tr>`;
              });
            }
          });
        }
 
        // --- 모델 성능(참고용) ---
        function fmtPctVal(v) {
          return (typeof v === 'number') ? (v * 100).toFixed(1) + '%' : '-';
        }
 
        function loadReferencePerf() {
          fetch('/reference_perf').then(r => r.json()).then(data => {
            const box = document.getElementById('perf_content');
            const deployed = data.deployed;
            if (!deployed) {
              const expected = data._expected_files ? data._expected_files.deployed : null;
              box.innerHTML = '<p>참고 성능 파일을 찾지 못했어요.</p>' +
                (expected ? `<p style="font-size:12px; color:#888;">다음 경로에 파일이 있는지 확인해주세요:<br><code>${expected}</code></p>` : '') +
                '<p style="font-size:12px; color:#888;">경로 자체를 바꾸고 싶으면 서버 실행 전에 REFERENCE_PERF_DIR 환경변수를 설정하세요.</p>';
              return;
            }
            const labelMap = data._label_map;
            const tm = deployed.test_metrics;
            let rowsHtml = '';
            const f1s = [], precisions = [], recalls = [];
            let totalSupport = 0;
            for (const [enCls, korCls] of Object.entries(labelMap)) {
              if (!tm[enCls]) continue;
              const r = tm[enCls];
              rowsHtml += `<tr>
                <td>${korCls}</td>
                <td>${fmtPctVal(r.f1)}</td>
                <td>${fmtPctVal(r.precision)}</td>
                <td>${fmtPctVal(r.recall)}</td>
                <td>${r.support_pos ?? '-'}</td>
              </tr>`;
              if (typeof r.f1 === 'number') f1s.push(r.f1);
              if (typeof r.precision === 'number') precisions.push(r.precision);
              if (typeof r.recall === 'number') recalls.push(r.recall);
              if (typeof r.support_pos === 'number') totalSupport += r.support_pos;
            }
            const avg = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
            const macroF1 = (typeof tm.macro_f1 === 'number') ? tm.macro_f1 : avg(f1s);
            const macroPrecision = (typeof tm.macro_precision === 'number') ? tm.macro_precision : avg(precisions);
            const macroRecall = (typeof tm.macro_recall === 'number') ? tm.macro_recall : avg(recalls);
            const accuracy = (typeof tm.accuracy === 'number') ? tm.accuracy
              : (typeof deployed.accuracy === 'number' ? deployed.accuracy : null);
 
            let statsHtml = `<div class="stat-row perf-stat-row"><span>참고 모델 macro F1</span><span><b>${fmtPctVal(macroF1)}</b></span></div>`;
            statsHtml += `<div class="stat-row perf-stat-row"><span>참고 모델 macro Precision</span><span><b>${fmtPctVal(macroPrecision)}</b></span></div>`;
            statsHtml += `<div class="stat-row perf-stat-row"><span>참고 모델 macro Recall</span><span><b>${fmtPctVal(macroRecall)}</b></span></div>`;
            if (accuracy != null) {
              statsHtml += `<div class="stat-row perf-stat-row"><span>참고 모델 Accuracy</span><span><b>${fmtPctVal(accuracy)}</b></span></div>`;
            }
            statsHtml += `<div class="stat-row perf-stat-row"><span>test 표본 수(양성 합계)</span><span><b>${totalSupport}</b></span></div>`;
 
            box.innerHTML = statsHtml + `
              <table style="margin-top:14px;">
                <thead><tr><th>결함 유형</th><th>F1</th><th>Precision</th><th>Recall</th><th>test 양성 수</th></tr></thead>
                <tbody>${rowsHtml}</tbody>
              </table>
            `;
          });
        }
 
        function refreshAll() { refreshLive(); refreshStats(); loadReferencePerf(); }
        setInterval(refreshAll, 2000);
        window.onload = function() {
          refreshAll();
        };
      </script>
    </head>
    <body>
      <h2>배터리 코팅 결함 실시간 검사 대시보드</h2>
 
      <div class="tabs">
        <button class="tab-btn active" id="btn-live" onclick="showTab('live')">🔴 실시간 판정</button>
        <button class="tab-btn" id="btn-history" onclick="showTab('history')">📋 이력·통계</button>
        <button class="tab-btn" id="btn-perf" onclick="showTab('perf')">📊 모델 성능</button>
      </div>
 
      <div class="tab-panel active" id="tab-live">
        <div class="grid">
          <div class="card live-card">
            <div class="live-image">
              <div class="img-label">원본</div>
              <img id="img" src="/latest.jpg">
              <div class="img-label">결함 마스크 (빨강=핀홀, 초록=박리, 파랑=크랙)</div>
              <img id="mask_img" src="/latest_mask.jpg">
            </div>
            <div class="live-info">
              <h3 id="label">대기 중...</h3>
              <p>신뢰도: <span id="conf">-</span></p>
              <p>온도: <span id="temp">-</span> / 습도: <span id="hum">-</span></p>
              <p>촬영 시각: <span id="ts">-</span></p>
 
              <div class="section-title" style="margin-top:16px; text-align:left;">결함 유형별 판정 결과</div>
              <table style="text-align:left;">
                <thead><tr><th>결함 유형</th><th>판정</th><th>신뢰도</th></tr></thead>
                <tbody id="class_breakdown_body"></tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
 
      <div class="tab-panel" id="tab-history">
        <div class="grid">
          <div class="card stat-card">
            <div class="section-title" style="margin-top:0;">세션 요약</div>
            <div class="stat-row"><span>세션 시작</span><span id="session_start">-</span></div>
            <div class="stat-row"><span>총 검사 수</span><span id="total">0</span></div>
            <div class="stat-row"><span>양품</span><span id="normal_count" class="normal">-</span></div>
            <div class="stat-row"><span>불량</span><span id="defect_count" class="defect">-</span></div>
            <div class="stat-row"><span>평균 신뢰도</span><span id="avg_conf">-</span></div>
            <div class="stat-row"><span>온도</span><span id="temp_stat">-</span></div>
            <div class="stat-row"><span>습도</span><span id="hum_stat">-</span></div>
            <div class="stat-row"><span>센서 결측치</span><span id="missing_env">-</span></div>
 
            <div class="section-title">라벨별 판정 결과</div>
            <table><tbody id="label_counts_body"></tbody></table>
          </div>
        </div>
 
        <div class="card" style="margin-top:20px;">
          <div class="section-title" style="margin-top:0;">최근 판정 이력 (최신순)</div>
          <table>
            <thead><tr><th>시각</th><th>라벨</th><th>신뢰도</th><th>온도</th><th>습도</th></tr></thead>
            <tbody id="recent_body"></tbody>
          </table>
        </div>
 
        <div class="grid" style="margin-top:20px;">
          <div class="card stat-card">
            <div class="section-title" style="margin-top:0;">저신뢰도 판정 목록</div>
            <table>
              <thead><tr><th>시각</th><th>라벨</th><th>신뢰도</th></tr></thead>
              <tbody id="low_conf_body"></tbody>
            </table>
          </div>
          <div class="card stat-card">
            <div class="section-title" style="margin-top:0;">최근 불량 발생 이력</div>
            <table>
              <thead><tr><th>시각</th><th>라벨</th><th>온도</th><th>습도</th></tr></thead>
              <tbody id="defect_recent_body"></tbody>
            </table>
          </div>
        </div>
      </div>
 
      <div class="tab-panel" id="tab-perf">
        <div class="warn-banner">
          ⚠️ 아래 수치는 <b>defect_model.keras(이 서버가 실제로 쓰는 모델)의 성능이 아니라</b>,
          같은 결함 유형(크랙/박리/핀홀)을 다루는 <b>다른 프로젝트의 세그멘테이션 모델</b>이 test set에서 검증한 성능이에요.
          감을 잡는 참고용으로만 봐주세요 — defect_model.keras 자체의 test 결과 파일이 준비되면 이 자리에 바로 교체할 수 있어요.
          <br><br>
          또한 두 모델은 접근 방식 자체가 달라요: 참고 지표의 세그멘테이션 모델은 이미지를 CNN에 직접 넣어 픽셀 단위로 예측하지만,
          <b>defect_model.keras는 cv2 규칙 기반으로 컬러 마스크를 먼저 만들고 거기서 뽑은 15개 수치 특징(면적비율·픽셀수·덩어리 개수 등)을
          얕은 신경망에 넣는 구조</b>예요. 그래서 이 숫자가 실제 배포 모델의 강점/약점을 그대로 반영한다고 보기 어려워요.
        </div>
        <div class="card">
          <div id="perf_content">불러오는 중...</div>
        </div>
      </div>
    </body>
    </html>
    """
    return Response(html, mimetype="text/html")
 
 
def write_session_csv(rows: list, path: str):
    """이번 실행에서 촬영된 결과물만 담은, 한 번 생성되는 스냅샷 """
    with open(path, "w", newline="", encoding="utf-8-sig")as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
 
def compute_session_stats(rows: list, recent_n: int = 20):
    """generate_report()와 같은 지표를 '파일로 안 쓰고' dict로 반환 — 세션 종료 전에도(실시간으로) 호출 가능.
    /stats 엔드포인트와 실시간 뷰어 페이지에서 공용으로 사용."""
    total = len(rows)
    normal_rows = [r for r in rows if str(r.get("label", "")).startswith("정상")]
    defect_rows = [r for r in rows if not str(r.get("label", "")).startswith("정상")]
    normal_count = len(normal_rows)
    defect_count = len(defect_rows)
 
    label_counts = {}
    for r in rows:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
 
    env_rows = [r for r in rows if r.get("temperature") not in (None, "") and r.get("humidity") not in (None, "")]
    temps = [float(r["temperature"]) for r in env_rows]
    hums = [float(r["humidity"]) for r in env_rows]
 
    conf_rows = [r for r in rows if r.get("confidence") not in (None, "")]
    confidences = [float(r["confidence"]) for r in conf_rows]
    avg_confidence = sum(confidences) / len(confidences) if confidences else None
    low_confidence_rows = sorted(
        [r for r in conf_rows if float(r["confidence"]) < LOW_CONFIDENCE_THRESHOLD],
        key=lambda r: float(r["confidence"]),
    )
 
    return {
        "session_start": session_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "total": total,
        "normal_count": normal_count,
        "defect_count": defect_count,
        "normal_pct": round(normal_count / total * 100, 1) if total else None,
        "defect_pct": round(defect_count / total * 100, 1) if total else None,
        "label_counts": sorted(label_counts.items(), key=lambda x: -x[1]),
        "temp_avg": round(sum(temps) / len(temps), 1) if temps else None,
        "temp_min": round(min(temps), 1) if temps else None,
        "temp_max": round(max(temps), 1) if temps else None,
        "hum_avg": round(sum(hums) / len(hums), 1) if hums else None,
        "hum_min": round(min(hums), 1) if hums else None,
        "hum_max": round(max(hums), 1) if hums else None,
        "missing_env": total - len(env_rows),
        "missing_env_pct": round((total - len(env_rows)) / total * 100, 1) if total else None,
        "avg_confidence": round(avg_confidence * 100, 1) if avg_confidence is not None else None,
        "low_confidence_threshold_pct": int(LOW_CONFIDENCE_THRESHOLD * 100),
        "low_confidence_rows": low_confidence_rows[-10:][::-1],  # 최근 신뢰도 낮은 순 최대 10건
        "recent_rows": rows[-recent_n:][::-1],  # 최신순
        "defect_recent_rows": defect_rows[-10:][::-1],
    }
 
 
def generate_report(rows: list, report_path: str, session_csv_name: str):
    """새션 결과를 요약한 텍스트 리포트 생성"""
    total = len(rows)
    normal_rows = [r for r in rows if str(r.get("label", "")).startswith("정상")]
    defect_rows = [r for r in rows if not str(r.get("label", "")).startswith("정상")]
    normal_count = len(normal_rows)
    defect_count = len(defect_rows)
 
    label_counts = {}
    for r in rows:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
 
    env_rows = [r for r in rows if r.get("temperature") not in (None, "") and r.get("humidity") not in (None, "")]
    temps = [float(r["temperature"]) for r in env_rows]
    hums = [float(r["humidity"]) for r in env_rows]
 
    # --- 신뢰도(confidence) 통계 ---
    conf_rows = [r for r in rows if r.get("confidence") not in (None, "")]
    confidences = [float(r["confidence"]) for r in conf_rows]
    avg_confidence = sum(confidences) / len(confidences) if confidences else None
    low_confidence_rows = sorted(
        [r for r in conf_rows if float(r["confidence"]) < LOW_CONFIDENCE_THRESHOLD],
        key=lambda r: float(r["confidence"]),
    )
 
    lines = []
    lines.append("=" * 50)
    lines.append("배터리 코팅 결함 검사 - 새션 리포트")
    lines.append("=" * 50)
    lines.append(f"새션 시작 : {session_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"새션 종료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"새션 스냅샷 CSV: {session_csv_name}")
    lines.append("")
    lines.append(f"총 검사수: {total} 개")
    if total:
        lines.append(f"양품: {normal_count}개 ({normal_count/total * 100: .1f})")
        lines.append(f"불량: {normal_count}개 ({defect_count/total * 100: .1f})")
    lines.append("")
    lines.append("{라벨링 판정 결과}")
    for label, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        lines.append(f" - {label}: {cnt} 개")
    lines.append("")
    lines.append("{환경 통계}")
    if temps:
        lines.append(f" 온도: 평균 {sum(temps)/len(temps): .1f}°C (최소 {min(temps):.1f} / 최대{max(temps):.1f})")
        lines.append(f" 습도: 평균 {sum(hums)/len(temps): .1f}% (최소 {min(hums):.1f} / 최대{max(hums):.1f})")
    else:
        lines.append(" 온습도 데이터 없음")
    missing_env = total - len(env_rows)
    lines.append(f" 센서 결측치: {missing_env}건 ({missing_env/total*100:.1f}%)"if total else " 센서 결측치: 0건")
    lines.append("")
    lines.append("{판정 신뢰도 통계}")
    if avg_confidence is not None:
        lines.append(f" 평균 신뢰도: {avg_confidence*100:.1f}%")
    else:
        lines.append(" 신뢰도 데이터 없음")
    lines.append("")
    lines.append(f"{{신뢰도 {int(LOW_CONFIDENCE_THRESHOLD*100)}% 이하 판정 목록}}")
    if low_confidence_rows:
        for r in low_confidence_rows:
            lines.append(
                f" {r['timestamp']} {r['filename']} {r['label']} 신뢰도:{float(r['confidence'])*100:.1f}%"
            )
    else:
        lines.append(f" 신뢰도 {int(LOW_CONFIDENCE_THRESHOLD*100)}% 이하인 판정 없음")
    lines.append("")
    lines.append("{결함 발생 시각 및 환경}")
    if defect_rows:
        for r in defect_rows:
            temp = r.get("temperature") or "-"
            hum = r.get("humidity") or "-"
            lines.append(f" {r['timestamp']} {r['label']} 온도:{temp}°C 습도:{hum}%")
    else:
        lines.append(" 불량없음")
    lines.append("="*50)
 
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
 
def shutdown_handler(signum, frame):
    print("\nCtrl+C 감지됨 -> 세션 마무리 중...")
 
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_csv_path = os.path.join(RESULTS_DIR, f"results_{ts}.csv")
    report_path = os.path.join(REPORT_DIR, f"report_{ts}.txt")
 
    write_session_csv(session_rows, session_csv_path)
    generate_report(session_rows, report_path, session_csv_path)
    print(f"세션 CSV 생성: {session_csv_path}")
    print(f"리포트 생성: {report_path}")
 
    try:
        Nucleo_bridge.close()
    except Exception:
        pass
 
    # streamlit 대시보드도 항상 이 프로젝트 폴더(BASE_DIR)의 데이터를 보도록
    # DATA_DIR 환경변수와 최신 세션 CSV 파일명을 넘겨준다.
    env = os.environ.copy()
    env["LATEST_SESSION_CSV"] = os.path.basename(session_csv_path)
    env["DATA_DIR"] = RESULTS_DIR
    dashboard_path = os.path.join(BASE_DIR, "dashboard.py")
 
    # Windows에서는 Ctrl+C(SIGINT)가 콘솔의 프로세스 그룹 전체로 전달되어
    # 방금 띄운 streamlit 프로세스까지 같이 죽어버릴 수 있다.
    # CREATE_NEW_PROCESS_GROUP으로 분리해서 그걸 막는다.
    popen_kwargs = {"env": env, "cwd": BASE_DIR}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
 
    try:
        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", dashboard_path],
            **popen_kwargs,
        )
        print(f"Streamlit 대시보드 실행중 (데이터 폴더: {RESULTS_DIR})")
    except Exception as e:
        print(f"Streamlit 실행 실패({e}) 수동으로 'streamlit run dashboard.py' 실행하세요")
 
    os._exit(0)
 
signal.signal(signal.SIGINT, shutdown_handler)
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)