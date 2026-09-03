import os
import csv
import threading
from datetime import datetime

import serial
from flask import Flask, jsonify, send_from_directory, Response

from vision_model import load_model, predict as vision_predict
import Nucleo_bridge

SERIAL_PORT = "COM10"
BAUD_RATE = 921600

START_MARKER = bytes([0xAA, 0x55, 0xAA, 0x55])
END_MARKER   = bytes([0x55, 0xAA, 0x55, 0xAA])

UPLOAD_DIR = "uploads"
RESULTS_CSV = "results.csv"
os.makedirs(UPLOAD_DIR, exist_ok=True)
 
lock = threading.Lock()
latest = {"filename": None, "label": None, "confidence": None, "timestamp": None,
          "temperature": None, "humidity": None}
 
app = Flask(__name__)

def compute_checksum(data: bytes) -> int:
    return sum(data) & 0xFFFF

def append_result(row: dict):
    file_exists = os.path.isfile(RESULTS_CSV)
    with lock:
        with open(RESULTS_CSV, "a", newlines="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "filename", "label", "confidence",
                                                   "temperature", "humidity"])
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

def process_frame(jpeg_bytes: bytes):
    """사진 한 장 받았을 때 저장 -> 보드에 온습도 요쳥 -> csv기록 -> 보드에 결과 전송"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{ts}.jpg"
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(jpeg_bytes)

     # 사진이 도착한 이 시점에 Nucleo에 온습도를 요청 -> 같은 timestamp로 묶임
    temperature, humidity = Nucleo_bridge.read_temp_humidity()
 
    label, confidence = vision_predict(filepath)
 
    # Nucleo로 판정 결과 전송 -> LED/부저 구동
    Nucleo_bridge.send_result(label)
 
    row = {
        "timestamp": ts, "filename": filename, "label": label, "confidence": confidence,
        "temperature": temperature, "humidity": humidity,
    }
    append_result(row)
 
    with lock:
        latest.update(row)
 
    print(f"[{ts}] 판별 결과: {label} (신뢰도 {confidence:.2f}, 온도 {temperature}, 습도 {humidity}) -> {filename}")   

def serial_reader_loop():
    """cam 포트를 계속 읽으면서 프레임을 찾아 파싱"""
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"ESP32 시리얼 포트 열림: {SERIAL_PORT} @ {BAUD_RATE}")

    buf = bytearray()

    while True:
        chunk = ser.read(4096)
        if chunk:
            buf.extend(chunk)

        start_idx = buf.find(START_MARKER)
        if start_idx == -1:
            if len(buf) > 8192:
                buf = buf[-4:]
            continue

        if start_idx > 0:
            del buf[:start_idx]

        if len(buf) < 8:
            continue

        length = int.from_bytes(buf[4:8], byteorder="little")

        frame_total_len = 8 + length + 2 + 4
        if len(buf) < frame_total_len:
            continue

        jpeg_data = bytes(buf[8:8 + length])
        recv_checksum = int.from_bytes(buf[8+ length:8+ length + 2], byteorder="little")
        end_marker = bytes(buf[8 + length + 2:8 + length + 2 + 4])

        del buf[:frame_total_len]

        if end_marker != END_MARKER:
            print("경고 : 종료 마커 불일치, 프레임 버림 (데이터 밀림 가능성)")
            continue
        if compute_checksum(jpeg_data) != recv_checksum:
            print("경고: 체크섬 불일치, 손상된 프레임 버림")
            continue

        process_frame(jpeg_data)

#웹 뷰어 와이파이 접속
@app.route("/latest.jpg")
def latest_jpg():
    with lock:
        filename = latest["filename"]
    if not filename:
        return "", 404
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/latest_result")
def latest_result():
    with lock:
        return jsonify(dict(latest))
    
@app.route("/")
def viewer():
    html = """
    <html>
    <head>
      <meta charset="utf-8">
      <title>배터리 코팅 결함 검사</title>
      <script>
        function refresh() {
          fetch('/latest_result').then(r => r.json()).then(d => {
            document.getElementById('label').innerText = d.label || '대기 중...';
            document.getElementById('conf').innerText = d.confidence != null ? (d.confidence*100).toFixed(1) + '%' : '';
            document.getElementById('ts').innerText = d.timestamp || '';
            document.getElementById('temp').innerText = d.temperature != null ? d.temperature + ' °C' : '-';
            document.getElementById('hum').innerText = d.humidity != null ? d.humidity + ' %' : '-';
            if (d.filename) document.getElementById('img').src = '/latest.jpg?' + Date.now();
          });
        }
        setInterval(refresh, 2000);
        window.onload = refresh;
      </script>
    </head>
    <body style="font-family: sans-serif; text-align:center;">
      <h2>배터리 코팅 결함 실시간 검사 (시리얼 수신)</h2>
      <img id="img" src="/latest.jpg" style="max-width:480px; border:1px solid #ccc;">
      <h3 id="label">대기 중...</h3>
      <p>신뢰도: <span id="conf"></span></p>
      <p>온도: <span id="temp"></span> / 습도: <span id="hum"></span></p>
      <p>촬영 시각: <span id="ts"></span></p>
    </body>
    </html>
    """
    return Response(html, mimetype="text/html")    


if __name__ == "__main__":
    load_model()
    Nucleo_bridge.connect()

    t = threading.Thread(target=serial_reader_loop, daemon=True)
    t.start()

    app.run(host= "0.0.0.0", port=5000)