import threading
import serial

NUCLEO_PORT = "COM9"
NUCLEO_BAUD = 115200

_ser = None
_lock = threading.Lock()

def connect():
    global _ser
    _ser = serial.Serial(NUCLEO_PORT, NUCLEO_BAUD, timeout=2)
    print(f"Nucle 연결됨 : {NUCLEO_PORT} @ {NUCLEO_BAUD}")

def read_temp_humidity():
    """Nucleo 에 온습도를 요청하고 온도, 습도를 반환"""
    if _ser is None:
        return None, None

    with _lock:
        _ser.reset_input_buffer()
        _ser.write(b"R\n")
        line = _ser.readline().decode(errors="ignore").strip()

    # 기대 형식: "T:24.5,H:55.0"  (DHT 읽기 실패 시 Nucleo가 "T:ERR,H:ERR" 응답)
    try:
        parts = dict(p.split(":") for p in line.split(","))
        return float(parts["T"]), float(parts["H"])
    except Exception:
        if line == "":
            print("Nucleo 온습도 응답 파싱 실패: 응답이 없음(타임아웃) -> UART/NVIC/포트 설정 확인 필요")
        else:
            print(f"Nucleo 온습도 응답 파싱 실패: 받은 값 = '{line}' -> DHT 센서 배선/타이밍 확인 필요")
        return None, None

def send_result(label: str):
    """판별 결과를 Nucleo로 보내 LED 구동 (응답 대기 없음)"""
    if _ser is None:
        return
    code = b"N\n" if label.startswith("정상") else b"D\n"
    with _lock:
        _ser.write(code)

def close():
    """프로그램 종료 시 시리얼 포트를 정리"""
    global _ser
    if _ser is not None:
        try:
            _ser.close()
            print("Nucleo 시러얼 포트 닺음")
        except Exception:
            pass
        _ser = None