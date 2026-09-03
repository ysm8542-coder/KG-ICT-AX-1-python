#include "esp_camera.h"

// ===== 카메라 핀 설정 (AI-Thinker ESP32-CAM 기준) =====
struct CameraPins {
  int pwdn;
  int reset;
  int xclk;
  int siod;
  int sioc;
  int d0, d1, d2, d3, d4, d5, d6, d7; // Y2 ~ Y9 순서
  int vsync;
  int href;
  int pclk;
};

const CameraPins CAM_PINS = {
  .pwdn  = 32,
  .reset = -1,
  .xclk  = 0,
  .siod  = 26,
  .sioc  = 27,
  .d0 = 5,   // Y2
  .d1 = 18,  // Y3
  .d2 = 19,  // Y4
  .d3 = 21,  // Y5
  .d4 = 36,  // Y6
  .d5 = 39,  // Y7
  .d6 = 34,  // Y8
  .d7 = 35,  // Y9
  .vsync = 25,
  .href  = 23,
  .pclk  = 22
};

// ===== 시리얼 전송 프로토콜 =====
// [시작마커 4바이트][길이 4바이트][JPEG 데이터][체크섬 2바이트][종료마커 4바이트]
const uint8_t START_MARKER[4] = {0xAA, 0x55, 0xAA, 0x55};
const uint8_t END_MARKER[4]   = {0x55, 0xAA, 0x55, 0xAA};

const unsigned long CAPTURE_INTERVAL_MS = 5000; // 5초마다 촬영
unsigned long lastCaptureTime = 0;
unsigned long photoIndex = 0;

void startCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = CAM_PINS.d0;
  config.pin_d1 = CAM_PINS.d1;
  config.pin_d2 = CAM_PINS.d2;
  config.pin_d3 = CAM_PINS.d3;
  config.pin_d4 = CAM_PINS.d4;
  config.pin_d5 = CAM_PINS.d5;
  config.pin_d6 = CAM_PINS.d6;
  config.pin_d7 = CAM_PINS.d7;
  config.pin_xclk  = CAM_PINS.xclk;
  config.pin_pclk  = CAM_PINS.pclk;
  config.pin_vsync = CAM_PINS.vsync;
  config.pin_href  = CAM_PINS.href;
  config.pin_sccb_sda = CAM_PINS.siod;
  config.pin_sccb_scl = CAM_PINS.sioc;
  config.pin_pwdn  = CAM_PINS.pwdn;
  config.pin_reset = CAM_PINS.reset;
  config.xclk_freq_hz = 20000000;
  config.pixel_format  = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size   = FRAMESIZE_VGA;   // 640x480
    config.jpeg_quality  = 10;
    config.fb_count      = 2;
  } else {
    config.frame_size   = FRAMESIZE_SVGA;
    config.jpeg_quality  = 12;
    config.fb_count      = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    pinMode(4, OUTPUT);
    while (true) {
      digitalWrite(4, HIGH); delay(100);
      digitalWrite(4, LOW);  delay(100);
    }
  }
}

uint16_t computeChecksum(const uint8_t* data, size_t len) {
  uint16_t sum = 0;
  for (size_t i = 0; i < len; i++) {
    sum = (sum + data[i]) & 0xFFFF;
  }
  return sum;
}

void sendFrameOverSerial(const uint8_t* data, size_t len) {
  uint32_t length32 = (uint32_t)len;
  uint16_t checksum = computeChecksum(data, len);

  Serial.write(START_MARKER, 4);
  Serial.write((uint8_t*)&length32, 4);
  Serial.write(data, len);
  Serial.write((uint8_t*)&checksum, 2);
  Serial.write(END_MARKER, 4);
  Serial.flush();
}

void setup() {
  Serial.begin(921600);
  delay(200);
  startCamera();
}

void loop() {
  unsigned long now = millis();
  if (now - lastCaptureTime >= CAPTURE_INTERVAL_MS) {
    lastCaptureTime = now;

    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) {
      sendFrameOverSerial(fb->buf, fb->len);
      esp_camera_fb_return(fb);
      photoIndex++;
    }
  }
}
