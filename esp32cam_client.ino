#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_wifi.h"


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

// AI-Thinker ESP32-CAM 핀맵. 다른 보드로 바꾸면 이 값들만 교체하면 됨.
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

// ===== WiFi 설정 =====
const char* ssid     = "ORD_AX_Campus_B2";
const char* password = "ordaxcampusb2!";

// ===== PC(Flask 서버)의 로컬 IP. 반드시 같은 와이파이 대역이어야 함. =====
const char* serverUrl = "http://172.30.1.91:5000/upload";

// ===== 촬영 주기 =====
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
    Serial.printf("카메라 초기화 실패: 0x%x\n", err);
    while (true) delay(1000);
  }
}

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("WiFi 연결 중");
  int tryCount = 0;
  while (WiFi.status() != WL_CONNECTED && tryCount < 40) {
    delay(500);
    Serial.print(".");
    tryCount++;
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("WiFi 연결됨, IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("WiFi 연결 실패, 재부팅합니다");
    ESP.restart();
  }
}

void sendPhoto() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("프레임 캡처 실패");
    return;
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi 끊김, 전송 건너뛰고 재연결 시도");
    esp_camera_fb_return(fb);
    connectWiFi();
    return;
  }

  HTTPClient http;
  http.begin(serverUrl);
  http.setTimeout(20000);
  http.setConnectTimeout(8000);
  http.addHeader("Content-Type", "image/jpeg");
  http.addHeader("X-Photo-Index", String(photoIndex));

  int code = http.POST(fb->buf, fb->len);
  if (code > 0) {
    String response = http.getString();
    Serial.printf("[%lu] 전송 성공 (%d): %s\n", photoIndex, code, response.c_str());
  } else {
    Serial.printf("[%lu] 전송 실패: %s\n", photoIndex, http.errorToString(code).c_str());
  }
  http.end();

  esp_camera_fb_return(fb);
  photoIndex++;
}

void setup() {
  Serial.begin(115200);
  startCamera();
  connectWiFi();
}

void countdownBeforePhoto() {
  for (int i = 3; i >= 1; i--) {
    Serial.printf("사진찍기 전 -%d\n", i);
    delay(1000);
  }
}

void loop() {
  countdownBeforePhoto();
  sendPhoto();
}
