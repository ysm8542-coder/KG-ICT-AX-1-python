# KG-ICT-AX-1-python
**AI-Powered Defect Detection**: Utilizes a custom AI model to analyze electrode coating quality.
* **Multi-Modal Data Integration**: Combines visual data captured via camera with real-time industrial sensor values.
* **Real-Time Classification**: Automatically distinguishes between normal and defective electrodes.
* **Streamlit Dashboard**: Provides an intuitive, interactive web interface for monitoring inspection results.
* **Automated Reporting**: Generates detailed status reports based on AI evaluation outcomes.

File Descriptions
main.c

Firmware for the STM32 Nucleo board. Reads temperature and humidity from a DHT-series sensor over a bit-banged single-wire protocol (custom timing via TIM3), and listens for single-character commands over UART2 (R = report temp/humidity, N/D = drive the green/red LED to indicate a normal/defective result, S = handshake). Acts purely as a sensor + indicator peripheral; it has no knowledge of the defect-detection logic itself.

Nucleo_bridge.py

Python-side serial bridge to the Nucleo board. Opens the COM port, sends R to request a temperature/humidity reading and parses the "T:..,H:.." response, sends N/D to trigger the LEDs after a defect judgment, and handles clean connection/disconnection.

mask_utils.py

Classical (non-deep-learning) OpenCV pipeline that turns a raw RGB surface image into a color-coded defect mask. Uses top-hat/black-hat morphology with dynamically computed thresholds to isolate bright and dark anomalies, then classifies each connected component as a crack (blue, thin/elongated/low-circularity), peeling (green, large bright or saturated regions), or pinhole (red, small round bright spots) based on shape, thickness, and adjacency heuristics refined through iterative empirical tuning.

feature_extraction.py

Converts the color-coded defect mask from mask_utils.py into a compact, interpretable 15-value numeric feature vector (5 features — area ratio, pixel count, component count, max/mean component area — for each of the 3 defect classes). Designed so a lightweight classifier can be trained on limited data and so SHAP explanations map cleanly back to specific defect types.

dataset_utils.py

Loads a labeled CSV + image folder into training-ready arrays. Parses defect labels (English or Korean) into multi-hot vectors, runs each image through the mask-generation and feature-extraction pipeline, and assembles the final (X, Y) dataset. Also provides label-string conversion utilities used at inference time.

model_utils.py

Defines and trains the multi-label defect classifier: a small dense (fully-connected) Keras network that takes the 15-dimensional feature vector and outputs independent sigmoid probabilities for pinhole/peeling/crack, trained with binary cross-entropy and early stopping.

train.py

End-to-end training script: loads the dataset, splits train/validation, standard-scales the features, computes per-class weights to counter label imbalance, trains the model, and saves the model, scaler, and metadata to disk.

evaluate_model.py

Loads a trained model and scaler, runs predictions on a held-out labeled set, and exports evaluation artifacts: a per-class precision/recall/F1/ROC-AUC CSV, a bar-chart comparison image, and a multi-label confusion matrix CSV.

predict_single_image.py

Inference entry point for a single image: builds the defect mask, extracts features, scales them, runs the trained model, and returns the predicted label, normal/defect flag, and per-class probabilities. Caches loaded models/scalers to avoid reloading on repeated calls.

Server_update.py

Flask server that ties the whole system together for live operation. Receives uploaded images, requests a matching temperature/humidity reading from the Nucleo board, runs inference (with temperature/humidity-aware threshold adjustments computed fresh per request, never mutating global state), drives the board's LEDs with the result, saves a mask image and a results-log row, and serves a live HTML dashboard plus session statistics/report generation on shutdown.

dashboard.py

Streamlit dashboard for offline/historical analysis of logged results. Reads the CSV log(s) produced by Server_update.py and visualizes inspection counts, pass/fail ratio, defect-type frequency, temperature/humidity trends and their correlation with defect rate, throughput, and prediction confidence distribution (including a low-confidence review list).

esp32cam_client.ino — Wi-Fi HTTP Upload

Firmware for an AI-Thinker ESP32-CAM module that captures photos and uploads them to a Flask server over Wi-Fi.

What it does:

Initializes the OV2640 camera (JPEG output, VGA/SVGA depending on PSRAM availability)
Connects to a Wi-Fi network (WIFI_STA mode, with auto-reboot on failure)
Every ~3 seconds (after a countdown), captures a JPEG frame and sends it via HTTP POST to a Flask endpoint (http://<PC_IP>:5000/upload)
Each request includes an X-Photo-Index header for sequencing, and Content-Type: image/jpeg
Automatically attempts Wi-Fi reconnection if the connection drops

Requirements: ESP32-CAM board must be on the same Wi-Fi network/subnet as the PC running the Flask server. Update ssid, password, and serverUrl (server's local IP) before flashing.

Use case: Best when the ESP32-CAM has reliable Wi-Fi access and you want a simple network-based image pipeline.

esp32cam_serial.ino — USB Serial Streaming

Firmware for the same ESP32-CAM hardware that streams photos to a host PC over a USB/UART serial connection instead of Wi-Fi — useful when no Wi-Fi network is available or a wired connection is preferred.

What it does:

Initializes the camera identically to the Wi-Fi version
Captures a JPEG frame every 5 seconds (CAPTURE_INTERVAL_MS)
Sends each frame over serial (baud rate 921600) using a custom binary framing protocol:
  [4-byte start marker: AA 55 AA 55]
  [4-byte length (uint32, frame size)]
  [JPEG image data]
  [2-byte checksum (sum of all bytes, mod 0xFFFF)]
  [4-byte end marker: 55 AA 55 AA]
On camera init failure, blinks an onboard LED (pin 4) as an error indicator instead of logging to Serial (since Serial is used for data, not debug text)

Python-side counterpart: Your Python script should open the serial port at 921600 baud, scan for the start marker, read the length field, read exactly that many bytes as JPEG data, then validate the checksum and end marker before saving/displaying the frame.

Use case: Best when a direct USB cable connection to the host PC is available and you want to avoid Wi-Fi networking/latency.
