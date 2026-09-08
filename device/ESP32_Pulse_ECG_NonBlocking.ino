#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include "MAX30105.h"

// =====================================================
// CONFIG
// =====================================================
const char* AP_SSID     = "ESP32_Health";
const char* AP_PASSWORD = "health1234";   // must be 8+ chars

WebServer server(80);
MAX30105 particleSensor;

// ---------- MAX30102 (pulse) ----------
// I2C pins: uses Wire.begin() defaults for your board.
// On most ESP32-S3 DevKitC-1 boards that's SDA=GPIO8, SCL=GPIO9.
#define PULSE_BUFFER_SIZE 500
long pulseBuffer[PULSE_BUFFER_SIZE];
uint32_t pulseIndex = 0;               // total pulse samples taken since boot

unsigned long lastPulseSampleTime = 0;
const unsigned long pulseIntervalMs = 10;   // ~100 Hz

// ---------- AD8232 (ECG) ----------
const int ECG_PIN = 4;                 // AD8232 OUTPUT -> GPIO4 (ADC1 channel)
#define ECG_BUFFER_SIZE 2500           // 10 sec of buffer at 250Hz
uint16_t ecgBuffer[ECG_BUFFER_SIZE];
uint32_t ecgIndex = 0;                 // total ECG samples taken since boot

uint32_t nextEcgSampleTime = 0;        // micros()-based timer, non-blocking
const uint32_t ECG_SAMPLE_RATE = 250;  // Hz
const uint32_t ECG_INTERVAL_US = 1000000UL / ECG_SAMPLE_RATE;

// =====================================================
// Generic "give me everything since index N" JSON builder
// =====================================================
String buildSamplesJson(uint32_t latestIndex, uint32_t since,
                         uint32_t bufferSize,
                         long (*getLong)(uint32_t),
                         bool useLong) {
  uint32_t start = since;
  if (latestIndex > bufferSize && since < latestIndex - bufferSize) {
    start = latestIndex - bufferSize;   // client fell too far behind
  }

  String json = "{\"latestIndex\":" + String(latestIndex) + ",\"samples\":[";
  bool first = true;
  for (uint32_t i = start; i < latestIndex; i++) {
    if (!first) json += ",";
    json += String(getLong(i));
    first = false;
  }
  json += "]}";
  return json;
}

long getPulseSample(uint32_t i) { return pulseBuffer[i % PULSE_BUFFER_SIZE]; }
long getEcgSample(uint32_t i)   { return (long) ecgBuffer[i % ECG_BUFFER_SIZE]; }

// =====================================================
// /data  -> pulse JSON, poll with ?since=N
// =====================================================
void handleData() {
  String sinceParam = server.hasArg("since") ? server.arg("since") : "0";
  uint32_t since = (uint32_t) sinceParam.toInt();
  String json = buildSamplesJson(pulseIndex, since, PULSE_BUFFER_SIZE, getPulseSample, true);
  server.send(200, "application/json", json);
}

// =====================================================
// /ecg  -> ECG JSON, poll with ?since=N (same style, no blocking)
// =====================================================
void handleEcg() {
  String sinceParam = server.hasArg("since") ? server.arg("since") : "0";
  uint32_t since = (uint32_t) sinceParam.toInt();
  String json = buildSamplesJson(ecgIndex, since, ECG_BUFFER_SIZE, getEcgSample, false);
  server.send(200, "application/json", json);
}

// =====================================================
// /  -> info page
// =====================================================
void handleRoot() {
  String html =
    "<html><head><title>ESP32-S3 Pulse + ECG (non-blocking)</title></head><body>"
    "<h1>ESP32-S3 Pulse + ECG Server</h1>"
    "<p>Pulse (100Hz): <a href='/data'>/data?since=N</a></p>"
    "<p>ECG (250Hz): <a href='/ecg'>/ecg?since=N</a></p>"
    "<p>Both endpoints can be polled at the same time — sampling for both "
    "sensors runs independently in the background.</p>"
    "</body></html>";
  server.send(200, "text/html", html);
}

void setup() {
  Serial.begin(115200);
  delay(200);

  // ---- MAX30102 init ----
  Wire.begin();  // if sensor isn't found, try Wire.begin(SDA_PIN, SCL_PIN)
  Serial.println("Initializing MAX30102...");
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 was not found. Check wiring/power.");
    while (1) { delay(1000); }
  }
  particleSensor.setup();
  particleSensor.setPulseAmplitudeRed(0x0A);
  particleSensor.setPulseAmplitudeGreen(0);

  // ---- AD8232 ADC init ----
  analogReadResolution(12);
  analogSetPinAttenuation(ECG_PIN, ADC_11db);
  pinMode(ECG_PIN, INPUT);

  // ---- WiFi AP ----
  Serial.println("Starting Access Point...");
  WiFi.mode(WIFI_AP);
  if (!WiFi.softAP(AP_SSID, AP_PASSWORD, 6, false, 1)) {
    Serial.println("ERROR: Failed to start Wi-Fi AP");
    while (true) { delay(1000); }
  }
  IPAddress apIP = WiFi.softAPIP();
  Serial.print("AP started. SSID: ");
  Serial.println(AP_SSID);
  Serial.print("IP: ");
  Serial.println(apIP);
  Serial.print("Pulse: http://"); Serial.print(apIP); Serial.println("/data");
  Serial.print("ECG:   http://"); Serial.print(apIP); Serial.println("/ecg");

  server.on("/", handleRoot);
  server.on("/data", handleData);
  server.on("/ecg", handleEcg);
  server.begin();
  Serial.println("HTTP server started on port 80.");

  nextEcgSampleTime = micros();
}

void loop() {
  server.handleClient();

  // ---- Pulse sampling: non-blocking, millis()-based ----
  unsigned long nowMs = millis();
  if (nowMs - lastPulseSampleTime >= pulseIntervalMs) {
    lastPulseSampleTime = nowMs;
    long irValue = particleSensor.getIR();
    pulseBuffer[pulseIndex % PULSE_BUFFER_SIZE] = irValue;
    pulseIndex++;
  }

  // ---- ECG sampling: non-blocking, micros()-based ----
  uint32_t nowUs = micros();
  if ((int32_t)(nowUs - nextEcgSampleTime) >= 0) {
    nextEcgSampleTime += ECG_INTERVAL_US;
    uint16_t ecgValue = analogRead(ECG_PIN);
    ecgBuffer[ecgIndex % ECG_BUFFER_SIZE] = ecgValue;
    ecgIndex++;
  }

  // No delay() here — keep the loop free-running so both sensors and
  // the web server all get serviced as fast as possible.
}
