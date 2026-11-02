// UNO + DRI0043, common-anode inputs. D2=PUL-, D3=DIR-.
#include <Arduino.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

const byte PULSE_PIN = 2, DIR_PIN = 3;
long positionSteps = 0, targetSteps = 0;
long lowerLimit = -100, upperLimit = 100;
float maxSpeed = 50, speedNow = 10;
const float ACCEL = 100; // steps/second squared
bool moving = false, referenced = false;
unsigned long lastContact = 0, nextStep = 0;
char line[80];
byte used = 0;
bool overflowed = false;

void status() {
  Serial.print(F("STATE ")); Serial.print(positionSteps);
  Serial.print(' '); Serial.print(targetSteps);
  Serial.print(' '); Serial.print(moving ? 1 : 0);
  Serial.print(' '); Serial.print(referenced ? 1 : 0);
  Serial.print(' '); Serial.print(lowerLimit);
  Serial.print(' '); Serial.print(upperLimit);
  Serial.print(' '); Serial.println((long)maxSpeed);
}

void stopMotion(const __FlashStringHelper *reason) {
  moving = false;
  targetSteps = positionSteps;
  Serial.println(reason);
  status();
}

bool number(const char *s, long &value) {
  if (!s || !*s) return false;
  char *end;
  errno = 0;
  value = strtol(s, &end, 10);
  return !errno && !*end && value >= -1000000L && value <= 1000000L;
}

void command(char *input) {
  char *op = strtok(input, " \t");
  if (!op) return;
  char *a = strtok(NULL, " \t");
  char *b = strtok(NULL, " \t");
  char *extra = strtok(NULL, " \t");
  long x = 0, y = 0;
  if (!strcmp(op, "STATUS") && !a) {
    lastContact = millis(); status(); return;
  }
  if (!strcmp(op, "STOP") && !a) {
    lastContact = millis(); stopMotion(F("STOPPED")); return;
  }
  if (!strcmp(op, "ZERO") && !a) {
    if (moving) { Serial.println(F("ERR BUSY")); return; }
    positionSteps = targetSteps = 0; referenced = true;
    lastContact = millis(); Serial.println(F("OK ZERO")); status(); return;
  }
  if (extra || !number(a, x)) { Serial.println(F("ERR COMMAND")); return; }
  if (moving) { Serial.println(F("ERR BUSY")); return; }
  if (!strcmp(op, "LIMIT")) {
    if (!number(b, y) || x >= y || x > 0 || y < 0 ||
        positionSteps < x || positionSteps > y) {
      Serial.println(F("ERR LIMIT")); return;
    }
    lowerLimit = x; upperLimit = y;
  } else if (b) {
    Serial.println(F("ERR COMMAND")); return;
  } else if (!strcmp(op, "SPEED")) {
    if (x < 10 || x > 200) { Serial.println(F("ERR SPEED")); return; }
    maxSpeed = x;
  } else if (!strcmp(op, "MOVE") || !strcmp(op, "GOTO")) {
    if (!referenced) { Serial.println(F("ERR ZERO_REQUIRED")); return; }
    long destination = !strcmp(op, "MOVE") ? positionSteps + x : x;
    if (destination < lowerLimit || destination > upperLimit) {
      Serial.println(F("ERR OUT_OF_RANGE")); return;
    }
    targetSteps = destination;
    moving = positionSteps != targetSteps;
    speedNow = 10;
    digitalWrite(DIR_PIN, targetSteps > positionSteps ? HIGH : LOW);
    nextStep = micros() + 10000UL; // direction setup time
    if (!moving) Serial.println(F("DONE"));
  } else {
    Serial.println(F("ERR COMMAND")); return;
  }
  lastContact = millis(); Serial.println(F("OK")); status();
}

void setup() {
  digitalWrite(PULSE_PIN, HIGH); pinMode(PULSE_PIN, OUTPUT);
  digitalWrite(DIR_PIN, LOW); pinMode(DIR_PIN, OUTPUT);
  Serial.begin(115200);
  lastContact = millis();
  Serial.println(F("READY MIRROR_CONTROL 1")); status();
}

void loop() {
  // Bound serial work so a stream of bytes cannot starve motion service.
  for (byte count = 0; count < 32 && Serial.available(); ++count) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      if (overflowed) Serial.println(F("ERR LINE_TOO_LONG"));
      else { line[used] = 0; command(line); }
      used = 0; overflowed = false;
    } else if (used < sizeof(line) - 1 && !overflowed) line[used++] = c;
    else overflowed = true;
  }
  if (!moving) return;
  if ((unsigned long)(millis() - lastContact) > 1500UL) {
    stopMotion(F("STOPPED TIMEOUT")); return;
  }
  if ((long)(micros() - nextStep) < 0) return;
  digitalWrite(PULSE_PIN, LOW);
  delayMicroseconds(20);
  digitalWrite(PULSE_PIN, HIGH);
  positionSteps += targetSteps > positionSteps ? 1 : -1;
  if (positionSteps == targetSteps) {
    moving = false; Serial.println(F("DONE")); status(); return;
  }
  long remaining = labs(targetSteps - positionSteps);
  float brakingSpeed = sqrtf(2.0f * ACCEL * remaining);
  float accelerated = sqrtf(speedNow * speedNow + 2.0f * ACCEL);
  speedNow = min(maxSpeed, min(brakingSpeed, accelerated));
  nextStep = micros() + (unsigned long)(1000000.0f / speedNow);
}
