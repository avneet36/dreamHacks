#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Create the PWM driver object (default I2C address is 0x40)
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

// Define minimum and maximum pulse lengths for your servos
#define SERVOMIN  150  // Minimum pulse length count
#define SERVOMAX  600  // Maximum pulse length count

// Servo channels
#define SERVO_X 0  // Channel 0 (X movement)
#define SERVO_Y 1  // Channel 1 (Y movement)
#define SERVO_2 2  // Channel 2 (spray routine)
#define SERVO_3 3  // Channel 3 (spray routine)

// Function to map an angle (0-180) to a pulse length
uint16_t mapAngleToPulse(int angle) {
  return map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

// Variables to store last angles for X and Y (to reduce jitter)
int lastX = -1;
int lastY = -1;
const int angleThreshold = 2; // Only update if change > 2 degrees

// Limit update frequency for x/y commands
unsigned long lastUpdateTime = 0;
const unsigned long updateInterval = 50; // milliseconds

// Flag to indicate that the spray routine is running
bool inSprayMode = false;

// Function to move a given servo to the specified angle
void moveServo(uint8_t channel, int angle) {
  int pulse = mapAngleToPulse(angle);
  pwm.setPWM(channel, 0, pulse);
}

void setup() {
  Serial.begin(9600);
  pwm.begin();
  pwm.setPWMFreq(60);  // Set to 60 Hz for typical servos
  moveServo(SERVO_Y, 10);
  moveServo(SERVO_X, 50);
}

// Function that executes the spray routine (runs completely regardless of new commands)
void executeSprayRoutine() {
  inSprayMode = true;
  // Optionally clear any lingering serial input:
  while (Serial.available() > 0) { Serial.read(); }
  
  // Example loop: run the spray routine 10 times
  for (int i = 0; i < 10; i++) {
    moveServo(SERVO_2, 0);
    moveServo(SERVO_3, 180);
    delay(400);
    moveServo(SERVO_2, 180);
    moveServo(SERVO_3, 0);
    delay(250);
  }
  inSprayMode = false;
  // Indicate completion (this print is optional)
  Serial.println("Done");
}

void loop() {
  // Only process incoming commands if not currently in spray mode
  if (!inSprayMode && Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    // If command is "start", run the spray routine
    if (command == "start") {
      executeSprayRoutine();
    }
    else {
      // Otherwise, process x/y servo commands, expecting format "X<angle>:Y<angle>"
      int indexX = command.indexOf('X');
      int indexColon = command.indexOf(':');
      int indexY = command.indexOf('Y');

      if (indexX != -1 && indexColon != -1 && indexY != -1) {
        String xStr = command.substring(indexX + 1, indexColon);
        String yStr = command.substring(indexY + 1);

        int xAngle = xStr.toInt();
        int yAngle = yStr.toInt();

        unsigned long now = millis();
        if (now - lastUpdateTime >= updateInterval) {
          if (abs(xAngle - lastX) > angleThreshold || abs(yAngle - lastY) > angleThreshold) {
            moveServo(SERVO_X, xAngle);
            moveServo(SERVO_Y, yAngle);
            lastX = xAngle;
            lastY = yAngle;
          }
          lastUpdateTime = now;
        }
      }
    }
  }
}
