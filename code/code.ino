#define USE_ARDUINO_INTERRUPTS true
#include <PulseSensorPlayground.h>
#include<LiquidCrystal.h>
LiquidCrystal lcd(7, 6, 5, 4, 3, 2);
 
// Variables
const int PulseWire = 0;
const int LED13 = 13;
int Threshold = 550;
 
PulseSensorPlayground pulseSensor;
void setup() {
 
Serial.begin(9600); // For Serial Monitor
lcd.begin(20,4);
 

pulseSensor.analogInput(PulseWire);
pulseSensor.blinkOnPulse(LED13); 
pulseSensor.setThreshold(Threshold);

if (pulseSensor.begin()) {
Serial.println("heart rate monitoring system");
lcd.setCursor(0,0);
lcd.print(" Heart Rate");
lcd.setCursor(0,1);
lcd.print("  Monitor");
 
}
}
 
void loop() {
 
int myBPM = pulseSensor.getBeatsPerMinute();
if (pulseSensor.sawStartOfBeat()) { 
Serial.print("BPM: "); 
Serial.println(myBPM);
lcd.setCursor(0,2);
lcd.print("HeartBeat!");
lcd.setCursor(0,3);
lcd.print("BPM: ");
lcd.print(myBPM);
}
delay(20);
}
