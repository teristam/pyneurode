/*
  Reading a serial ASCII-encoded string.

  This sketch demonstrates the Serial parseInt() function.
  It looks for an ASCII string of comma-separated values.
  It parses them into ints, and uses those to fade an RGB LED.

  Circuit: Common-Cathode RGB LED wired like so:
  - red anode: digital pin 3
  - green anode: digital pin 5
  - blue anode: digital pin 6
  - cathode: GND

  created 13 Apr 2012
  by Tom Igoe
  modified 14 Mar 2016
  by Arturo Guadalupi

  This example code is in the public domain.
*/



// pins for the LEDs:
bool portNeedTrigger = false;

void setup() {
  // initialize serial:
  Serial.begin(115200);
  Serial.print("Please send a port to trigger!\n");
  Serial.setTimeout(1);

  //set all pin to output
  for (int i=0;i<=13;i++){
    pinMode(i,OUTPUT);
  }
  // make the pins outputs:
  // pinMode(redPin, OUTPUT);
  // pinMode(greenPin, OUTPUT);
  // pinMode(bluePin, OUTPUT);

}

void loop() {
  // if there's any serial available, read it:
  while (Serial.available() > 0) {

      //keep reading the serial 
      int port = Serial.parseInt(); 
      if (port>0){
          digitalWrite(port, HIGH);
          portNeedTrigger = true;
      }

    }

    if (portNeedTrigger){
       delay(1);
      //reset all the pins
      for (int i=0;i<=13;i++){
        digitalWrite(i, LOW);
      }
      portNeedTrigger=false;
      Serial.println("done");

    }


}
