
#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiAP.h>
#include <WiFiUdp.h>
#include <Wire.h>

//#define LED_BUILTIN 2   // Set the GPIO pin where you connected your test LED or comment this line out if your dev board has a built-in LED

// Set these to your desired credentials.
const char *ssid = "ESP_AP";
const char *password = "somepass";


#define I2C_SDA 21
#define I2C_SCL 22

//IP address to send UDP data to:
// either use the ip address of the server or 
// a network broadcast address
const char * udpAddress = "192.168.4.2";
const int udpPort = 13333;

//Are we currently connected?
boolean connected = false;

//The udp library class
WiFiUDP udp;

// BMX055 Accl I2C address is 0x18(24)
#define Addr_Accl 0x18
// BMX055 Gyro I2C address is 0x68(104)
#define Addr_Gyro 0x68

// Stop button is attached to PIN 0 (IO0)
#define BTN_STOP_ALARM    0

hw_timer_t * timer = NULL;
portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;

volatile uint32_t isrCounter = 0;
volatile uint32_t lastIsrAt = 0;

void IRAM_ATTR onTimer() {
  portENTER_CRITICAL_ISR(&timerMux);
  isrCounter++;
  portEXIT_CRITICAL_ISR(&timerMux); 
}

void setupBMX055(int num){

  digitalWrite(num, HIGH);
  
  // Initialise I2C communication as MASTER
  Wire.begin(I2C_SDA, I2C_SCL, (uint32_t)400000);

  // Start I2C Transmission
  Wire.beginTransmission(Addr_Accl);
  // Select PMU_Range register
  Wire.write(0x0F);
  // Range = +/- 8g
  Wire.write(0x08);
  // Stop I2C Transmission
  Wire.endTransmission();

  // Start I2C Transmission
  Wire.beginTransmission(Addr_Accl);
  // Select PMU_BW register
  Wire.write(0x10);
  // Bandwidth = 500 Hz  ****
  Wire.write(0x0E);
  // Stop I2C Transmission
  Wire.endTransmission();

  // Start I2C Transmission
  Wire.beginTransmission(Addr_Accl);
  // Select PMU_LPW register
  Wire.write(0x11);
  // Normal mode, Sleep duration = 0.5ms
  Wire.write(0x00);
  // Stop I2C Transmission on the device
  Wire.endTransmission();

  // Start I2C Transmission
  Wire.beginTransmission(Addr_Gyro);
  // Select Range register
  Wire.write(0x0F);
  // Full scale = +/- 1000 degree/s
  Wire.write(0x01);
  // Stop I2C Transmission
  Wire.endTransmission();

  // Start I2C Transmission
  Wire.beginTransmission(Addr_Gyro);
  // Select Bandwidth register
  Wire.write(0x10);
  // ODR = 100 Hz ****
  Wire.write(0x02);
  // Stop I2C Transmission
  Wire.endTransmission();

  // Start I2C Transmission
  Wire.beginTransmission(Addr_Gyro);
  // Select LPM1 register
  Wire.write(0x11);
  // Normal mode, Sleep duration = 2ms
  Wire.write(0x00);
  // Stop I2C Transmission
  Wire.endTransmission();

  delay(300);
  digitalWrite(num, LOW);
}


void getSensorData(byte* data)
{
  Wire.beginTransmission(Addr_Accl);
  Wire.write(0x02);
  Wire.endTransmission(0);

  // Request 1 byte of data
  Wire.requestFrom(Addr_Accl, 6);
  
  for (int i = 0; i < 6; i++)
  {
      *(data+i) = Wire.read();   
  }
    
  // Start I2C Transmission
  Wire.beginTransmission(Addr_Gyro);
  // Select data register
  Wire.write(0x02);
    // Stop I2C Transmission
  Wire.endTransmission(0);
  
  // Request 1 byte of data
  Wire.requestFrom(Addr_Gyro, 6);
  for (int i = 6; i < 12; i++)
  {
      *(data+i) = Wire.read();
  }
}

volatile uint32_t tachoCounter = 0;
void IRAM_ATTR tachoMagnetRise() {
    tachoCounter++;
}

void setup() {
    
  timer = timerBegin(0, 80, true);
  timerAttachInterrupt(timer, &onTimer, true);
  timerAlarmWrite(timer, 100, true);
  timerAlarmEnable(timer);

  pinMode(12, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(12), tachoMagnetRise, RISING);

  pinMode(2, OUTPUT);
  pinMode(4, OUTPUT);
  pinMode(5, OUTPUT);
  digitalWrite(2, LOW);
  digitalWrite(4, LOW);
  digitalWrite(5, LOW);
  setupBMX055(2);
  setupBMX055(4);
  setupBMX055(5);

  // You can remove the password parameter if you want the AP to be open.
  WiFi.softAP(ssid, password);
  IPAddress myIP = WiFi.softAPIP();

//  Serial.begin(115200);
//  Serial.println(myIP);  
 
  udp.begin(udpPort);
  connected = true;
}

void loop() {
  byte data[44];

  data[0] = (isrCounter >> 24) & 0xFF;
  data[1] = (isrCounter >> 16) & 0xFF;  
  data[2] = (isrCounter >> 8)  & 0xFF;
  data[3] = (isrCounter >> 0)  & 0xFF;

  data[4] = (tachoCounter >> 24) & 0xFF;
  data[5] = (tachoCounter >> 16) & 0xFF;  
  data[6] = (tachoCounter >> 8)  & 0xFF;
  data[7] = (tachoCounter >> 0)  & 0xFF;

  digitalWrite(5, HIGH);
  getSensorData(&data[8]);
  digitalWrite(5, LOW);
  digitalWrite(2, HIGH);
  getSensorData(&data[20]);
  digitalWrite(2, LOW);
  digitalWrite(4, HIGH);
  getSensorData(&data[32]);
  digitalWrite(4, LOW);
  udp.beginPacket(udpAddress, udpPort);
  udp.write(data, 44);
  //udp.printf("Seconds since boot: %d", isrCounter);
  udp.endPacket();

  //Serial.println(i);
}
  
//wifi event handler
void WiFiEvent(WiFiEvent_t event){
    switch(event) {
      case SYSTEM_EVENT_STA_GOT_IP:
          //When connected set 
          Serial.print("WiFi connected! IP address: ");
          Serial.println(WiFi.localIP());  
          //initializes the UDP state
          //This initializes the transfer buffer
          udp.begin(WiFi.localIP(),udpPort);
          connected = true;
          break;
      case SYSTEM_EVENT_STA_DISCONNECTED:
          Serial.println("WiFi lost connection");
          connected = false;
          break;
      default: break;
    }
}
