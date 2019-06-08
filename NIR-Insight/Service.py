import RPi.GPIO as GPIO
import time
from lib.device.VeinAuthDevice import *

url = 'http://localhost/camera/cam_pic.php'
fileName = 'settings.json'

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(12, GPIO.OUT)
GPIO.setup(4, GPIO.OUT)
GPIO.setup(23, GPIO.OUT)
GPIO.setup(24, GPIO.OUT)
GPIO.setup(26, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print("Ready")
isPressed = False
while True:
    input_state = GPIO.input(26)
    if input_state == False and isPressed == False:
        isPressed = True
        GPIO.output(4, GPIO.HIGH)
        now = time.time()
        minRegTime = now + 2
    if input_state == True and isPressed == True: 
        if isPressed == True:
            isPressed = False
            GPIO.output(4, GPIO.LOW)
            if time.time() < minRegTime:
                va = VeinAuth(url, fileName)
                print('\nVerifying')
                va.Verify()
            else:
                va = VeinAuth(url, fileName)
                print('\nRegistering')
                va.Register()

        #GPIO.output(4, GPIO.HIGH)
        #time.sleep(1)



