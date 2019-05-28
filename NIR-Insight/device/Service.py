import RPi.GPIO as GPIO
import time
from .VeinAuthDevice import *

url = 'http://local/camera/cam_pic.php'
fileName = 'settings.json'

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(12, GPIO.OUT)
GPIO.setup(4, GPIO.OUT)
GPIO.setup(26, GPIO.IN, pull_up_down=GPIO.PUD_UP)

isPressed = False
while True:
    input_state = GPIO.input(26)
    if input_state == False:
        isPressed = True
        GPIO.output(12, GPIO.HIGH)
        now = time.time()
        minRegTime = now + 2
    else: 
        if isPressed == True:
            isPressed = False
            GPIO.output(12, GPIO.LOW)
            if time.time() < minRegTime:
                va = VeinAuth(url, fileName)
                va.Verify()
            else:
                va = VeinAuth(url, fileName)
                va.Register()

        #GPIO.output(4, GPIO.HIGH)
        #time.sleep(1)


