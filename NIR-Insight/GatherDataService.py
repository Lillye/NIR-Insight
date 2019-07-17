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

while True:
    input_state = GPIO.input(26)
    if input_state == False:
        tmp = []
        va = VeinAuth(url, fileName)
        va.CaptureAndProcessImage(tmp,0)



