from lib.modules.VeinAuth import *

url = 'http://172.16.44.102/camera/cam_pic.php'
fileName = 'settings.json'

va = VeinAuth(url, fileName)
va.Verify()