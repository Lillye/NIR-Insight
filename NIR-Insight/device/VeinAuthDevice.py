import random
import fastpbkdf2 
from fuzzy_extractor import FuzzyExtractor
import math
import operator
import string
import cv2 as cv
from lib.components.Pipeline import Pipeline
from lib.modules.FuzzyGate import FuzzyGate
from lib.modules.ProcessingLine import *
from lib.modules.Stages import *
from lib.modules.Services import *
from lib.modules.VeinAuth import *
import time
import json
import RPi.GPIO as GPIO
import hashlib
import lib.device.CryptoIO as CryptoIO 
import lib.device.AES as AES

salt = "demoSalt"
key = "demoKey"

def EncryptString(hash_string):
    sha_signature = hashlib.sha256(hash_string.encode()).hexdigest()
    return sha_signature

class VeinAuth:

    def __init__(self, url, fileName):
        self.url = url
        f = open(fileName) 
        s = json.load(f)

        self.fMode = s["general"]["mode"]

        if self.fMode == 0 or self.fMode == 1:
            self.limit = s["features"]["clip limit"]
            self.allowedDeviation = s["features"]["allowed angle deviation"]
            self.numberOfCells = s["features"]["number of cells"]
            self.type = s["features"]["description type"]
            self.div = s["features"]["number of code parts"]
            self.prec = s["features"]["required fuzzy extractor precision"]
            self.threshold = s["features"]["threshold"]
        elif self.fMode == 2:
            self.limit = s["intersections"]["clip limit"]
            self.spurIter = s["intersections"]["number of pruning iterations"]
            self.gridEdge = s["intersections"]["averaging grid edge length"]
            self.div = s["intersections"]["number of code parts"]
            self.prec = s["intersections"]["required fuzzy extractor precision"]
            self.threshold = s["intersections"]["threshold"]
        elif self.fMode == 3:
            self.limit = s["direct"]["clip limit"]
            self.allowedDeviation = s["direct"]["allowed angle deviation"]
            self.threshold = s["direct"]["threshold"]
        else:
            raise ValueError("Incorrect mode")

        self.showImages = s["general"]["show images"]
        self.saveImages = s["general"]["save images"]
        self.showDiag = s["general"]["show diagnostics"]

    def CaptureAndProcessImage(self, rois, i):
        img = GetImgFromUrl(self.url)
        err = False
        try: 
            rois.append(GetRoiFromImg(img,self.limit,self.saveImages,str(i),self.showImages))
        except ValueError as e:
            err = True
            print('Unable to enroll image')
            GPIO.output(12, GPIO.HIGH)
            time.sleep(0.2)
            GPIO.output(12, GPIO.LOW)
            time.sleep(1.5) # w sekundach
        if err == False:
            cv.imwrite('./out/' + str(random.randint(1,1000001)) + '.jpg',img)
            GPIO.output(4, GPIO.HIGH)
            time.sleep(0.3)
            GPIO.output(4, GPIO.LOW)

    def ProcessFeatures(self, rois):
        while len(rois) < 1:
            self.CaptureAndProcessImage(rois,0)
            time.sleep(3)
        while len(rois) < 2:
            self.CaptureAndProcessImage(rois,1)
        cl = 0
        while cl == 0:
            try:
                inp, cl = ComputeCodeFromFeatures(rois[0],rois[1],self.allowedDeviation,self.numberOfCells,self.type,self.fMode,self.showDiag,self.showImages,self.saveImages)
            except ValueError as e:
                rois = []
                while len(rois) < 1:
                    self.CaptureAndProcessImage(rois,0)
                time.sleep(3)
                while len(rois) < 2:
                    self.CaptureAndProcessImage(rois,1)
        return inp, cl

    def ProcessIntersections(self, rois):
        while len(rois) < 1:
            self.CaptureAndProcessImage(rois,0)
        inp, cl = ComputeCodeFromSkeleton(rois[0],self.spurIter,self.gridEdge,0,self.showDiag,self.showImages,self.saveImages)
        return inp, cl

    def ProcessDirect(self, rois):
        while len(rois) < 1:
            self.CaptureAndProcessImage(rois,0)
        kp1, des1 = GetFeatures(rois[0])
        return kp1, des1

    def SecureKeys(self,keys):
        codes = []
        for i in keys:
            codes.append((EncryptString(str(i) + salt).encode('utf-8')))
        if self.showDiag == 1:
            print('\nCodes:')
            print(codes)
        CryptoIO.write(codes)

    def SaveHelpers(self,helpers):
        d = ""
        fh = open("outH.txt","w+")
        for i in range(0, len(helpers)):
            for j in range(0, len(helpers[i])):
                for k in range(0, len(helpers[i][j])):
                    for n in range(0,len(helpers[i][j][k])):
                        if n == len(helpers[i][j][k])-1:
                            d += str(helpers[i][j][k][n])
                        else:
                            d += str(helpers[i][j][k][n])+','
                    if k != len(helpers[i][j])-1:
                        d += " "
                if j != len(helpers[i])-1:
                    d += ";"
            if i != len(helpers)-1:
                d += "\n"
        fh.write(d)

    def EncryptAndSaveData(self,kp,des):
        d = ""
        fk = open("outKp.txt","w+")
        for i in range(len(kp)):
            for j in range(len(kp[i])):
                d += str(kp[i][j].pt[0])+','
                d += str(kp[i][j].pt[1])+','
                d += str(kp[i][j].size)+','
                d += str(kp[i][j].angle)+','
                d += str(kp[i][j].response)+','
                d += str(kp[i][j].octave)+','
                d += str(kp[i][j].class_id)
                if j != len(kp[i])-1:
                    d += ';'
            if i != len(kp)-1:
                d += '\n';
        private_key = hashlib.sha256(key.encode("utf-8")).digest()
        CryptoIO.write([private_key[0:32]])
        if self.showDiag == 1:
            print('\nSaved key:')
            print(private_key[0:32])
        out = AES.encrypt(d,private_key[0:32])
        fk.write(out.decode('utf-8'))

        d = ""
        fk = open("outDs.txt","w+")
        for i in range(len(des)):
            for j in range(len(des[i])):
                for k in range(len(des[i][j])):
                    d += str(des[i][j][k])
                    if k != len(des[i][j])-1:
                        d += ','
                if j != len(des[i])-1:
                    d += ';'
            if i != len(des)-1:
                d += '\n'
        out = AES.encrypt(d,private_key[0:32])
        fk.write(out.decode('utf-8'))

    def Register(self):
        rois = []

        GPIO.output(23, GPIO.HIGH)
        GPIO.output(24, GPIO.HIGH)
        time.sleep(3)
        if self.fMode == 0 or self.fMode == 1:
            inp, cl = self.ProcessFeatures(rois)
        elif self.fMode == 2:
            inp, cl = self.ProcessIntersections(rois)
        else:
            kp, des = self.ProcessDirect(rois)
        GPIO.output(23, GPIO.LOW)
        GPIO.output(24, GPIO.LOW)

        if self.fMode != 3:
            inp = np.fromstring(inp, dtype=np.uint8, sep=',')
            gate = FuzzyGate(cl,self.div,self.prec)
            khs = gate.Generate(inp)
            keys = [ seq[0] for seq in khs ]
            helpers = [ seq[1] for seq in khs ]
            
            if self.showDiag == 1:
                print('\nKeys:')
                print(keys)
            
            self.SecureKeys(keys)
            
            if self.showDiag == 1:
                print('\nSaving helpers')
            self.SaveHelpers(helpers)
        else:
            self.EncryptAndSaveData(kp,des)

        GPIO.output(4, GPIO.HIGH)
        time.sleep(2)
        GPIO.output(4, GPIO.LOW)
        print('Registration completed')

    def LoadKeys(self,num):
        return CryptoIO.read(num)

    def LoadHelpers(self):
        fh = open("outH.txt", "r")
        content = fh.readlines()
        newH = []
        for line in content:
            newLine = []
            cells = line.split(';')
            for cell in cells:
                arrays = cell.split(' ')
                newCell = []
                first = True
                for array in arrays:
                    newArray = []
                    values = array.split(',')
                    for value in values:
                        newArray.append(int(value))
                    if first is True:
                        newCell = np.array(newArray, dtype='uint8')
                        first = False
                    else:
                        newCell = np.vstack((newCell,newArray))
                newLine.append(newCell.astype('uint8'))
            newH.append(tuple(newLine))
        return newH

    def Evaluate(self, keys, ks):
        i = 0
        cnt = 0
        for a in keys:
            if a == ks[i][0:32]:
                cnt += 1
            i += 1

        print('\nCorrect:')
        print(str(cnt) + '/' + str(len(ks)))
        if cnt/float(len(ks)) > self.threshold:
            print('Accepted')
            GPIO.output(4, GPIO.HIGH)
            time.sleep(2)
            GPIO.output(4, GPIO.LOW)
        else:
            print('Rejected')
            GPIO.output(12, GPIO.HIGH)
            time.sleep(2)
            GPIO.output(12, GPIO.LOW)

    def LoadData(self):
        fkp = open("outKp.txt", "r")
        enc = fkp.read()
        key = CryptoIO.read(1)
        key = bytes(key[0])
        if self.showDiag == 1:
            print('\nLoaded key:')
            print(key)
        dec = AES.decrypt(enc,key)
        dec = dec.decode('utf-8')
        content = dec.split('\n')
        kps = []
        for line in content:
            ar = []
            cells = line.split(';')
            for x in cells:
                c = x.split(',')
                kp = cv.KeyPoint(float(c[0]),float(c[1]),float(c[2]),float(c[3]),float(c[4]),int(c[5]),int(c[6]))
                ar.append(kp)
            kps.append(ar)

        fds = open("outDs.txt", "r")
        enc = fds.read()
        dec = AES.decrypt(enc,key)
        dec = dec.decode('utf-8')
        content = dec.split('\n')
        des = []
        for line in content:
            cells = line.split(';')
            first = True
            for c in cells:
                ar = []
                values = c.split(',')
                for v in values:
                    ar.append(int(v))
                if first is True:
                    ou = np.array(ar, dtype='uint8')
                    first = False
                else:
                    ou = np.vstack((ou,ar))
            des.append(ou.astype('uint8'))

        return kps, des
                
    def Verify(self):
        rois = []
    
        GPIO.output(23, GPIO.HIGH)
        GPIO.output(24, GPIO.HIGH)
        time.sleep(3)
        if self.fMode == 0 or self.fMode == 1:
            inp, cl = self.ProcessFeatures(rois)
        elif self.fMode == 2:
            inp, cl = self.ProcessIntersections(rois)
        else:
            kp, des = self.ProcessDirect(rois)
        GPIO.output(23, GPIO.LOW)
        GPIO.output(24, GPIO.LOW)

        if self.fMode != 3:
            inp = np.fromstring(inp, dtype=np.uint8, sep=',')
            gate = FuzzyGate(cl,self.div,self.prec)

            keys = self.LoadKeys(self.div)
            
            tmpK = []
            for k in keys:
                tmpK.append(k.decode("utf-8"));
            keys = tmpK
            
            if self.showDiag == 1:
                print('\nLoading helpers')
            
            helpers = self.LoadHelpers()
            
            if self.showDiag == 1:
                print('\nLoaded codes:')
                print(keys)
    
            ks = gate.Reproduce(inp, helpers)
            
            if self.showDiag == 1:
                print('\nNew keys:')
                print(ks)
                
            tmpKs = []
            letters = string.ascii_lowercase
            for i in ks:
                if i is None:
                    i = ''.join(random.choice(letters) for i in range(cl))
                tmpKs.append(EncryptString(str(i) + salt))
            ks = tmpKs
            
            if self.showDiag == 1:
                print('\nNew codes:')
                print(ks)
            
            self.Evaluate(keys, ks)
        else:
            kp2, des2 = self.LoadData()
            sum = Match(rois[0],None,des,kp,des2,kp2,self.allowedDeviation,self.showDiag,self.showImages,self.saveImages)
            if sum > self.threshold:
                print('Accepted')
                GPIO.output(4, GPIO.HIGH)
                time.sleep(2)
                GPIO.output(4, GPIO.LOW)
            else:
                print('Rejected') 
                GPIO.output(12, GPIO.HIGH)
                time.sleep(2)
                GPIO.output(12, GPIO.LOW)



