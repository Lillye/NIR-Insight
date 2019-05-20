import random
import fastpbkdf2 
from fuzzy_extractor import FuzzyExtractor
import math
import operator
import cv2 as cv
from lib.components.Pipeline import Pipeline
from lib.modules.FuzzyGate import FuzzyGate
from lib.modules.ProcessingLine import *
from lib.modules.Stages import *
from lib.modules.Services import *
from lib.modules.VeinAuth import *
import time
import json

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
        try: 
            rois.append(GetRoiFromImg(img,self.limit,self.saveImages,str(i),self.showImages))
            cv.imwrite('./out/' + str(random.randint(1,1000001)) + '.jpg',img)
        except ValueError as e:
            print('Unable to enroll img ' + str(i))
            time.sleep(1.5) # w sekundach

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

    # todo
    def SecureKeys(self,keys):
        fk = open("outK.txt","w+")
        for i in range(0, len(keys)):
             fk.write(str(keys[i][0]) + ' ' + str(keys[i][1]) + "\n")

    def SaveHelpers(self,helpers):
        fh = open("outH.txt","w+")
        for i in range(0, len(helpers)):
            for j in range(0, len(helpers[i])):
                for k in range(0, len(helpers[i][j])):
                    for n in range(0,len(helpers[i][j][k])):
                        if n == len(helpers[i][j][k])-1:
                            fh.write(str(helpers[i][j][k][n]))
                        else:
                            fh.write(str(helpers[i][j][k][n])+',')
                    if k != len(helpers[i][j])-1:
                        fh.write(" ")
                if j != len(helpers[i])-1:
                    fh.write(";")
            if i != len(helpers)-1:
                fh.write("\n")

    # todo
    def EncryptAndSaveData(self,kp,des):
        fk = open("outKp.txt","w+")
        for i in range(len(kp)):
            for j in range(len(kp[i])):
                fk.write(str(kp[i][j].pt[0])+',')
                fk.write(str(kp[i][j].pt[1])+',')
                fk.write(str(kp[i][j].size)+',')
                fk.write(str(kp[i][j].angle)+',')
                fk.write(str(kp[i][j].response)+',')
                fk.write(str(kp[i][j].octave)+',')
                fk.write(str(kp[i][j].class_id))
                if j != len(kp[i])-1:
                    fk.write(';')
            if i != len(kp)-1:
                fk.write("\n")

        fk = open("outDs.txt","w+")
        for i in range(len(des)):
            for j in range(len(des[i])):
                for k in range(len(des[i][j])):
                    fk.write(str(des[i][j][k]))
                    if k != len(des[i][j])-1:
                        fk.write(',')
                if j != len(des[i])-1:
                    fk.write(';')
            if i != len(des)-1:
                fk.write("\n")

    def Register(self):
        rois = []

        if self.fMode == 0 or self.fMode == 1:
            inp, cl = self.ProcessFeatures(rois)
        elif self.fMode == 2:
            inp, cl = self.ProcessIntersections(rois)
        else:
            kp, des = self.ProcessDirect(rois)

        if self.fMode != 3:
            inp = np.fromstring(inp, dtype=np.uint8, sep=',')
            gate = FuzzyGate(cl,self.div,self.prec)
            khs = gate.Generate(inp)
            keys = [ seq[0] for seq in khs ]
            helpers = [ seq[1] for seq in khs ]
            self.SecureKeys(keys)
            self.SaveHelpers(helpers)
        else:
            self.EncryptAndSaveData(kp,des)

    # todo
    def LoadKeys(self):
        keys = []
        fk = open("outK.txt", "r")
        content = fk.readlines()
        for line in content:
            values = line.split(' ')
            value = str(values[0]) + str(values[1].strip())
            value = value.encode('utf-8') 
            keys.append(value)
        return keys


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
            if a == ks[i]:
                cnt += 1
            i += 1

        print('\nCorrect:')
        print(str(cnt) + '/' + str(len(ks)))
        if cnt/float(len(ks)) > self.threshold:
            print('Accepted')
        else:
            print('Rejected') 

    # todo
    def LoadData(self):
        fkp = open("outKp.txt", "r")
        content = fkp.readlines()
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
        content = fds.readlines()
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

        if self.fMode == 0 or self.fMode == 1:
            inp, cl = self.ProcessFeatures(rois)
        elif self.fMode == 2:
            inp, cl = self.ProcessIntersections(rois)
        else:
            kp, des = self.ProcessDirect(rois)

        if self.fMode != 3:
            inp = np.fromstring(inp, dtype=np.uint8, sep=',')
            gate = FuzzyGate(cl,self.div,self.prec)

            keys = self.LoadKeys()
            helpers = self.LoadHelpers()
    
            ks = gate.Reproduce(inp, helpers)
            tmpKs = []
            for x in ks:
                if x != None:
                    y = str(x[0]) + str(x[1])
                    tmpKs.append(y.encode('utf-8'))
                else:
                    tmpKs.append(x)
            ks = tmpKs
            self.Evaluate(keys, ks)
        else:
            kp2, des2 = self.LoadData()
            sum = Match(rois[0],None,des,kp,des2,kp2,self.allowedDeviation,self.showDiag,self.showImages,self.saveImages)
            if sum > self.threshold:
                print('Accepted')
            else:
                print('Rejected') 