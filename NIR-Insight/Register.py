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
import time

url = 'http://172.16.44.102/camera/cam_pic.php'

f = open('settings.json') 
s = json.load(f)

fMode = s["general"]["mode"]
if fMode == 2:
    limit = s["intersections"]["clip limit"]
else:
    limit = s["features"]["clip limit"]
showImages = s["general"]["show images"]
saveImages = s["general"]["save images"]
showDiag = s["general"]["show diagnostics"]

def CaptureAndProcessImage(i):
    img = GetImgFromUrl(url)
    #cv.imshow('img',img)
    #cv.waitKey(0)
    try: 
        rois.append(GetRoiFromImg(img,limit,saveImages,str(i),showImages))
    except ValueError as e:
        print('Unable to enroll img ' + str(i))
        time.sleep(1.5) # w sekundach

rois = []

if fMode == 2:
    spurIter = s["intersections"]["number of pruning iterations"]
    gridEdge = s["intersections"]["averaging grid edge length"]
    while len(rois) < 1:
        CaptureAndProcessImage(0)
    ComputeCodeFromSkeleton(rois[0],spurIter,gridEdge,0,showDiag,showImages,saveImages)
    div = s["intersections"]["number of code parts"]
    prec = s["intersections"]["required fuzzy extractor precision"]
else:
    while len(rois) < 1:
        CaptureAndProcessImage(0)
        time.sleep(3)
    while len(rois) < 2:
        CaptureAndProcessImage(1)
    allowedDeviation = s["features"]["allowed angle deviation"]
    numberOfCells = s["features"]["number of cells"]
    type = s["features"]["description type"]
    cl = 0
    while cl == 0:
        try:
            inp, cl = ComputeCodeFromFeatures(rois[0],rois[1],allowedDeviation,numberOfCells,type,fMode,showDiag,showImages,saveImages)
        except ValueError as e:
            rois = []
            while len(rois) < 1:
                CaptureAndProcessImage(0)
            time.sleep(3)
            while len(rois) < 2:
                CaptureAndProcessImage(1)
    inp = np.fromstring(inp, dtype=np.uint8, sep=',')
    div = s["features"]["number of code parts"]
    prec = s["features"]["required fuzzy extractor precision"]

#print(len(cl))
#print(len(cl)/div)
gate = FuzzyGate(len(cl),div,prec)
khs = gate.Generate(inp)
keys = [ seq[0] for seq in khs ]
helpers = [ seq[1] for seq in khs ]
#print(keys)

fk = open("outK.txt","w+")
for i in range(0, len(keys)):
     fk.write(f"{keys[i][0]} {keys[i][1]}\n")

fh = open("outH.txt","w+")
for i in range(0, len(helpers)):
    for j in range(0, len(helpers[i])):
        for k in range(0, len(helpers[i][j])):
            for n in range(0,len(helpers[i][j][k])):
                if n == len(helpers[i][j][k])-1:
                    fh.write(f"{helpers[i][j][k][n]}")
                else:
                    fh.write(f"{helpers[i][j][k][n]},")
            if k != len(helpers[i][j])-1:
                fh.write(" ")
        if j != len(helpers[i])-1:
            fh.write(";")
    if i != len(helpers)-1:
        fh.write("\n")
