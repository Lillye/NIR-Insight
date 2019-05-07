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
import json

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
inp = 0
cl = 0
div = 0
prec = 0

if fMode == 2:
    spurIter = s["intersections"]["number of pruning iterations"]
    gridEdge = s["intersections"]["averaging grid edge length"]
    while len(rois) < 1:
        CaptureAndProcessImage(0)
    inp, cl = ComputeCodeFromSkeleton(rois[0],spurIter,gridEdge,0,showDiag,showImages,saveImages)
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
    inp, cl = ComputeCodeFromFeatures(rois[0],rois[1],allowedDeviation,numberOfCells,type,fMode,showDiag,showImages,saveImages)
    div = s["features"]["number of code parts"]
    prec = s["features"]["required fuzzy extractor precision"]

#print(len(cl))
#print(len(cl)/div)
inp = np.fromstring(inp, dtype=np.uint8, sep=',')
gate = FuzzyGate(cl,div,prec)

keys = []
helpers = []

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
    
helpers = newH

ks = gate.Reproduce(inp, helpers)
#print(ks)

fk = open("outK.txt", "r")
content = fk.readlines()
for line in content:
    values = line.split(' ')
    value = str(values[0]) + str(values[1].strip())
    value = value.encode('utf-8') 
    keys.append(value)

tmpKs = []
for x in ks:
    if x != None:
        y = str(x[0]) + str(x[1])
        tmpKs.append(y.encode('utf-8'))
    else:
        tmpKs.append(x)
ks = tmpKs

i = 0
cnt = 0
for a in keys:
    if a == ks[i]:
        cnt += 1
    i += 1

#print('\nCorrect:')
print(str(cnt) + '/' + str(len(ks)))

threshold = 0
if fMode == 2:
    threshold = s["intersections"]["threshold"]
else:
    threshold = s["features"]["threshold"]

if cnt/float(len(ks)) > threshold:
    print('Accepted')
else:
    print('Rejected') 