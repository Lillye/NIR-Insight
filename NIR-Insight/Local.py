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
import json

f = open('settings.json') 
s = json.load(f)

fMode = s["general"]["mode"]

imageDir = []

imageDir.append('./images/top/6.jpg')
imageDir.append('./images/top/8.jpg')
if fMode != 2:
    imageDir.append('./images/top/8.jpg')
    imageDir.append('./images/top/7.jpg')

rois = []
if fMode == 2:
    limit = s["intersections"]["clip limit"]
else:
    limit = s["features"]["clip limit"]

rois.append(GetRoi(imageDir[0],limit,'1'))
rois.append(GetRoi(imageDir[1],limit,'2'))
if fMode != 2:
    rois.append(GetRoi(imageDir[2],limit,'3'))
    rois.append(GetRoi(imageDir[3],limit,'4'))

if fMode == 2:
    spurIter = s["intersections"]["number of pruning iterations"]
    gridEdge = s["intersections"]["averaging grid edge length"]
    inp, cl = ComputeCodeFromSkeleton(rois[0],spurIter,gridEdge,0)
    onp, _ = ComputeCodeFromSkeleton(rois[1],spurIter,gridEdge,1)
else:
    allowedDeviation = s["features"]["allowed angle deviation"]
    numberOfCells = s["features"]["number of cells"]
    type = s["features"]["description type"]
    inp, cl = ComputeCodeFromFeatures(rois[0],rois[1],allowedDeviation,numberOfCells,type,fMode)
    onp, _ = ComputeCodeFromFeatures(rois[2],rois[3],allowedDeviation,numberOfCells,type,fMode)

inp = np.fromstring(inp, dtype=np.uint8, sep=',')
onp = np.fromstring(onp, dtype=np.uint8, sep=',')

div = 0
prec = 0
if fMode == 2:
    div = s["intersections"]["number of code parts"]
    prec = s["intersections"]["required fuzzy extractor precision"]
else:
    div = s["features"]["number of code parts"]
    prec = s["features"]["required fuzzy extractor precision"]

print(cl)
print(cl/div)

gate = FuzzyGate(cl,div,prec)
khs = gate.Generate(inp)
keys = [ seq[0] for seq in khs ]
helpers = [ seq[1] for seq in khs ]
print(keys)

ks = gate.Reproduce(onp, helpers)
print(ks)

i = 0
cnt = 0
for a in keys:
    if a == ks[i]:
        cnt += 1
    i += 1
print(cnt)
print(len(ks))

threshold = 0
if fMode == 2:
    threshold = s["intersections"]["threshold"]
else:
    threshold = s["features"]["threshold"]

if cnt/float(len(ks)) > threshold:
    print('Accepted')
else:
    print('Rejected') 