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

def Local(im0,im1):
    f = open('settings.json') 
    s = json.load(f)

    fMode = s["general"]["mode"]
    showImages = s["general"]["show images"]
    saveImages = s["general"]["save images"]
    showDiag = s["general"]["show diagnostics"]

    imageDir = []

    imageDir.append(im0)
    imageDir.append(im1)

    if fMode == 0 or fMode == 1:
        imageDir.append('./images/top/2.jpg')
        imageDir.append('./images/top/3.jpg')
    rois = []
    if fMode == 2:
        limit = s["veins"]["clip limit"]
    if fMode == 3:
        limit = s["direct"]["clip limit"]
    else:
        limit = s["features"]["clip limit"]

    rois.append(GetRoi(imageDir[0],limit,saveImages,'1',showImages))
    rois.append(GetRoi(imageDir[1],limit,saveImages,'2',showImages))
    if fMode == 0 or fMode == 1:
        rois.append(GetRoi(imageDir[2],limit,saveImages,'3',showImages))
        rois.append(GetRoi(imageDir[3],limit,saveImages,'4',showImages))

    if fMode == 2:
        approach = s["veins"]["approach"]
        spurIter = s["veins"]["number of pruning iterations"]
        gridEdge = s["veins"]["averaging grid edge length"]
        inp, cl = ComputeCodeFromSkeleton(rois[0],spurIter,gridEdge,0,approach,showDiag,showImages,saveImages)
        onp, _ = ComputeCodeFromSkeleton(rois[1],spurIter,gridEdge,1,approach,showDiag,showImages,saveImages)
    elif fMode == 3:
        allowedDeviation = s["direct"]["allowed angle deviation"]
        kp1, des1 = GetFeatures(rois[0])
        kp2, des2 = GetFeatures(rois[1])
    else:
        allowedDeviation = s["features"]["allowed angle deviation"]
        numberOfCells = s["features"]["number of cells"]
        type = s["features"]["description type"]
        inp, cl = ComputeCodeFromFeatures(rois[0],rois[1],allowedDeviation,numberOfCells,type,fMode,showDiag,showImages,saveImages)
        onp, _ = ComputeCodeFromFeatures(rois[2],rois[3],allowedDeviation,numberOfCells,type,fMode,showDiag,showImages,saveImages)

    if fMode != 3:
        inp = np.fromstring(inp, dtype=np.uint8, sep=',')
        onp = np.fromstring(onp, dtype=np.uint8, sep=',')

        if fMode == 2:
            div = s["veins"]["number of code parts"]
            prec = s["veins"]["required fuzzy extractor precision"]
        else:
            div = s["features"]["number of code parts"]
            prec = s["features"]["required fuzzy extractor precision"]

        #print(cl)
        #print(cl/div)
        #print()

        accVal = 0
        accDiv = 0
        for n in range(3):
            gate = FuzzyGate(cl,div,prec,0.05)
            khs = gate.Generate(inp)
            keys = [ seq[0] for seq in khs ]
            helpers = [ seq[1] for seq in khs ]

            if showDiag:
                print('\nInput keys')
                print(keys)
            ks = gate.Reproduce(onp, helpers)
            if showDiag:
                print('\nOutput keys')
                print(ks)

            i = 0
            cnt = 0
            for a in keys:
                if a == ks[i]:
                    cnt += 1
                i += 1
            print('\nCorrect:')
            print(str(cnt) + '/' + str(len(ks)))
            accVal = accVal + cnt
            accDiv = accDiv + len(ks)

        threshold = 0
        if fMode == 2:
            threshold = s["veins"]["threshold"]
        else:
            threshold = s["features"]["threshold"]

        print('\nOutput:')
        if cnt/float(len(ks)) > threshold:
            print('Accepted')
        else:
            print('Rejected') 
        return accVal/accDiv

    else:    
        value = Match(rois[0],rois[1],des1,kp1,des2,kp2,allowedDeviation,showDiag,showImages,saveImages)
        return value

if __name__ == "__main__":
    Local(2,3)