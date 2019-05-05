import math
import operator
import cv2 as cv
from ..components.Pipeline import Pipeline
from .Stages import *
from .Services import *

def GetRoi(imageDir,clipLimit,saveName):
    image = cv.imread(imageDir,0)
    return GetRoiFromImg(image,clipLimit,saveName)

def GetRoiFromImg(image,clipLimit,saveName):
    image = cv.resize(image,(400,300))
    cv.imshow('Original', image)
    cv.imwrite('./out/' + 'Original' + '.jpg',image)

    vp = Pipeline()
    vp.Add(Threshold(ThresholdType.Otsu,showImage=True))
    vp.Add(Morph(MorphType.Open,Kernel.Rectangle,5,5,showImage=True))
    vp.Add(Blur(BlurType.Gaussian,11,0,showImage=True))
    vp.Add(Threshold(ThresholdType.Otsu,showImage=True))
    vp.Add(Blur(BlurType.Gaussian,11,0,showImage=True))
    vp.Add(GetApproximateAngle(True))
    out = vp.Run(image)

    i = len(out) - 1
    rt = Rotate(180-out[i],True)
    out.append(rt.Process(out[i-1],i+1))
    image = rt.Process(image)
    out.append(UpConvex(out[i+1],True))
    dy = DownConvex(out[i+1],-10,True)
    angle,im = FindLineConnectingFingers(out[i+1],out[i+2],True)
    rt = Rotate(-angle,True)
    out.append(rt.Process(im,i+2))
    image = rt.Process(image)
    out.append(ComputeROI(out[-1],dy,True,True))
    ShowROI(image,out[-1])
    roi = Crop(image,out[-1])
    roi = cv.resize(roi,(400,400))
    cv.imwrite('./out/' + 'roiBeforeAHE' + '.jpg',roi)
    ah = AdaptiveHistogramEqualization(clipLimit,True)
    roi = ah.Process(roi, 0)
    cv.imshow('roi',roi)
    cv.imwrite('./out/' + saveName + '.jpg',roi)

    cv.waitKey(0)
    cv.destroyAllWindows()
    return roi