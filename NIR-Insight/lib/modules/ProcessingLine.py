import math
import operator
import cv2 as cv
from ..components.Pipeline import Pipeline
from .Stages import *
from .Services import *

def GetRoi(imageDir,clipLimit,saveImages,saveName,showImages):
    image = cv.imread(imageDir,0)
    return GetRoiFromImg(image,clipLimit,saveImages,saveName,showImages)

def GetRoiFromImg(image,clipLimit,saveImages,saveName,showImages):
    image = cv.resize(image,(400,300))

    if showImages is True:
        cv.imshow('Original', image)
    if saveImages is True:
        cv.imwrite('./out/' + 'Original' + '.jpg',image)

    vp = Pipeline()
    vp.Add(Threshold(ThresholdType.Otsu,showImage=showImages))
    vp.Add(Morph(MorphType.Open,Kernel.Rectangle,5,5,showImage=showImages))
    #vp.Add(Morph(MorphType.Close,Kernel.Rectangle,2,1,showImage=showImages))
    vp.Add(Blur(BlurType.Gaussian,15,0,showImage=showImages))
    vp.Add(Threshold(ThresholdType.Otsu,showImage=showImages))
    #vp.Add(Blur(BlurType.Gaussian,11,0,showImage=showImages))
    vp.Add(GetApproximateAngle(showImages))
    out = vp.Run(image)

    i = len(out) - 1
    rt = Rotate(180-out[i],showImages)
    out.append(rt.Process(out[i-1],i+1))
    image = rt.Process(image)
    points, mx = UpConvex(out[i+1],saveImages,showImages)
    out.append(points)
    isLeft = DownConvex(out[i+1],saveImages,showImages)
    angle,dist,im = FindLineConnectingFingers(out[i+1],out[i+2], mx, saveImages, showImages)
    rt = Rotate(-angle,showImages)
    out.append(rt.Process(im,i+2))
    image = rt.Process(image)
    out.append(ComputeROI(out[-1],dist,isLeft,showImages))
    if showImages:
        ShowROI(image,out[-1],saveImages)
    roi = Crop(image,out[-1])
    roi = cv.resize(roi,(400,400))
    if saveImages:
        cv.imwrite('./out/' + 'roiBeforeAHE' + '.jpg',roi)
    ah = AdaptiveHistogramEqualization(clipLimit,showImages)
    roi = ah.Process(roi, 0)
    if saveImages:
        cv.imwrite('./out/' + saveName + '.jpg',roi)

    if showImages:
        cv.imshow('roi',roi)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    return roi