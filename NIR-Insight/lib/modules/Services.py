import cv2 as cv
import numpy as np
import math
import sys
from matplotlib import pyplot as plt
from operator import add
from PIL import Image
import requests
from io import BytesIO
from ..components.Pipeline import Pipeline
from .Stages import *
from ..external.bwmorph import *


def Compare(title, img1, img2):
    out = cv.add(img1, img2)
    cv.imshow(title,out)
    cv.imwrite("./out/" + title + ".jpg", out)
    return out

def FindCorrectPoints(points, div):
    tmp = []
    tmp2 = []
    all = []
    for i in range(len(points)):
        for j in range(len(points)):
            if (points[j][1] > (points[i][1]-div)) & (points[j][1] < (points[i][1]+div)):
                tmp.append(points[j])
        all.append(tmp)
        tmp = []
    pts = []
    for list in all:
        if len(list) >= 4:
            tmp2 = sorted(list, key=lambda x: x[0])
            avp = 0
            for p in tmp2:
                avp = avp+p[1]
            avp = avp/len(tmp2)
            avp -= 5
            if (tmp2[0][1] > avp) & (tmp2[len(tmp2)-1][1] > avp):
                print(tmp2)
                pts = tmp2
            if div > 100:
                pts = tmp2
    return pts

def UpConvex(input, showImage=False):
    image = input.copy()
    M = cv.moments(image)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    upper = image[0:cY, 0:image.shape[1]]
    if showImage:
        cv.imshow('Upper',upper)
        cv.imwrite("./out/Upper.jpg", upper)
    _,contours,_ = cv.findContours(upper,2,3)
    cnt = contours[0]
    hull = cv.convexHull(cnt,returnPoints = False)
    defects = cv.convexityDefects(cnt,hull)
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        if showImage:
            cv.line(image,start,end,[175,255,0],2)
            cv.circle(image,far,5,[100,0,255],-1)
    if showImage:
        cv.circle(image, (cX, cY), 5, (0, 255, 255), -1)
        cv.imshow('ConvexUp',image)
        cv.imwrite("./out/ConvexUp.jpg", image)
    points = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        far = tuple(cnt[f][0])
        points.append(far)
    p = []
    div = 1
    need5 = False
    for point in points:
        if (point[0] < cX+15) & (point[0] > cX-15):
            need5 = True
    print(need5)
    i = 0
    iterationLimit = 150
    if need5:
        while len(p) < 5:
            p = FindCorrectPoints(points, div)
            print(p)
            div += 1
            i += 1
            if i > iterationLimit:
                raise ValueError('Unable to enroll')
    else:
        while len(p) < 4:
            p = FindCorrectPoints(points, div)
            div += 1
            i += 1
            if i > iterationLimit:
                raise ValueError('Unable to enroll')
    print(div)
    points = p
    points = sorted(points, key=lambda x: x[1])
    if len(points) >= 6:
        points = points[1:6]
    elif len(points) >= 5:
        points = points[0:5]
    else:
        points = points[0:4]
    return sorted(points, key=lambda x: x[0])

def DownConvex(input, C, showImage=False):
    image = input.copy()
    M = cv.moments(image)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    lower = image[cY:image.shape[0], 0:image.shape[1]]
    if showImage:
        cv.imshow('Lower',lower)
        cv.imwrite("./out/Lower.jpg", lower)
    _,contours,_ = cv.findContours(lower,2,3)
    cnt = contours[0]
    hull = cv.convexHull(cnt,returnPoints = False)
    defects = cv.convexityDefects(cnt,hull)
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    leftEnd = image.shape[1]
    rightEnd = 0
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        if end[0] < leftEnd:
            leftEnd = end[0]
        if end[0] > rightEnd:
            rightEnd = end[0]
        if showImage:
            start = (start[0], start[1]+cY)
            end = (end[0], end[1]+cY)
            far = (far[0], far[1]+cY)
            cv.line(image,start,end,[175,255,0],2)
            cv.circle(image,far,5,[100,0,255],-1)
    isLeft = True
    if abs(leftEnd-cX) > abs(rightEnd-cX):
        print('lewa')
    else:
        isLeft = False
        print('prawa')
    if showImage:
        cv.circle(image, (cX, cY), 5, (0, 255, 255), -1)
        cv.imshow('ConvexDown',image)
        cv.imwrite("./out/ConvexDown.jpg", image)
    points = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        far = tuple(cnt[f][0])
        points.append(far)
    points = sorted(points, key=lambda x: x[1])
    points = [i[1] for i in points]
    return max(points) + cY + C, isLeft

def AdjustLine(input,x,y):
    image = input.copy()
    adjust = False
    xl = x[0]-y[0]
    yl = x[1]-y[1]
    p = x
    r = 100
    #image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    for i in range(r):
        pint = (int(p[0]),int(p[1]))
        #cv.circle(image,pint,2,[100,0,255],-1)
        if image[pint[1]][pint[0]] < 100:
            adjust = True
            break
        p = ((p[0]-xl/r),(p[1]-yl/r))
    return adjust

def FindLineConnectingFingers(input, points, showImage=False):
    image = input.copy()
    if len(points) > 4:
        x = (int((points[0][0]+points[1][0])/2),int((points[0][1]+points[1][1])/2))
        y = (int((points[3][0]+points[4][0])/2),int((points[3][1]+points[4][1])/2))
    else:
        x = (int((points[0][0]+points[1][0])/2),int((points[0][1]+points[1][1])/2))
        y = (int((points[2][0]+points[3][0])/2),int((points[2][1]+points[3][1])/2))
    if showImage:  
        cv.circle(image, (x[0],x[1]), 8, (100, 100, 100), -1)
        cv.circle(image, y, 8, (100, 100, 100), -1)
    factor = 1
    while AdjustLine(image,x,y):
        x = (x[0]+factor,x[1]+factor)
        y = (y[0]-factor,y[1]+factor)
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    cv.line(image,x,y,(255,255,0),2)
    if showImage:  
        cv.imshow('Line',image)
        cv.imwrite("./out/Line.jpg", image)
    (x1, y1) = x
    (x2, y2) = y
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle, image

def ComputeROI(image, y, leftHand=True, showImage=False):
    image = image.copy()
    lower = np.array([250,255,0])  
    upper = np.array([255,255,0])  
    mask = cv.inRange(image, lower, upper)  
    crd = cv.findNonZero(mask)
    minX = 10000
    maxX = 0
    for i in range(crd.shape[0]):
        if crd[i][0][0] < minX:
            minX = crd[i][0][0]
        if crd[i][0][0] > maxX:
            maxX = crd[i][0][0]
    Y = crd[0][0][1]
    if leftHand:
        box = [[maxX+5, Y+4],[minX+10, Y+4],[minX+10, y],[maxX+5, y]]
    else:
        box = [[maxX-10, Y+4],[minX-5, Y+4],[minX-5, y],[maxX-10, y]]
    box = np.int0(box)
    if showImage:
        cv.drawContours(image,[box],0,(255,255,0),1)
        cv.imshow('bROI',image)
        cv.imwrite("./out/bROI.jpg", image)
    return box
    
def ShowROI(image, box):
    image = image.copy()
    cv.drawContours(image,[box],0,(255,255,0),1)
    cv.imshow('ROI',image)
    cv.imwrite("./out/ROI.jpg", image)
    
def Crop(image, box):
    return image[box[0][1]:box[2][1], box[1][0]:box[0][0]]

# From: https://stackoverflow.com/questions/41705405/finding-intersections-of-a-skeletonised-image-in-python-opencv
def ZNeighbours(x,y,image):
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1;
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1], img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]   

# From: https://stackoverflow.com/questions/41705405/finding-intersections-of-a-skeletonised-image-in-python-opencv
def FindSkeletonIntersections(skeleton):
    validIntersection = [[0,1,0,1,0,0,1,0],[0,0,1,0,1,0,0,1],[1,0,0,1,0,1,0,0],
                         [0,1,0,0,1,0,1,0],[0,0,1,0,0,1,0,1],[1,0,0,1,0,0,1,0],
                         [0,1,0,0,1,0,0,1],[1,0,1,0,0,1,0,0],[0,1,0,0,0,1,0,1],
                         [0,1,0,1,0,0,0,1],[0,1,0,1,0,1,0,0],[0,0,0,1,0,1,0,1],
                         [1,0,1,0,0,0,1,0],[1,0,1,0,1,0,0,0],[0,0,1,0,1,0,1,0],
                         [1,0,0,0,1,0,1,0],[1,0,0,1,1,1,0,0],[0,0,1,0,0,1,1,1],
                         [1,1,0,0,1,0,0,1],[0,1,1,1,0,0,1,0],[1,0,1,1,0,0,1,0],
                         [1,0,1,0,0,1,1,0],[1,0,1,1,0,1,1,0],[0,1,1,0,1,0,1,1],
                         [1,1,0,1,1,0,1,0],[1,1,0,0,1,0,1,0],[0,1,1,0,1,0,1,0],
                         [0,0,1,0,1,0,1,1],[1,0,0,1,1,0,1,0],[1,0,1,0,1,1,0,1],
                         [1,0,1,0,1,1,0,0],[1,0,1,0,1,0,0,1],[0,1,0,0,1,0,1,1],
                         [0,1,1,0,1,0,0,1],[1,1,0,1,0,0,1,0],[0,1,0,1,1,0,1,0],
                         [0,0,1,0,1,1,0,1],[1,0,1,0,0,1,0,1],[1,0,0,1,0,1,1,0],
                         [1,0,1,1,0,1,0,0]];
    image = skeleton.copy();
    image = image/255;
    intersections = list();
    for x in range(1,len(image)-1):
        for y in range(1,len(image[x])-1):
            if image[x][y] == 1:
                neighbours = ZNeighbours(x,y,image);
                valid = True;
                if neighbours in validIntersection:
                    intersections.append((y,x));
    for point1 in intersections:
        for point2 in intersections:
            if (((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 10**2) and (point1 != point2):
                intersections.remove(point2);
    intersections = list(set(intersections));
    return intersections;

def GridAverage(img, points, gridSize):
    r = gridSize
    yf = len(img) 
    xf = len(img[0])
    cx = int(xf/r)
    cy = int(yf/r)
    posx = 0
    ar = np.zeros((r,r),object)
    centers = np.zeros((r,r),object)
    for x in range(0,r):
        posy = 0
        for y in range(0,r):
            ar[x][y] = []
            for p in points:
                if (p[0] >= posx and p[0] < (posx+cx)) and (p[1] >= posy and p[1] < (posy+cy)):
                    ar[x][y].append(p)
            tmpx = 0
            tmpy = 0
            le = len(ar[x,y])
            if le > 0:
                for s in ar[x][y]:
                    tmpx += s[0]
                    tmpy += s[1]
                    ar[x][y] = (tmpx/le,tmpy/le)
            else:
                ar[x,y] = (posx+int(cx/2),posy+int(cy/2))
            centers[x,y] = (posx+int(cx/2),posy+int(cy/2))
            posy += cy
        posx += cx
    return ar.ravel(), centers.ravel()

def IntToGray(n):
    n ^= (n >> 1)
    return bin(n)[2:]

def Average(lst): 
    return sum(lst) / len(lst) 

def GetReducedVector(la,des1,n,pos):
    u = math.ceil(255/n)
    tmp = []
    for i in range(1,n+1):
        tmp.append(list(filter(lambda x: (x[pos]<i*u) & (x[pos]>((i*u)-u)),la)))
    out = []
    for j in range(len(tmp)):
        li1 = [0] * len(des1[0])
        for i in range(len(tmp[j])):
            li1 = list(map(add, li1, tmp[j][i]))
        for i in range(len(li1)):
            if len(tmp[j]) != 0:
                li1[i] = int(round(li1[i] / len(tmp[j])))
        out.extend(li1)
    print(out)
    return out

def ComputeCodeFromFeatures(img1, img2, div, featureDiv=6, featureType=4, method=0):
    if method == 0:
        #orb = cv.ORB_create(nfeatures=400,scaleFactor=1.1,edgeThreshold=25,patchSize=25)
        orb = cv.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

    if method == 1:
        star = cv.xfeatures2d.StarDetector_create(maxSize = 45, responseThreshold = 30, lineThresholdProjected = 10, lineThresholdBinarized = 8, suppressNonmaxSize = 5)
        brief = cv.xfeatures2d.BriefDescriptorExtractor_create(bytes = 32, use_orientation = False )
        kp1 = star.detect(img1,None)
        kp2 = star.detect(img2,None)
        kp1, des1 = brief.compute(img1, kp1)
        kp2, des2 = brief.compute(img2, kp2)

    '''
    ORB - 3/5
    STAR - 4/5
    '''
    #np.set_printoptions(threshold=sys.maxsize)
    #print(des1)
    #print("next")
    #print(des2)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    ddx = []
    ddy = []
    for i in range(5):
        dx = kp1[matches[i].queryIdx].pt[0] - kp2[matches[i].trainIdx].pt[0]
        dy = kp1[matches[i].queryIdx].pt[1] - kp2[matches[i].trainIdx].pt[1]
        ddx.append(dx)
        ddy.append(dy)
    adx = Average(ddx)
    ady = Average(ddy)

    m = []
    for i in range(len(matches)):
        dx = kp1[matches[i].queryIdx].pt[0] - kp2[matches[i].trainIdx].pt[0]
        dy = kp1[matches[i].queryIdx].pt[1] - kp2[matches[i].trainIdx].pt[1]
        if ((dx < (adx + div)) & (dx > (adx - div))) & ((dy < (ady + div)) & (dy > (ady - div))):
            m.append(matches[i]);

    mSorted = sorted(m, key=lambda x: kp1[x.queryIdx].response, reverse = True)
    for i in range(len(m)):
        print(kp1[mSorted[i].queryIdx].response)

    limit = 15
    matching_result = cv.drawMatches(img1, kp1, img2, kp2, mSorted[:limit], None, flags=2)

    la = []
    #print(len(m))
    for i in range(limit): #len(mSorted)
        la.append(des1[mSorted[i].queryIdx].tolist())

    print('stop')
    print('')

    li1 = GetReducedVector(la,des1,featureDiv,featureType) #0 10 8 11 4
    print('Vector')
    print(li1)
    print('')
    code = str(np.uint8(li1[0]))
    for i in range(1, len(li1)):
        code += ',' + str(np.uint8(li1[i]))

    cl = chr(li1[0])
    for i in range(1, len(li1)):
        cl += chr(li1[i])

    print(code)

    cv.imshow("Img1", img1)
    cv.imshow("Img2", img2)
    cv.imshow("Matching result", matching_result)
    cv.imwrite("./out/Img1.jpg", img1)
    cv.imwrite("./out/Img2.jpg", img2)
    cv.imwrite("./out/Matching result.jpg", matching_result)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return code, len(cl)

def ImageToPointCoordinates(image):
    w, h = image.shape
    coordinates = []
    for i in range(w):
        for j in range(h):
            if image[i][j] == 255:
                coordinates.append((j,i))
    print(coordinates)
    return coordinates

def ComputeCodeFromSkeleton(img, spurIter, gridDiv, i):
    vp = Pipeline()
    vp.Add(Blur(BlurType.Bilateral,15,0,105,105,showImage=True))
    vp.Add(Threshold(ThresholdType.Mean,23,3,True))
    vp.Add(Invert(True))
    #vp.Add(Morph(MorphType.Open,Kernel.Rectangle,3,1,True))
    vp.Add(Blur(BlurType.Gaussian,33,0,showImage=True))
    vp.Add(Threshold(ThresholdType.Otsu,showImage=True))
    #vp.Add(Morph(MorphType.Open,Kernel.Ellipse,5,1,True))
    #vp.Add(Morph(MorphType.Open,Kernel.Cross,3,1,True))
    vp.Add(Skeletonize(True))
    out = vp.Run(img)

    out[-1] = spur(out[-1].astype(bool),spurIter)
    out[-1] = img_as_ubyte(out[-1])
    cv.imshow('InterL',out[-1])
    cv.imwrite('./out/InterL.jpg', out[-1])

    check = Compare('Check - sk',img,out[-1])

    inter = FindSkeletonIntersections(out[-1])   
    #inter = ImageToPointCoordinates(out[-1])

    check = cv.cvtColor(check, cv.COLOR_GRAY2BGR)
    
    for p in inter:
        cv.circle(check,p,2,(0,255,0),2)
    cv.imshow('Inter',check)
    cv.imwrite('./out/Inter.jpg', check)

    ar, centers = GridAverage(img, inter, gridDiv)
    for p in ar:
        p = (int(p[0]),int(p[1]))
        cv.circle(check,p,2,(255,0,0),4)
    for p in centers:
        p = (int(p[0]),int(p[1]))
        cv.circle(check,p,2,(255,0,255),4)
    cv.imshow('GridPoints',check)
    cv.imwrite('./out/GridPoints' + str(i) + '.jpg',check)

    '''
    tmpx = 0
    tmpy = 0
    for p in inter:
        tmpx += p[0]
        tmpy += p[1]
    center = (tmpx/len(inter),tmpy/len(inter))
    #center = (int(len(img[0])),int(len(img)))
    dist = []
    for a in ar:
        dist.append(int(distance.euclidean(a,center)))

    p1 = (0,0)
    p2 = (0,0)
    angles = []
    dist = []
    for i in range(1,len(ar)):
        p1 = ar[i-1]
        p2 = ar[i]
        angles.append(int(math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))))
        dist.append(int(distance.euclidean(p1,p2)))
    '''
    '''
    tmpx = 0
    tmpy = 0
    for p in inter:
        tmpx += p[0]
        tmpy += p[1]
    center = (tmpx/len(inter),tmpy/len(inter))
    #center = (int(len(img[0])),int(len(img)))
    angles = []
    dist = []
    for a in ar:
        angles.append(int(math.degrees(math.atan2(center[1]-a[1], center[0]-a[0]))))
        dist.append(int(distance.euclidean(a,center)))
    '''
    tmpx = 0
    tmpy = 0
    for p in inter:
        tmpx += p[0]
        tmpy += p[1]
    center = (tmpx/len(inter),tmpy/len(inter))

    #print(centers)
    angles = []
    dist = []
    cAngles = []
    cDist = []
    for i in range(0,len(ar)):
        d = 100*(distance.euclidean(ar[i],centers[i])/(img.shape[0]/gridDiv))
        dist.append(int(d))
        deg = math.degrees(math.atan2(ar[i][0]-centers[i][0], ar[i][1]-centers[i][1]))
        if deg < 0:
            deg = 180 + abs(deg)
        angles.append(int((deg/(3.6))))
        #cAngles.append(int(math.degrees(math.atan2(center[1]-ar[i][1], center[0]-ar[i][0]))))
        #cDist.append(int(distance.euclidean(ar[i],center)))

    #code = ''.join(map(str, dist))
    code = str(np.uint8(angles[0])) + ',' + str(np.uint8(dist[0])) #+ ',' + str(np.uint8(cAngles[0])) + ',' + str(np.uint8(cDist[0]))
    for i in range(1, len(dist)):
        code += ',' + str(np.uint8(angles[i])) + ',' + str(np.uint8(dist[i])) #+ ',' + str(np.uint8(cAngles[i])) + ',' + str(np.uint8(cDist[i]))

    cl = 2 * len(dist)

    print(code)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return code, cl


def GetImgFromUrl(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB') 
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    return img

