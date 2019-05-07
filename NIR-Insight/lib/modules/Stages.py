import cv2 as cv
import numpy as np
from skimage import exposure
from skimage.morphology import skeletonize, skeletonize_3d
from scipy.spatial import distance
from skimage import img_as_ubyte, img_as_bool
from enum import Enum
import math

class LoadImage:

    def __init__(self, showImage=False):
        self.showImage = showImage

    def Process(self, input, i):
        output = cv.imread(input,0)
        if self.showImage:
            cv.imshow(str(i) + " Original", output)
            cv.imwrite("./out/" + str(i) +  "Original.jpg", output)
        return output

class ThresholdType(Enum):
    Mean = 0
    Gaussian = 1
    Otsu = 2

class Threshold:

    def __init__(self, type, blockSize=11, C=2, showImage=False):
        self.type = type
        self.blockSize = blockSize
        self.C = C
        self.showImage = showImage

    def Process(self, input, i):
        if self.type == ThresholdType.Mean:
            output = cv.adaptiveThreshold(input,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,self.blockSize,self.C)
        elif self.type == ThresholdType.Gaussian:
            output = cv.adaptiveThreshold(input,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,self.blockSize,self.C)
        elif self.type == ThresholdType.Otsu:
            _,output = cv.threshold(input,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        else:
            raise ValueError('Operaton type not recognised; Use one of: [Mean, Gaussian, Otsu]')
        if self.showImage:
            cv.imshow(str(i) + " Binary",output)
            cv.imwrite("./out/" + str(i) + "Binary.jpg", output)
        return output

class Kernel(Enum):
    Rectangle = 0
    Cross = 1
    Ellipse = 2

class MorphType(Enum):
    Erode = 0
    Dilate = 1
    Open = 2
    Close = 3

class Morph:

    def __init__(self, type, kernelType, kernelSize, iter, showImage=False):
        self.type = type
        self.iter = iter
        self.showImage = showImage
        if kernelType == Kernel.Rectangle:
            shape = cv.MORPH_RECT
        elif kernelType == Kernel.Cross:
            shape = cv.MORPH_CROSS
        elif kernelType == Kernel.Ellipse:
            shape = cv.MORPH_ELLIPSE
        else:
            raise ValueError('Kernel type not recognised; Use one of: [Rectangle, Cross, Ellipse]')
        self.kernel = cv.getStructuringElement(shape,(kernelSize,kernelSize))

    def Process(self, input, i):
        if self.type == MorphType.Erode:
            output = cv.erode(input,self.kernel,iterations=self.iter)
        elif self.type == MorphType.Dilate:
            output = cv.dilate(input,self.kernel,iterations=self.iter)
        elif self.type == MorphType.Open:
            output = cv.morphologyEx(input, cv.MORPH_OPEN, self.kernel, iterations=self.iter)
        elif self.type == MorphType.Close:
            output = cv.morphologyEx(input, cv.MORPH_CLOSE, self.kernel, iterations=self.iter)
        else:
            raise ValueError('Operaton type not recognised; Use one of: [Erode, Dilate, Open, Close]')
        if self.showImage:
            cv.imshow(str(i) + str(self.type), output)
            cv.imwrite("./out/" + str(i) + str(self.type) + ".jpg", output)
        return output

class BlurType(Enum):
    Average = 0
    Gaussian = 1
    Median = 2
    Bilateral = 3

class Blur:

    def __init__(self, type, kernelSize, sigmaX=0, vColor=75, vSpace=75, showImage=False):
        self.type = type
        self.kernelSize = kernelSize
        self.showImage = showImage
        if type == BlurType.Gaussian:
            self.sigmaX = sigmaX
        elif type == BlurType.Bilateral:
            self.vColor = vColor
            self.vSpace = vSpace

    def Process(self, input, i):
        if self.type == BlurType.Average:
            output = cv.blur(input,(self.kernelSize,self.kernelSize))
        elif self.type == BlurType.Gaussian:
            output = cv.GaussianBlur(input,(self.kernelSize,self.kernelSize),self.sigmaX)
        elif self.type == BlurType.Median:
            output = cv.medianBlur(input,self.kernelSize)
        elif self.type == BlurType.Bilateral:
            output = cv.bilateralFilter(input,self.kernelSize,self.vColor,self.vSpace)
        else:
            raise ValueError('Operaton type not recognised; Use one of: [Average, Gaussian, Median, Bilateral]')
        if self.showImage:
            cv.imshow(str(i) + str(self.type), output)
            cv.imwrite("./out/" + str(i) + str(self.type) + ".jpg", output)
        return output

class AdaptiveHistogramEqualization:

    def __init__(self, clipLimit, showImage=False):
        self.clipLimit = clipLimit
        self.showImage = showImage

    def Process(self, input, i):
        output = exposure.equalize_adapthist(input, clip_limit=self.clipLimit)
        output = img_as_ubyte(output)
        if self.showImage:
            cv.imshow(str(i) + " Eq", output)
            cv.imwrite("./out/" + str(i) + " Eq.jpg", output)
        return output

class Rotate:

    def __init__(self, angle, showImage=False):
        self.angle = angle
        self.showImage = showImage

    def Process(self, input, i=0):
        (h, w) = input.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv.getRotationMatrix2D((cX, cY), -self.angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        output = cv.warpAffine(input, M, (nW, nH))
        if self.showImage:
            cv.imshow(str(i) + " Rotation", output)
            cv.imwrite("./out/" + str(i) + " Rotation.jpg", output)
        return output


class ContrastAdjustment:
    
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
        #scikit-image.org/docs/dev/auto_examples/color_exposure/plot_log_gamma.html#sphx-glr-auto-examples-color-exposure-plot-log-gamma-py

class Invert:

    def __init__(self, showImage=False):
        self.showImage = showImage

    def Process(self, input, i):
        inv = cv.bitwise_not(input)
        if self.showImage:
            cv.imshow(str(i) + " Invert",inv)
            cv.imwrite("./out/" + str(i) + " Invert.jpg", inv)
        return inv

class Skeletonize:

    def __init__(self, showImage=False):
        self.showImage = showImage

    def Process(self, input, i):
        sk = skeletonize_3d(input.astype(bool))
        sk = img_as_ubyte(sk)
        if self.showImage:
            cv.imshow(str(i) + " Thin",sk)
            cv.imwrite("./out/" + str(i) + " Thin.jpg", sk)
        return sk
    import cv2 as cv
import numpy as np
from skimage import exposure
from skimage.morphology import skeletonize, skeletonize_3d
from scipy.spatial import distance
from skimage import img_as_ubyte, img_as_bool
from enum import Enum
import math

class LoadImage:

    def __init__(self, showImage=False):
        self.showImage = showImage

    def Process(self, input, i):
        output = cv.imread(input,0)
        if self.showImage:
            cv.imshow(str(i) + " Original", output)
            cv.imwrite("./out/" + str(i) +  "Original.jpg", output)
        return output

class ThresholdType(Enum):
    Mean = 0
    Gaussian = 1
    Otsu = 2

class Threshold:

    def __init__(self, type, blockSize=11, C=2, showImage=False):
        self.type = type
        self.blockSize = blockSize
        self.C = C
        self.showImage = showImage

    def Process(self, input, i):
        if self.type == ThresholdType.Mean:
            output = cv.adaptiveThreshold(input,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,self.blockSize,self.C)
        elif self.type == ThresholdType.Gaussian:
            output = cv.adaptiveThreshold(input,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,self.blockSize,self.C)
        elif self.type == ThresholdType.Otsu:
            _,output = cv.threshold(input,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        else:
            raise ValueError('Operaton type not recognised; Use one of: [Mean, Gaussian, Otsu]')
        if self.showImage:
            cv.imshow(str(i) + " Binary",output)
            cv.imwrite("./out/" + str(i) + "Binary.jpg", output)
        return output

class Kernel(Enum):
    Rectangle = 0
    Cross = 1
    Ellipse = 2

class MorphType(Enum):
    Erode = 0
    Dilate = 1
    Open = 2
    Close = 3

class Morph:

    def __init__(self, type, kernelType, kernelSize, iter, showImage=False):
        self.type = type
        self.iter = iter
        self.showImage = showImage
        if kernelType == Kernel.Rectangle:
            shape = cv.MORPH_RECT
        elif kernelType == Kernel.Cross:
            shape = cv.MORPH_CROSS
        elif kernelType == Kernel.Ellipse:
            shape = cv.MORPH_ELLIPSE
        else:
            raise ValueError('Kernel type not recognised; Use one of: [Rectangle, Cross, Ellipse]')
        self.kernel = cv.getStructuringElement(shape,(kernelSize,kernelSize))

    def Process(self, input, i):
        if self.type == MorphType.Erode:
            output = cv.erode(input,self.kernel,iterations=self.iter)
        elif self.type == MorphType.Dilate:
            output = cv.dilate(input,self.kernel,iterations=self.iter)
        elif self.type == MorphType.Open:
            output = cv.morphologyEx(input, cv.MORPH_OPEN, self.kernel, iterations=self.iter)
        elif self.type == MorphType.Close:
            output = cv.morphologyEx(input, cv.MORPH_CLOSE, self.kernel, iterations=self.iter)
        else:
            raise ValueError('Operaton type not recognised; Use one of: [Erode, Dilate, Open, Close]')
        if self.showImage:
            cv.imshow(str(i) + str(self.type), output)
            cv.imwrite("./out/" + str(i) + str(self.type) + ".jpg", output)
        return output

class BlurType(Enum):
    Average = 0
    Gaussian = 1
    Median = 2
    Bilateral = 3

class Blur:

    def __init__(self, type, kernelSize, sigmaX=0, vColor=75, vSpace=75, showImage=False):
        self.type = type
        self.kernelSize = kernelSize
        self.showImage = showImage
        if type == BlurType.Gaussian:
            self.sigmaX = sigmaX
        elif type == BlurType.Bilateral:
            self.vColor = vColor
            self.vSpace = vSpace

    def Process(self, input, i):
        if self.type == BlurType.Average:
            output = cv.blur(input,(self.kernelSize,self.kernelSize))
        elif self.type == BlurType.Gaussian:
            output = cv.GaussianBlur(input,(self.kernelSize,self.kernelSize),self.sigmaX)
        elif self.type == BlurType.Median:
            output = cv.medianBlur(input,self.kernelSize)
        elif self.type == BlurType.Bilateral:
            output = cv.bilateralFilter(input,self.kernelSize,self.vColor,self.vSpace)
        else:
            raise ValueError('Operaton type not recognised; Use one of: [Average, Gaussian, Median, Bilateral]')
        if self.showImage:
            cv.imshow(str(i) + str(self.type), output)
            cv.imwrite("./out/" + str(i) + str(self.type) + ".jpg", output)
        return output

class AdaptiveHistogramEqualization:

    def __init__(self, clipLimit, showImage=False):
        self.clipLimit = clipLimit
        self.showImage = showImage

    def Process(self, input, i):
        output = exposure.equalize_adapthist(input, clip_limit=self.clipLimit)
        output = img_as_ubyte(output)
        if self.showImage:
            cv.imshow(str(i) + " Eq", output)
            cv.imwrite("./out/" + str(i) + " Eq.jpg", output)
        return output

class Rotate:

    def __init__(self, angle, showImage=False):
        self.angle = angle
        self.showImage = showImage

    def Process(self, input, i=0):
        (h, w) = input.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv.getRotationMatrix2D((cX, cY), -self.angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        output = cv.warpAffine(input, M, (nW, nH))
        if self.showImage:
            cv.imshow(str(i) + " Rotation", output)
            cv.imwrite("./out/" + str(i) + " Rotation.jpg", output)
        return output

class GetApproximateAngle:

    def __init__(self, showImage=False):
        self.showImage = showImage

        # input: binary image
    def Process(self, input, i):
        image = input.copy()
        image = cv.resize(image,(1200,900))
        _,contours,_ = cv.findContours(image,2,3)
        cnt = contours[0]
        ellipse = cv.fitEllipse(cnt)
        ellipse = ((ellipse[0][0]+50,ellipse[0][1]+50),(ellipse[1][0],ellipse[1][1]),ellipse[2])
        eli = np.zeros((image .shape[0]+100, image .shape[1]+100, 1), np.uint8)
        cv.ellipse(eli,ellipse,(255,255,255),1)
        ret,eli = cv.threshold(eli,127,255,0)
        _,contours,_ = cv.findContours(eli,1,5)
        cnt = contours[0]
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(eli,[box],0,(255,255,255),1)
        tmp1 = (int((box[0][0]+box[3][0])/2)-50, int((box[0][1]+box[3][1])/2)-50)
        tmp2 = (int((box[1][0]+box[2][0])/2)-50, int((box[1][1]+box[2][1])/2)-50)
        tmp3 = (int((box[0][0]+box[1][0])/2)-50, int((box[0][1]+box[1][1])/2)-50)
        tmp4 = (int((box[2][0]+box[3][0])/2)-50, int((box[2][1]+box[3][1])/2)-50)
        p1 = (0,0)
        p2 = (0,0)
        if distance.euclidean(tmp1, tmp2) <= distance.euclidean(tmp3, tmp4):
            p1 = tmp1
            p2 = tmp2
        else:
            p1 = tmp3
            p2 = tmp4
        cv.line(image,p1,p2,(0,0,0),2)
        if self.showImage:
            image = cv.resize(image,(400,300))
            eli = cv.resize(eli,(500,400))
            cv.imshow(str(i) + " Ellipse",eli)
            cv.imshow(str(i) + " Line",image)
            cv.imwrite("./out/" + str(i) + " Ellipse.jpg", eli)
            cv.imwrite("./out/" + str(i) + " Line.jpg", image)
        (x1, y1) = p1
        (x2, y2) = p2
        return math.degrees(math.atan2(y2 - y1, x2 - x1))

class ContrastAdjustment:
    
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
        #scikit-image.org/docs/dev/auto_examples/color_exposure/plot_log_gamma.html#sphx-glr-auto-examples-color-exposure-plot-log-gamma-py

class Invert:

    def __init__(self, showImage=False):
        self.showImage = showImage

    def Process(self, input, i):
        inv = cv.bitwise_not(input)
        if self.showImage:
            cv.imshow(str(i) + " Invert",inv)
            cv.imwrite("./out/" + str(i) + " Invert.jpg", inv)
        return inv

class Skeletonize:

    def __init__(self, showImage=False):
        self.showImage = showImage

    def Process(self, input, i):
        sk = skeletonize_3d(input.astype(bool))
        sk = img_as_ubyte(sk)
        if self.showImage:
            cv.imshow(str(i) + " Thin",sk)
            cv.imwrite("./out/" + str(i) + " Thin.jpg", sk)
        return sk
    