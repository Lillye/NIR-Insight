import math
import fastpbkdf2 
from fuzzy_extractor import FuzzyExtractor
import math
import operator
import cv2 as cv
import numpy as np

class FuzzyGate:

    def __init__(self, inputLength, numberOfUnits, tolerancePerUnit, repError = 0.001):
        self.unitInputLength = math.ceil(inputLength/numberOfUnits)
        self.numberOfUnits = numberOfUnits
        self.tolerancePerUnit = tolerancePerUnit
        self.extractors = []
        for i in range(numberOfUnits):
            self.extractors.append(FuzzyExtractor(self.unitInputLength, tolerancePerUnit, repError))

    def Generate(self, input):
        n = self.unitInputLength
        parts = [input[i:i+n] for i in range(0, len(input), n)]
        if len(parts[-1]) < n:
            for i in range(0, self.unitInputLength-len(parts[-1])):
                parts[-1] = np.append(parts[-1],0)
        out = []
        for i in range(0, self.numberOfUnits):
            out.append(self.extractors[i].generate(parts[i]))
        return out

    def Reproduce(self, input, helpers):
        n = self.unitInputLength
        parts = [input[i:i+n] for i in range(0, len(input), n)]
        if len(parts[-1]) < n:
            for i in range(0, self.unitInputLength-len(parts[-1])):
                parts[-1] = np.append(parts[-1],0)
        out = []
        for i in range(self.numberOfUnits):
            out.append(self.extractors[i].reproduce(parts[i], helpers[i]))
        return out
