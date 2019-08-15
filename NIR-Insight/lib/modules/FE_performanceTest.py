from fuzzy_extractor import FuzzyExtractor
import time

def SaveHelpers(helpers):
        d = []
        fh = open("testHp.txt","w+")
        for j in range(0, len(helpers)):
            for k in range(0, len(helpers[j])):
                for n in range(0,len(helpers[j][k])):
                    if n == len(helpers[j][k])-1:
                        d.append(str(helpers[j][k][n]))
                    else:
                        d.append(str(helpers[j][k][n]))
                        d.append(',')
                if k != len(helpers[j])-1:
                    d.append(' ')
            if j != len(helpers)-1:
                d.append(';')
        fh.write(''.join(d))

extractor = FuzzyExtractor(32, 12)
t = time.time()
key, helper = extractor.generate([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
elapsed = time.time() - t
print('generate')
print(elapsed)
print(key)
t = time.time()
r_key = extractor.reproduce([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], helper)  
elapsed = time.time() - t
print('reproduce')
print(elapsed)
print(r_key)
SaveHelpers(helper)