from Local import *
import glob
import os

#path0 = 'D:/C#/Mgr/NIR-Insight/NIR-Insight/test-data/lewa-top'
#path1 = 'D:/C#/Mgr/NIR-Insight/NIR-Insight/test-data/prawa-top'
#filenames0 = glob.glob(os.path.join(path0, '*.jpg'))
#filenames1 = glob.glob(os.path.join(path1, '*.jpg'))
#f = open("test1.log","w+")
#for i in range(len(filenames0)):
#    fn1 = filenames0[i]
#    print("For: " + fn1)
#    print()
#    for j in range(len(filenames1)):
#        fn2 = filenames1[j]
#        print('Test :' + fn1 + ' ' + fn2)
#        try:
#            out = Local(os.path.join(path0, fn1),os.path.join(path1, fn2))
#            print(out)
#            f.write(str(out) + " ")
#        except Exception as ex:
#            print(ex.args)
#            f.write(str(0) + " ")
#        print()
#    f.write('\n')
#f.close()

path = 'd:/c#/mgr/nir-insight/nir-insight/test-data/prawa-top'
filenames = glob.glob(os.path.join(path, '*.jpg'))
f = open("test3.log","w+")
for i in range(len(filenames)):
    fn1 = filenames[i]
    print("for: " + fn1)
    print()
    for j in range(len(filenames)):
        fn2 = filenames[j]
        if(fn1 != fn2):
            print('test :' + fn1 + ' ' + fn2)
            try:
                out = Local(os.path.join(path, fn1),os.path.join(path, fn2))
                print(out)
                f.write(str(out) + " ")
            except Exception as ex:
                print(ex.args)
                f.write(str(0) + " ")
            print()
    f.write('\n')
f.close()
       
#from Local import *
#import glob
#import os

#path = 'D:/C#/Mgr/NIR-Insight/NIR-Insight/test-data/lewa-top'
#filenames = glob.glob(os.path.join(path, '*.jpg'))
#f = open("test.log","w+")
#for i in range(len(filenames)-1):
#    fn1 = filenames[i]
#    print("For: " + fn1)
#    print()
#    for j in range(i+1,len(filenames)):
#        fn2 = filenames[j]
#        print('Test :' + fn1 + ' ' + fn2)
#        try:
#            out = Local(os.path.join(path, fn1),os.path.join(path, fn2))
#            print(out)
#            f.write(str(out) + " ")
#        except Exception as ex:
#            print(ex.args)
#            f.write(str(0) + " ")
#        print()
#    f.write('\n')
        