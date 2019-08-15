from Local import *
import glob
import os

path = 'd:/c#/mgr/nir-insight/nir-insight/test-data/prawa-top'
filenames = glob.glob(os.path.join(path, '*.jpg'))
f = open("test2.log","w+")
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
       

        