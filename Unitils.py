import cPickle as p  


def DumpObj(obj, fileName):
    f = file(fileName, 'w')  
    p.dump(obj, f) # dump the object to a file  
    f.close()


def LoadObj(fileName):
    f = file(fileName)
    obj = p.load(f)
    return obj


if __name__ == "__main__":
    a = [1,2,3,4]
    fileName = "test"
    DumpObj(a, fileName)
    b = LoadObj(fileName)
    print b