# -*- coding:utf-8 -*-


def cleaner(outputFile, features,  inputFiles, leadTimes):
    """
    clean multi label files into standard format
    header:
    time,feature1,...,featuren,label
    body:
    value
    """
    with open(outputFile, "w") as fOut:
        print >>fOut, '{0},{1},{2}'.format('Time', ",".join(features), "Label") 
        for inputFile, leadTime  in zip(inputFiles, leadTimes) :
            cleanFile(inputFile, fOut, leadTime)
            


def cleanFile(inputFile, fOut, leadTime):
    _list = []
    with open(inputFile, "r") as fIn:
        for line in fIn:
            _list.append(line.strip())
    _list1 = _list[::-1]
    label = []
    isAlert = leadTime
    for i, l in enumerate(_list1):
        if i == 0:
            label.append(l.split(" ")[-1])
            continue
        if _list1[i].endswith("2"):
            isAlert = leadTime
            label.append("2")
        else:
            if isAlert > 0 and isAlert < leadTime:
                label.append("1")
                isAlert -= 1
            elif isAlert == 0:
                isAlert = leadTime
                label.append("0")
            else:
                if _list1[i-1].endswith("2"):
                    isAlert -= 1
                    label.append("1")
                else:
                    label.append("0")

    for line, l in zip(_list, label[::-1]):
        newLine = convertLine(line)
        print >>fOut, '{0},{1}'.format(newLine, l) 

def convertLine(line, isLabel=False):
    line = line.strip()
    tokens = line.split(" ")
    r = ",".join(tokens[::2])
    if isLabel:
        r += "," + line[-1]
    return r
    

if __name__ == "__main__":
    features = ["CPU_CAP", "CPU_USAGE", "MEM_CAP", "MEM_USAGE", "CPU_AVAI", 
                "MEM_AVAI", "NET_IN", "NET_OUT", "VBD_OO", "VBD_RD", "VBD_WR", 
                "LOAD1", "LOAD5"]
    inputFiles = ["/Users/hsh/Downloads/temp/h/1.log",
                "/Users/hsh/Downloads/temp/h/2.log",
                "/Users/hsh/Downloads/temp/h/3.log",
                "/Users/hsh/Downloads/temp/h/4.log"]
    leadTimes = [50, 50, 50, 50]
    outputFile = "/Users/hsh/Downloads/temp/h/h1"
    cleaner(outputFile, features, inputFiles, leadTimes)