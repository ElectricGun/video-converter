#TODO: mpeg style compression algorithm to render massive resolutions at more than 1 fps lag probably wont go away because sorters are laggy
#      use walls instead of sorters to avoid lag
#      give each pixel a position or skip value, dont draw over similar pixels

import os
import json
import multiprocessing as mp
from queue import Queue
import numpy as np
import math as Math
import time as t
try:
    import cv2
except:
    print("OpenCV is not installed!")

outputDir = "./output"    #insert output dir
media = "media"    #insert file
fileName = media.split("/")[-1]

length = int(cv2.VideoCapture(media).get(cv2.CAP_PROP_FRAME_COUNT))
length -= 2    #gets rid of last few frames to avoid weird bug
fps = cv2.VideoCapture(media).get(cv2.CAP_PROP_FPS)

cpuThreads = mp.cpu_count()
nprocesses = cpuThreads

treshold = 500000    #maximum array size per file
lengthOverride = 10    #length override in seconds
lengthOverride = int(lengthOverride * fps)

if (lengthOverride > length / fps):
    pass
elif (lengthOverride <= 0):
    pass
else:
    length = lengthOverride
    
resize = 10 / 100    #resize percentage
#nprocesses = 1    #number of processes override
step = 1    #number of frames per step


framesPerProcess = Math.ceil(length / nprocesses)    #rough frames per process amount

palette = [    #values are adjusted red = blast, green = plast, blue = titanium
    [217, 157, 115], [140, 127, 169], [235, 238, 245], [178, 198, 210],    #copper, lead, metaglass, graphite
    [247, 203, 164], [39, 39, 39], [50, 50, 255], [249, 163, 199],    #sand, coal, titanium, thorium
    [119, 119, 119], [83, 86, 92], [50, 255, 50], [244, 186, 110],    #scrap, silicon, plastanium, phase
    [241, 231, 120], [255, 50, 255], [255, 50, 50], [253, 169, 94]    #surge, spore, blast, pyratite
]

resources = ("copper, lead, metaglass, graphite, sand, coal, titanium, throium, scrap, silicon, plastanium, phase, surge, spore, blast, pyratite").split(",")

def getHue(rgb):
    rgb = np.divide(rgb, 255)
    r, g, b = rgb[0], rgb[1], rgb[2]
    maxrgb = max(r, g, b)
    minrgb = min(r, g, b)
    diff = maxrgb - minrgb
    if (maxrgb == minrgb):
        return 0
    if (maxrgb == r):
        return (60 * ((g - b) / diff) + 360) % 360
    elif (maxrgb == g):
        return (60 * ((b - r) / diff) + 120) % 360
    elif (maxrgb == b):
        return (60 * ((r - g) / diff) + 240) % 360

def humanCol(col):
    humanEyeWeights = [0.3, 0.59, 0.11]
    return np.multiply(col, humanEyeWeights)
    
def compareCol(rgb1, rgb2):
    rgb1 = humanCol(rgb1)
    rgb2 = humanCol(rgb2)
    maxDist = 441.6729559300637
    x1, y1, z1 = rgb1[0], rgb1[1], rgb1[2]
    x2, y2, z2 = rgb2[0], rgb2[1], rgb2[2]
    dist = Math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return 1 - dist / maxDist
    #hue1 = getHue(humanCol(rgb1))
    #hue2 = getHue(humanCol(rgb2))
    #return abs(hue2 - hue1) / 360

hues = []
for col in palette:
    hues.append(getHue(col))

def colToResource(col1, palette):
    similarityIndeces = []
    for col2 in palette:
        similarityIndeces.append(compareCol(col1, col2))
    similar = similarityIndeces.index(max(similarityIndeces))
    print("waste of cpu detected")
    return similar    #get index of the highest value in indeces array

def scale(height, width, factor):    #define skips to work with an odd resize factor
                            
    newHeight, newWidth = Math.ceil(height * factor), Math.ceil(width * factor)

    try:
        skipH, skipW =  height / newHeight, width / newWidth
    except ZeroDivisionError:
        skipH, skipW = 1, 1

    skipsHeight, skipsWidth = [], []
    remainder0, remainder1 = 0, 0

    for i in range(newHeight):
        remainder0 += skipH - Math.floor(skipH)
        tempH = Math.floor(skipH)
        if remainder0 >= 1:
            tempH += 1
            remainder0 -= 1
        skipsHeight.append(tempH)

    for i in range(newWidth):
        remainder1 += skipW - Math.floor(skipW)
        tempW = Math.floor(skipW)
        if remainder1 >= 1:
            tempW += 1
            remainder1 -= 1
        skipsWidth.append(tempW) 

    return newHeight, newWidth, skipsHeight, skipsWidth

queue = Queue(maxsize=1)

def parseToFile(media, batchLength, endFrame, processName):

    startFrame = (processName * framesPerProcess) + 1
    firstStartFrame = startFrame
    
    subBatchNumber = 0
    stop = False
        
    resourceTable = {}    #table containing resources
    while True:
        print(processName, "resetting...")
        if stop == True or startFrame > endFrame: break
        #print(processName, startFrame, endFrame)

        outputFileName = os.path.join(outputDir, fileName+str(processName)+"_"+str(subBatchNumber)+"_TEMP")
        outputFile = open(outputFileName, "w")
        outputFile.write("{\"seq\":\n")    #open json
        outputFile.write("[")    #open batch array

        cap = cv2.VideoCapture(media)
        cap.set(1, 0)                                               
        res, frame = cap.read()
        height, width, channels = frame.shape    #get frame shape of first frame
        
        newHeight, newWidth, skipsHeight, skipsWidth = scale(height, width, resize)

        totalSubBatchOutputLength = 0

        for currFrame in range(startFrame, endFrame):
            outputFile.write("[")    #open frame array
            
            cap = cv2.VideoCapture(media)
            cap.set(1, currFrame)                                               
            res, frame = cap.read()
            try:
                height, width, channels = frame.shape
            except AttributeError:
                stop = True; break                                                                   
                
            #   parse and write current frame a file
            #print(processName, "parsing...", "\n")

            output, outputY = [], []
            pickerY = 0
            for y in range(0, newHeight - 1):
                pickerX = 0
                pickerY += skipsHeight[y]
                if len(outputY) > 0:
                    output.append(list(outputY))
                                                                        
                outputY = []
                for x in range(0, newWidth - 1):
                    pickerX += skipsWidth[x]
                    frameCol = frame[pickerY][pickerX]
                    #goofy ahh code
                    #try:
                    #    resourceTable = json.loads(open(os.path.join(outputDir, fileName+"colours")+".json", "r").read())
                    #    #print(json.loads(open(os.path.join(outputDir, fileName+"colours")+".json", "r").read()))
                    #except Exception as e:
                    #    #print(e)
                    #    queue.put("susssy")
                    #    flushColour(str(frameCol), colToResource(frameCol, palette))
                    #    pass
                    #
                    #try:
                    #    try:
                    #        resource = resourceTable[str(frameCol)]
                    #    except:
                    #        resource = colToResource(frameCol, palette)
                    #        resourceTable[str(frameCol)] = resource
                    #        #print(resourceTable)
                    #        queue.put("susssy")
                    #        flushColour(str(frameCol), resource)
                    try:
                        try:
                            resource = resourceTable[str(frameCol)]
                        except:
                            resource = colToResource(frameCol, palette)
                            resourceTable[str(frameCol)] = resource

                    except IndexError as e:
                        print(e)
                        resource = 5
                        pass

                    outputY.append(str(resource))

            outputLength = len(output) * len(output[0])          
            outputString = "],\n[".join(",".join(str(x[0]) for x in y) for y in output)
            outputFile.write("["+outputString+"]")
            outputFile.write("],\n")    #close frame array

            totalSubBatchOutputLength += outputLength

            if processName == 0:
                print(len(resourceTable), str(((currFrame - firstStartFrame) / (endFrame - firstStartFrame) * 100))+"% complete"+"\n")           #print percentage

            
            if (totalSubBatchOutputLength + outputLength > treshold) & (currFrame - startFrame != 0):
                subBatchNumber += 1
                startFrame = currFrame + 1
                print(processName, "Batch exceeded max treshold")
                break

            if currFrame >= endFrame - 1: stop = True
            
                
        outputFile.write("\"STOP\"]")    #close batch array
        outputFile.write("\n}")    #close json
        outputFile.close()
    print("Process", processName + 1, "complete")
    print(totalSubBatchOutputLength, "Process:", processName, "\n")   

def getFrameProperty(media, frame):
    cap = cv2.VideoCapture(media)
    cap.set(1, frame)                                               
    res, frame = cap.read()
    try:
        height, width, channels = frame.shape
    except AttributeError:
        pass
    height, width, var1, var2 = scale(height, width, resize)
    return height, width

def flushJson(var, fileName, index):

    print("Length", len(var[1]), "Index", index)
    outputFile = open(os.path.join(outputDir, fileName+str(index))+".json", "w")
    object = json.dumps({
        "fps": var[0] / step,
        "batchSize": len(var[1]), 
        "seq": var[1]
    })
    outputFile.write(object)

#goofy ahh function
#def flushColour(str, var, i):
#
#    object = {}
#    try:
#        object = json.loads(open(os.path.join(outputDir, fileName+"colours")+ i +".json", "r").read())
#        object[str] = var
#        print("object: ", object)
#        object = json.dumps(object)
#    except Exception as e:
#        print(e)
#        object = json.dumps({
#            str: var,
#        }, indent=0)
#    colourFile = open(os.path.join(outputDir, fileName+"colours")+".json", "w+")
#    print(object)
#
#    colourFile.write(object)
#    colourFile.close()
#    queue.get()

if __name__ == "__main__":

    processes = []
    totalLength = 0
    realTotalLength = 0
    for i in range(nprocesses):
        batchLength = framesPerProcess
        totalLength += batchLength
        remaindingLength = length - totalLength

        if remaindingLength < 0:
            batchLength += remaindingLength
        realTotalLength += batchLength


        process = mp.Process(target=parseToFile, args=(media,  batchLength, realTotalLength, i))
        processes.append(process)

    print("Output will be ", getFrameProperty(media, 0), "continue?")
    #input("")


    for i in range(nprocesses):
        processes[i].start()
        print(processes[i].name, "started! \n")
    
    done = 0
    while True:
        if (done == 1): break
        for i in range(nprocesses):
            if (processes[i].is_alive() == False):
                done = 1
            else: 
                done = 0
                break
    print("Complete")

    #    intitate finalisation

    batchID = 0
    subBatchID = 0
    currTotalFiles = 0
    loadFileName = os.path.join(outputDir, fileName)

    seq = []
    totalBatchLength = 0
    while True:
        subBatchID = 0
        print(loadFileName + str(batchID)+"_"+str(subBatchID)+"_TEMP")
        try:
            outputFile = open(loadFileName + str(batchID)+"_"+str(subBatchID)+"_TEMP").read()
        except FileNotFoundError:
            print("Next file not found, finishing...")
            output = [fps, list(seq)]
            flushJson(output, fileName, currTotalFiles); currTotalFiles += 1
            break

        while True:
            try:
                outputArray = json.loads(open(loadFileName + str(batchID)+"_"+str(subBatchID)+"_TEMP").read())["seq"]
                for i in range(len(outputArray)):

                    currFrame = outputArray[i]
                    frameLength = len(currFrame) * len(currFrame[0])
                    totalBatchLength += frameLength
                    if (isinstance(currFrame, str) != True):
                        seq.append(currFrame)

                    if totalBatchLength + frameLength > treshold:
                        output = [fps, list(seq)]
                        flushJson(output, fileName, currTotalFiles); currTotalFiles += 1
                        seq.clear()
                        totalBatchLength = 0

                os.remove(loadFileName + str(batchID)+"_"+str(subBatchID)+"_TEMP"); print("File deleted")
                subBatchID += 1
            except FileNotFoundError:
                print("File not found, next batch...")
                break

        batchID += 1
    print("Completed all processes, total", currTotalFiles, "files")

    headerFile = open(os.path.join(outputDir, fileName + "config")+".json", "w")
    headerFile.write('{"totalBatches": ' + str(currTotalFiles) + '}')
    headerFile.close()
