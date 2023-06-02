#TODO:
#test for windows  
#optimise but too lazy

import os
import json
import multiprocessing as mp
from queue import Queue
import numpy as np
import math as Math
import time as t
import shutil
try:
    import cv2
except:
    print("OpenCV-python is not installed!")
import colours
from compress import Compression as Comp
import argparse

#    Initialise

parser = argparse.ArgumentParser()
#        Required arguments
parser.add_argument("filename")
requiredArgs = parser.add_argument_group('required arguments')
requiredArgs.add_argument("-m", "--mode",     metavar= "\b", type=str, dest="mode",                                 help="Available modes: \n 'sorter' - converts video into sorter sequence \n 'raw' - converts video into colour sequence", required=True)
#        Optional arguments
parser.add_argument("-s", "--step",           metavar= "\b", type=int, default = 1,        dest="step",             help="Frame steps. Output every nth frame. (Default 1)")
parser.add_argument("-g", "--scale",          metavar= "\b", type=int, default = 1,        dest="scale",            help="Scale percentage. Scales the overall size of the media. Doesn't work with size-override (Default 100)")
parser.add_argument("-k", "--size-override",  metavar= "\b", type=str, default = None,     dest="sizeOverride",     help="Output aspect ratio. Overrides scale percentage factor. Example: '88x88' (Optional)")
parser.add_argument("-b", "--batch-treshold", metavar= "\b", type=int, default= 500000,    dest="batchTreshold",    help="Maximum array length per file. (Default 500000)")
parser.add_argument("-l", "--length-override",metavar= "\b", type=float,default= 0,        dest="lengthOverride",   help="Length of output in seconds. (Default max)")
parser.add_argument("-o", "--output",         metavar= "\b", type=str, default= "./output",dest="output",           help="Output destination (Default ./output)")
parser.add_argument("-p", "--cpu-cores",      metavar= "\b", type=int, default= "0",       dest="processesOverride",help="Amount of cpu cores to use (Default max)")
parser.add_argument("-j", "--integrity",      metavar= "\b", type=float,default= ".99",    dest="compression",      help="Higher integrity = lower compression (Default .99)")
parser.add_argument("-i", "--key-interval",   metavar= "\b", type=int,default= "30",       dest="iFrameInterval",   help="Keyframe interval (Default 30)")
args = parser.parse_args()


outputDir = args.output                 #insert output directory
treshold = args.batchTreshold           #maximum array size per file
colourMode = 3                  #0: euclidean compare
                                #1: hsv compare
                                #2: barycentric compare (very goofy dont use)
                                #3: flann
                                #creating cache for the first time may be extremely slow due to badly written code
if (colourMode != 3):           
    fileCaching = False         #caching is too slow without flann
else:           
    fileCaching = True          
            
resize = args.scale / 100               #resize percentage
sizeOverride = None
if (args.sizeOverride != None):
    sizeOverrideStr = args.sizeOverride.split("x")
    sizeOverride = int(sizeOverrideStr[0]), int(sizeOverrideStr[1])
processesOverride = args.processesOverride       #number of processes override
step = args.step                #frames to skip per cycle

cpuThreads = mp.cpu_count()
nprocesses = cpuThreads

if (processesOverride > 0):
    nprocesses = processesOverride

try:
    os.mkdir("./flann")
except:
    pass
paletteCacheDir = os.path.join("./flann", "flannCache.json")
print(paletteCacheDir)

palette = [
    [217, 157, 115], [140, 127, 169], [235, 238, 245], [149, 171, 217], #copper, lead, metaglass, graphite
    [247, 203, 164], [39, 39, 39], [141, 161, 227], [249, 163, 199],    #sand, coal, titanium, thorium
    [119, 119, 119], [83, 86, 92], [203, 217, 127], [244, 186, 110],    #scrap, silicon, plastanium, phase
    [243, 233, 121], [116, 87, 206], [255, 121, 94], [255, 170, 95],    #surge, spore, blast, pyratite
    [58, 143, 100], [118, 138, 154], [227, 255, 214], [137, 118, 154],  #beryllium, tungsten, oxide, carbide
    [94, 152, 141], [223, 130, 77]                                      #fissileMatter, dormantCyst
]

resources = [    #only used in debugging
            "copper", "lead", "metaglass", "graphite",
            "sand", "coal", "titanium", "thorium",
            "scrap", "silicon", "plastanium", "phase",
            "surge", "spore", "blast", "pyratite",
            "beryllium", "tungsten", "oxide", "carbide",
            "fissileMatter", "dormantCyst"
]

Colours = colours.Colours(palette)
Compress = Comp(args.compression)

#    Initiate flann
norm = cv2.NORM_L2
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
search_params = dict(checks=100)
fm = cv2.FlannBasedMatcher(index_params, search_params)

paletteHSV = []
for col in palette:
    paletteHSV.append(colours.getHSV(col))

def flannFrame(frame, width, height, depth):
    matches = fm.match(np.asarray(frame, dtype=np.float32).reshape(-1, 3), np.asarray(palette, dtype=np.float32))
    indices = np.uint8([m.trainIdx for m in matches])
    indices = indices.reshape(height, width, depth).tolist()
    return indices

def cache(palette, mode):
    if colourMode != 3:
        cache = np.zeros((256, 256, 256))
        i = 0
        for r in range(256):
            for g in range(256):
                for b in range(256):
                    similarIndex = Colours.colToResource([r, g, b], palette, mode)
                    print("Processing palette:", ([r, g, b], resources[similarIndex]), (i / 256**3) * 100, "percent")
                    i += 1
                    cache[r][g][b] = int(similarIndex)
        cache = np.uint32(cache)
    else:
        cache = np.uint32([
        [r, g, b]
        for r in range(256)
        for g in range(256)
        for b in range(256)
        ])
        print("Flanning the cache...")
        cache = flannFrame(cache, 256, 256, 256)
    return cache

def startCache(isFlann):
    if (os.path.exists(paletteCacheDir)):
        print("Loading palette...")
        paletteCache = json.loads(open(paletteCacheDir, "r").read())["cache"]
        print("Palette found")
    else:
        print("Creating cache, subsequent runs will be faster.")
        paletteCache = cache(palette, colourMode)
        if isFlann != True:
            open(paletteCacheDir, "w").write(json.dumps({"cache": paletteCache.tolist()}))
        else:
            open(paletteCacheDir, "w").write(json.dumps({"cache": paletteCache}))
        print("Cached palette")
    t.sleep(.5)
    cacheSize = len(paletteCache) * len(paletteCache[0]) * len(paletteCache[0][0])

    return paletteCache, cacheSize

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

    return newHeight, newWidth, skipsHeight, skipsWidth, 0, 0

def scale2(height, width, newRatio):    #define skips to work with new aspect ratio

    newWidth, newHeight = min(newRatio[0], width), min(newRatio[1], height)

    cropX = 0
    cropY = 0

    skipsHeight, skipsWidth = [], []
    remainder0, remainder1 = 0, 0

    skipH = height / newHeight
    skipW = width / newWidth

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

    print(skipsWidth, skipsHeight, newWidth, newHeight)
    return newHeight, newWidth, skipsHeight, skipsWidth, cropX, cropY

if fileCaching:
    paletteCache, cacheSize = startCache(colourMode == 3)

def convert(media, startFrame, endFrame, processName, fileName, outputFolderDir, mode):    #cant have enough args  (might rewrite if i HAVE to (i dont))

    #    define variables

    firstStartFrame = startFrame
    subBatchNumber = 0
    stop = False
    
    resourceTable = {}    #resource lookup table using ram
    while True:
        
        if stop == True or startFrame >= endFrame: break    #prevent infinite loop

        #    define output file

        outputFileName = os.path.join(outputFolderDir, fileName+str(processName)+"_"+str(subBatchNumber)+"_TEMP")
        outputFile = open(outputFileName, "w")
        outputFile.write("{\"seq\":\n")    #open json
        outputFile.write("[")              #open batch array

        #    define video shape

        cap = cv2.VideoCapture(media)
        cap.set(1, 0)                                               
        _, frame = cap.read()
        height, width, channels = frame.shape    #get frame shape of first frame

        if sizeOverride == None:
            newHeight, newWidth, skipsHeight, skipsWidth, cropX, cropY = scale(height, width, resize)
        else:
            newHeight, newWidth, skipsHeight, skipsWidth, cropX, cropY = scale2(height, width, sizeOverride)

        totalSubBatchOutputLength = 0
        for currFrame in range(startFrame, endFrame, step):

            #    read frame
            
            cap.set(1, currFrame)                                               
            _, frame = cap.read()
            try:
                height, width, channels = frame.shape
            except AttributeError as e:
                print(e)
                stop = True; break                                                                   

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            flannedFrame = []
            if colourMode == 3:
                if fileCaching != True:
                    flannedFrame = flannFrame(frame, width, height, 1)

            #    start the conversion

            output, outputY = [], []
            pickerY = cropY
            #print(len(skipsHeight))
            for y in range(len(skipsHeight)):
                pickerX = cropX

                outputY = []
                for x in range(len(skipsWidth)):
                    
                    frameCol = frame[pickerY][pickerX]
                    r, g, b = frameCol[0], frameCol[1], frameCol[2]
                    if mode == 0:
                        if colourMode == 3:
                            try:
                                if fileCaching != True:
                                    raise Exception
                                resource = paletteCache[r][g][b]
                            except Exception as e:
                                resource = flannedFrame[pickerY][pickerX]
                        else:
                            try:
                                try:
                                    resource = resourceTable[str(frameCol)]    
                                except:
                                    resource = Colours.colToResource(frameCol, palette, colourMode)
                                    resourceTable[str(frameCol)] = resource    #cache colour and its resource to a table

                            except IndexError as e:
                                print(e)
                                resource = 5
                                pass
                    elif mode == 1:
                        resource = [r, g, b]
                    print(pickerX, pickerY, (r, g, b))
                    #if processName == 0: print(x, y); print(resource)
                    outputY.append(resource)
                    pickerX += skipsWidth[x]

                pickerY += skipsHeight[y]
                if len(outputY) > 0:
                    output.append(list(outputY))
                    if (processName == 1): print(len(output))
                                                                  
            outputLength = len(output) * len(output[0])
            #print(len(output))          
            outputFile.write(str(output))
            outputFile.write(",\n")    #close frame array

            totalSubBatchOutputLength += outputLength

            #if processName == 0:
            #    if fileCaching:
            #        print(str(((currFrame - firstStartFrame) / (endFrame - firstStartFrame) * 100))+"% complete", " Cache size: ", cacheSize, "\n")    #print percentage
            #    else:
            #        print(str(((currFrame - firstStartFrame) / (endFrame - firstStartFrame) * 100))+"% complete", " Cache size: ", len(resourceTable), "\n")

            if (totalSubBatchOutputLength + outputLength > treshold) & (currFrame - startFrame != 0):
                subBatchNumber += 1
                startFrame = currFrame + 1
                print(processName, "Batch exceeded max treshold")
                break

            if currFrame + step >= endFrame: stop = True    #end
                
        outputFile.write("\"STOP\"]")    #close batch array
        outputFile.write("\n}")          #close json
        outputFile.close()
    print("Process", processName + 1, "complete")

def getAspectRatio(media, frame):
    cap = cv2.VideoCapture(media)
    cap.set(1, frame)                                               
    _, frame = cap.read()
    try:
        height, width, channels = frame.shape
    except AttributeError:
        pass
    if (sizeOverride != None):
        height, width, var1, var2, cropX, cropY = scale2(height, width, sizeOverride)
    else:
        height, width, var1, var2, cropX, cropY = scale(height, width, resize)
    return width, height, cropX, cropY

def move(input, output):
    if (os.path.exists(output) != True):
        os.mkdir(output)
    try:
        shutil.move(input, output)
    except:
        try:
            os.remove(os.path.join(output, os.path.basename(input)))
        except IsADirectoryError:
            shutil.rmtree(os.path.join(output, os.path.basename(input)))
        shutil.move(input, output)

def compressMedia(mediaFolder):
    keyframeInterval = args.iFrameInterval
    
    print("Compressing...")
    mediaName = os.path.basename(mediaFolder)
    outDirectory = os.path.normpath(mediaFolder + os.sep + os.pardir)
    try:
        configFile = json.loads(open(os.path.join(mediaFolder, "frameconfig.json"), "r").read())
    except Exception as e:
        print(e)
        return
    try:
        os.mkdir(os.path.join(outDirectory, mediaName + "COMPRESSED"))
        outputFolderDir = os.path.join(outDirectory, mediaName + "COMPRESSED")
    except Exception as e:
        outputFolderDir = os.path.join(outDirectory, mediaName + "COMPRESSED")

    #open(os.path.join(outputFolderDir, "frameconfig.json"), "w").write('{"totalBatches": ' + str(configFile["totalBatches"]) + ', "compressed": 1}')
    isRaw = configFile["isRaw"]
    open(os.path.join(outputFolderDir, "frameconfig.json"), "w").write(json.dumps({"totalBatches": configFile["totalBatches"],
                                                                                   "compressed": 1,
                                                                                   "isRaw": isRaw}))
    prevFrame = []
    keyFrame = []

    oldLength = 0
    newLength = 0

    frameNumber = 0
    for i in range(configFile["totalBatches"]):    #animation level
        j = 0
        output = []
        try:
            currBatch = json.loads(open(os.path.join(mediaFolder, "frame" + str(i) + ".json"), "r").read())
            currFps = currBatch["fps"]
            currSeq = currBatch["seq"]
            if (len(currSeq) != 0):
                oldLength += len(currSeq) * len(currSeq[0]) * len(currSeq[0][0])
            else:
                oldLength += 0
                continue
        except FileNotFoundError:
            break
        while True:    #batch level
            try:
                k = 0
                currFrame = currSeq[j]
            except IndexError:
                break
            outputFrame = []
            if (len(prevFrame) <= 0):
                keyFrame = currFrame
                packedKeyFrame, currLength = Compress.compressFrame(keyFrame, None, isRaw)
                newLength += currLength
                output.append(packedKeyFrame)
                prevFrame = currFrame
                j += 1
                frameNumber += 1
                print("keyframe")
                continue
            if (keyframeInterval != 0):
                if ((frameNumber / keyframeInterval) % 1 == 0):
                    packedFrame, currLength = Compress.compressFrame(currFrame, None, isRaw)
                    newLength += currLength
                    output.append(packedFrame)
                    prevFrame = currFrame
                    j += 1
                    frameNumber += 1
                    print("keyframe")
                    continue
            packedFrame, currLength = Compress.compressFrame(currFrame, prevFrame, isRaw)
            newLength += currLength
            output.append(packedFrame)
            prevFrame = currFrame
            j += 1
            frameNumber += 1
        flushBatch([currFps, output, step], i, outputFolderDir)
        print("Compressing ", i + 1, "of", configFile["totalBatches"])
    print("[Compression Fininished] Old Length:", oldLength, "New Length:", newLength)
    move(mediaFolder, "./trash")

def flushBatch(var, index, outputFolderDir):
    #print("Length", len(var[1]), "Index", index)
    outputFile = open(os.path.join(outputFolderDir, "frame"+str(index))+".json", "w")
    object = json.dumps({
        "fps": var[0],
        "batchSize": len(var[1]),
        "step": var[2],
        "seq": var[1]
    })
    outputFile.write(object)

def start(media, outputDir, lengthOverride, mode):
    startTime = t.perf_counter()

    if (os.path.exists("./output") != True):
        os.mkdir("./output")

    fileName = os.path.basename(media)
    length = int(cv2.VideoCapture(media).get(cv2.CAP_PROP_FRAME_COUNT))
    length -= 2    #gets rid of last few frames to avoid weird bug
    fps = cv2.VideoCapture(media).get(cv2.CAP_PROP_FPS)
    lengthOverride = int(lengthOverride * fps)
    outputFolderDir = ""

    if (lengthOverride > length):
        pass
    elif (lengthOverride <= 0):
        pass
    else:
        length = lengthOverride

    try:
        os.mkdir(os.path.join(outputDir, fileName))
        outputFolderDir = os.path.join(outputDir, fileName)
    except Exception as e:
        outputFolderDir = os.path.join(outputDir, fileName)
        print(e)

    processes = []
    totalLength = 0
    totalLength2 = 0
    totalRemainder = 0
    processLengths = []
    framesPerProcess = Math.ceil(length / nprocesses)

    #    Calculate lengths

    for i in range(nprocesses):
        batchLength = framesPerProcess
        batchRemainder = batchLength % step
        batchLength -= batchRemainder
        totalRemainder += batchRemainder
        if (totalRemainder / step >= 1):
            batchLength += step
            totalRemainder -= step
        processLengths.append(batchLength)

    #    Initiate and start processes

    startFrame = 0
    endFrame = 0
    for i in range(len(processLengths)):
        if (i - 1 >= 0):
            startFrame += processLengths[i - 1]
        else:
            startFrame += 0
        endFrame += processLengths[i]
        #print(startFrame + 1, endFrame)
        process = mp.Process(target=convert, args=(media, startFrame, endFrame, i, fileName, outputFolderDir, mode))
        processes.append(process)

    print("Output will be", getAspectRatio(media, 0), "Frames:",length , "continue?")
    input()

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
    print("Conversion Complete")

    #    Finalisation

    batchID = 0
    subBatchID = 0
    currTotalFiles = 0
    loadFileName = os.path.join(outputFolderDir, fileName)

    seq = []
    totalBatchLength = 0

    #   Iterate through json files

    while True:
        subBatchID = 0
        print(loadFileName + str(batchID)+"_"+str(subBatchID)+"_TEMP")
        try:
            outputFile = open(loadFileName + str(batchID)+"_"+str(subBatchID)+"_TEMP").read()
        except FileNotFoundError:
            print("Next file not found, finishing...")
            if (len(seq) == 0):
                break
            output = [fps, list(seq), step]
            flushBatch(output, currTotalFiles, outputFolderDir); currTotalFiles += 1
            break

        #   Iterate through batches

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
                        output = [fps, list(seq), step]
                        flushBatch(output, currTotalFiles, outputFolderDir); currTotalFiles += 1
                        seq.clear()
                        totalBatchLength = 0

                os.remove(loadFileName + str(batchID)+"_"+str(subBatchID)+"_TEMP")
                #print("File deleted")
                subBatchID += 1
            except FileNotFoundError:
                print("File not found, next batch...")
                break
        batchID += 1
    print("Completed all processes, total", currTotalFiles, "files", " Time:", t.perf_counter() - startTime, "seconds")

    headerFile = open(os.path.join(outputFolderDir, "frame" + "config")+".json", "w")
    #headerFile.write('{"totalBatches": ' + str(currTotalFiles) + ', "compressed": 0}')
    headerFile.write(json.dumps({"totalBatches": currTotalFiles, "compressed": 0, "isRaw": mode == 1}))
    headerFile.close()

modes = {"sorter": 0,
         "raw": 1}

start(args.filename, outputDir, args.lengthOverride, modes[args.mode])
compressMedia(os.path.join("./output", os.path.basename(args.filename)))
