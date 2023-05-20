#TODO: mpeg style compression algorithm to render huge resolutions
#      give each pixel a position or skip value, dont draw over similar pixels
#      multithreaded caching
#      optimise... or not

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
    print("OpenCV-python is not installed!")

outputDir = "./output"      #insert output directory
treshold = 500000           #maximum array size per file
colourMode = 3              #0: euclidean distance
                            #1: hsv compare
                            #2: barycentric distance
                            #3: flann
fileCaching = True          #creates file colour cache based on colour mode
                            #creating cache for the first time may be extremely slow due to badly written code
if (colourMode != 3):
    fileCaching = False     #caching is too slow without flann

resize = 40 / 100           #resize percentage
processesOverride = 0       #number of processes override
step = 4                   #frames to skip per cycle

cpuThreads = mp.cpu_count()
nprocesses = cpuThreads

if (processesOverride > 0):
    nprocesses = processesOverride

paletteCacheDir = "./flann/" + "flannCache" + ".json"
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

#    initiate flann
norm = cv2.NORM_L2
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
search_params = dict(checks=100)
fm = cv2.FlannBasedMatcher(index_params, search_params)

def flannFrame(frame, width, height, depth):
    matches = fm.match(np.asarray(frame, dtype=np.float32).reshape(-1, 3), np.asarray(palette, dtype=np.float32))
    indices = np.uint8([m.trainIdx for m in matches])
    indices = indices.reshape(height, width, depth).tolist()
    return indices

def getHSV(rgb):
    rgb = np.divide(rgb, 255)
    r, g, b = rgb[0], rgb[1], rgb[2]
    maxrgb = max(r, g, b)
    minrgb = min(r, g, b)
    diff = maxrgb - minrgb
    if (maxrgb == minrgb):
        hue = 0
    elif (maxrgb == r):
        hue = (60 * ((g - b) / diff) + 360) % 360
    elif (maxrgb == g):
        hue = (60 * ((b - r) / diff) + 120) % 360
    elif (maxrgb == b):
        hue = (60 * ((r - g) / diff) + 240) % 360
    if (maxrgb == 0):
        saturation = 0
    else:
        saturation = diff/maxrgb
    value = maxrgb
    return hue, saturation, value

def humanCol(col):
    humanEyeWeights = [0.3, 0.59, 0.11]
    return np.multiply(col, humanEyeWeights)

#paletteHuman = []
#for col in palette:
#    paletteHuman.append(humanCol(col))
paletteHSV = []
for col in palette:
    paletteHSV.append(getHSV(col))
    
def compareCol(rgb1, rgb2):
    #rgb1 = humanCol(rgb1)
    #rgb2 = humanCol(rgb2)
    maxDist = 441.6729559300637
    x1, y1, z1 = rgb1[0], rgb1[1], rgb1[2]
    x2, y2, z2 = rgb2[0], rgb2[1], rgb2[2]
    dist = Math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return 1 - dist / maxDist

def compareColHSV(hsv1, hsv2):
    maxDist = 3
    h1, s1, v1 = hsv1[0], hsv1[1], hsv1[2]
    h2, s2, v2 = hsv2[0], hsv2[1], hsv2[2] 
    dist = Math.sqrt((Math.sin(h1)*s1*v1 - Math.sin(h2)*s2*v2)**2 + (Math.cos(h1)*s1*v1 - Math.cos(h2)*s2*v2)**2 + (v2 - v1)**2)
    out = 1 - dist / maxDist
    return out

def barycentricDistance(rgb1, rgb2):
    #rgb1 = humanCol(rgb1)
    #rgb2 = humanCol(rgb2)
    rgb1, rgb2 = np.add(rgb1, 1), np.add(rgb2, 1)
    v1, v2 = (max(rgb1) / 255), (max(rgb2) / 255)
    diff = 1 - Math.sqrt((v2 - v1)**2)
    a, b, c = {"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 0, "y": 1}
    sum1, sum2 = sum(rgb1), sum(rgb2)
    average1, average2 = sum1 / 3, sum2 / 3
    rgb1 = np.divide(rgb1, sum1) 
    rgb2 = np.divide(rgb2, sum2) 
    x1, y1 = a["x"] * rgb1[0] + b["x"] * rgb1[1] + c["x"] * rgb1[2], a["y"] * rgb1[0] + b["y"] * rgb1[1] + c["y"] * rgb1[2]
    x2, y2 = a["x"] * rgb2[0] + b["x"] * rgb2[1] + c["x"] * rgb2[2], a["y"] * rgb2[0] + b["y"] * rgb2[1] + c["y"] * rgb2[2]
    dist = Math.dist([x1, y1, v1], [x2, y2, v2])
    #print(dist)
    return 1 - dist #, [x1, y1, v1 * 100], [x2, y2, v2 * 100]
    
def colToResource(col1, palette):
    similarityIndeces = []
    if colourMode == 0:
        for col2 in palette:
            similarityIndeces.append(compareCol(col1, col2))
            similar = similarityIndeces.index(max(similarityIndeces))
    elif colourMode == 1:
        for hsv2 in paletteHSV:
            hsv1 = getHSV(col1)
            similarityIndeces.append(compareColHSV(hsv1, hsv2))
            similar = similarityIndeces.index(max(similarityIndeces))
    elif colourMode == 2:
        for col2 in palette:
            similarityIndeces.append(barycentricDistance(col1, col2))
            similar = similarityIndeces.index(max(similarityIndeces))
    #elif colourMode == 3:
    #    similar =  flannFrame([[[col1]]], 1, 1)[0][0]
    else:
        for col2 in palette:
            similarityIndeces.append(compareCol(col1, col2))
            similar = similarityIndeces.index(max(similarityIndeces))

    #print(similar, resources[similar])
    return similar    #get index of the highest value in indeces array

def cache(palette):
    #cache = [[[[None] for i in range(256)] for j in range(256)] for k in range(256)]
    if colourMode != 3:
        cache = np.zeros((256, 256, 256))
        i = 0
        for r in range(256):
            for g in range(256):
                for b in range(256):
                    similarIndex = colToResource([r, g, b], palette)
                    #similarIndex = np.argmin(np.sqrt(np.sum(((palette) - np.array([r, g, b]))**2, axis=1)))
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
        #print(cache)
    return cache

if fileCaching:
    if (os.path.exists(paletteCacheDir)):
        print("Loading palette...")
        paletteCache = json.loads(open(paletteCacheDir, "r").read())["cache"]
        print("Palette found")
    else:
        print("Creating cache...")
        paletteCache = cache(palette)
        if colourMode != 3:
            open(paletteCacheDir, "w").write(json.dumps({"cache": paletteCache.tolist()}))
        else:
            open(paletteCacheDir, "w").write(json.dumps({"cache": paletteCache}))
        print("Cached palette")
    t.sleep(.5)
    cacheSize = len(paletteCache) * len(paletteCache[0]) * len(paletteCache[0][0])

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

def parseToFile(media, startFrame, endFrame, processName, fileName, outputFolderDir):    #cant have enough args

    #    define variables

    firstStartFrame = startFrame
    subBatchNumber = 0
    stop = False
    
    resourceTable = {}    #table containing resources using ram for when fileCaching == False
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
        newHeight, newWidth, skipsHeight, skipsWidth = scale(height, width, resize)

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

            #    start the conversion (might rewrite if i HAVE to (i dont))

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
                    r, g, b = frameCol[0], frameCol[1], frameCol[2]
                    if colourMode == 3:
                        try:
                            if fileCaching != True:
                                raise Exception
                            resource = paletteCache[r][g][b]
                        except Exception as e:
                            resource = flannedFrame[pickerY][pickerX]
                    else:
                        try:
                            if fileCaching != True:
                                raise Exception
                            resource = paletteCache[r][g][b]
                        except:
                            try:
                                try:
                                    resource = resourceTable[str(frameCol)]    
                                except:
                                    resource = colToResource(frameCol, palette)
                                    resourceTable[str(frameCol)] = resource    #cache colour and its resource to a table

                            except IndexError as e:
                                print(e)
                                resource = 5
                                pass

                    outputY.append(resource)

            outputLength = len(output) * len(output[0])          
            outputFile.write(str(output))
            outputFile.write(",\n")    #close frame array

            totalSubBatchOutputLength += outputLength

            if processName == 0:
                if fileCaching:
                    print(str(((currFrame - firstStartFrame) / (endFrame - firstStartFrame) * 100))+"% complete", " Cache size: ", cacheSize, "\n")    #print percentage
                else:
                    print(str(((currFrame - firstStartFrame) / (endFrame - firstStartFrame) * 100))+"% complete", " Cache size: ", len(resourceTable), "\n")

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
    height, width, var1, var2 = scale(height, width, resize)
    return width, height

def flushBatch(var, index, outputFolderDir):
    print("Length", len(var[1]), "Index", index)
    outputFile = open(os.path.join(outputFolderDir, "frame"+str(index))+".json", "w")
    object = json.dumps({
        "fps": var[0],
        "batchSize": len(var[1]), 
        "seq": var[1],
        "step": var[2]
    })
    outputFile.write(object)

def packPixel(pixel, x, y):
    packedPixel = pixel
    packedPixel |= (x << 8)
    packedPixel |= (y << 16)
    return packedPixel

def unpackPixel(packedPixel):
    pixel = (packedPixel & 0x000000FF)
    x = (packedPixel & 0x0000FF00) >> 8
    y = (packedPixel & 0x00FF0000) >> 16
    return [pixel, x, y]

def compressFrame(frame, prevFrame):
    output = []
    length = 0
    keyframe = prevFrame == None
    frame = frame[::-1]
    if keyframe != True:
        prevFrame = prevFrame[::-1]
        for y in range(len(frame)):
            #outputFrameX = []
            for x in range(len(frame[0])):
                prevPixel = prevFrame[y][x]
                currPixel = frame[y][x]
                if (currPixel == prevPixel):
                    pass#output.append(-1)
                else:
                    #packedPixel = currPixel
                    #packedPixel |= (y << 8)
                    #packedPixel |= (x << 16)
                    #outputFrameX.append(packedPixel)
                    #output.append(packPixel(currPixel, x, y))
                    output.append([currPixel, x, y])
                    length += 1
            #output.append(list(outputFrameX))
    else:
        for y in range(len(frame)):
            outputFrameX = []
            for x in range(len(frame[0])):
                currPixel = frame[y][x]
                #packedPixel = currPixel
                #packedPixel |= (y << 8)
                #packedPixel |= (x << 16)
                #outputFrameX.append(packedPixel)
                #output.append(packPixel(currPixel, x, y))
                output.append([currPixel, x, y])
                length += 1
            #output.append(list(outputFrameX))
    return output, length

def compressMedia(mediaFolder):
    mediaName = mediaFolder.split("/")[-1]
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
        print(e)

    open(os.path.join(outputFolderDir, "frameconfig.json"), "w").write('{"totalBatches": ' + str(configFile["totalBatches"]) + ', "compressed": 1}')
    prevFrame = []
    keyFrame = []

    oldLength = 0
    newLength = 0
    for i in range(configFile["totalBatches"]):    #animation level
        j = 0
        output = []
        try:
            currBatch = json.loads(open(os.path.join(mediaFolder, "frame" + str(i) + ".json"), "r").read())
            currFps = currBatch["fps"]
            currSeq = currBatch["seq"]
            oldLength += len(currSeq) * len(currSeq[0]) * len(currSeq[0][0])
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
                packedKeyFrame, currLength = compressFrame(keyFrame, None)
                newLength += currLength
                output.append(packedKeyFrame)
                prevFrame = currFrame
                j += 1
                continue
            packedFrame, currLength = compressFrame(currFrame, prevFrame)
            newLength += currLength
            output.append(packedFrame)
            prevFrame = currFrame
            j += 1
        flushBatch([currFps, output, step], i, outputFolderDir)
    print("[Compression Fininished] Old Length:", oldLength, "New Length:", newLength)



def start(media, outputDir, lengthOverride):
    startTime = t.perf_counter()

    fileName = media.split("/")[-1]
    length = int(cv2.VideoCapture(media).get(cv2.CAP_PROP_FRAME_COUNT))
    length -= 2    #gets rid of last few frames to avoid weird bug
    fps = cv2.VideoCapture(media).get(cv2.CAP_PROP_FPS)
    lengthOverride = int(lengthOverride * fps)
    outputFolderDir = ""

    if (lengthOverride > length / fps):
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

    #    Calculate lengths per process
    for i in range(nprocesses):
        batchLength = framesPerProcess
        batchRemainder = batchLength % step
        batchLength -= batchRemainder
        totalRemainder += batchRemainder
        if (totalRemainder / step >= 1):
            batchLength += step
            totalRemainder -= step
        print(batchLength, totalRemainder)
        processLengths.append(batchLength)                  ################# bookmark
        #batchLength = framesPerProcess - stepRemainder
        #totalLength += batchLength
        #remainingLength = length - totalLength

        #if remainingLength < 0:
        #    batchLength += remainingLength
        #totalLength2 += batchLength
    print(sum(processLengths))
    print(length)
    t.sleep(0)

    startFrame = 0
    endFrame = 0
    for i in range(len(processLengths)):
        if (i - 1 >= 0):
            startFrame += processLengths[i - 1]
        else:
            print("i")
            startFrame += 0
        endFrame += processLengths[i]
        print(startFrame + 1, endFrame)
        process = mp.Process(target=parseToFile, args=(media, startFrame, endFrame, i, fileName, outputFolderDir))
        processes.append(process)

    print("Output will be", getAspectRatio(media, 0), "continue?")
    t.sleep(4)

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
    loadFileName = os.path.join(outputFolderDir, fileName)

    seq = []
    totalBatchLength = 0
    while True:
        subBatchID = 0
        print(loadFileName + str(batchID)+"_"+str(subBatchID)+"_TEMP")
        try:
            outputFile = open(loadFileName + str(batchID)+"_"+str(subBatchID)+"_TEMP").read()
        except FileNotFoundError:
            print("Next file not found, finishing...")
            output = [fps, list(seq), step]
            flushBatch(output, currTotalFiles, outputFolderDir); currTotalFiles += 1
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
                        output = [fps, list(seq), step]
                        flushBatch(output, currTotalFiles, outputFolderDir); currTotalFiles += 1
                        seq.clear()
                        totalBatchLength = 0

                os.remove(loadFileName + str(batchID)+"_"+str(subBatchID)+"_TEMP"); print("File deleted")
                subBatchID += 1
            except FileNotFoundError:
                print("File not found, next batch...")
                break
        batchID += 1
    print("Completed all processes, total", currTotalFiles, "files", " Time:", t.perf_counter() - startTime, "seconds")

    headerFile = open(os.path.join(outputFolderDir, "frame" + "config")+".json", "w")
    headerFile.write('{"totalBatches": ' + str(currTotalFiles) + ', "compressed": 0}')
    headerFile.close()

start("badapple", outputDir, 20)
compressMedia("./output/badapple")
