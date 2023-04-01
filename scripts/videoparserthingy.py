import os
import json
import multiprocessing as mp
import cv2
import math as Math

outputDir = "./output"    #insert output dir
media = "media"    #insert file
fileName = media.split("/")[-1]

length = int(cv2.VideoCapture(media).get(cv2.CAP_PROP_FRAME_COUNT))
fps = cv2.VideoCapture(media).get(cv2.CAP_PROP_FPS)
length = int(10 * fps)    #length override in seconds
resize = 5 / 100    #resize percentage

cpuThreads = mp.cpu_count()
nprocesses = 12
framesPerProcess = Math.ceil(length / nprocesses)    #rough frames per process amount
treshold = 100000    #maximum array size per file


def scale(height, width, scale):    #define skips to work with an odd resize factor
                            
    newHeight, newWidth = Math.ceil(height * scale), Math.ceil(width * scale)
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
    


def parseToFile(media, batchLength, endFrame, processName):

    startFrame = processName * framesPerProcess
    

    subBatchNumber = 0
    stop = False
    while True:
        if stop == True: break

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
                
        ##############################################      parse and write current frame a file

            output, outputY = [], []
            pickerY = 0
            for y in range(0, newHeight):
                pickerX = 0
                pickerY += skipsHeight[y]
                if len(outputY) > 0:
                    output.append(list(outputY))
                                                                        
                outputY = []
                for x in range(0, newWidth):
                    pickerX += skipsWidth[x]
                    try:
                        bright = Math.floor(sum(frame   [pickerY][pickerX]) / 255)

                    except IndexError:
                        bright = 0
                        pass

                    outputY.append(str(bright))

            outputLength = len(output) * len(output[0])          
            outputString = "],\n[".join(",".join(str(x[0]) for x in y) for y in output)
            outputFile.write("["+outputString+"]")
            outputFile.write("],\n")    #close frame array

            totalSubBatchOutputLength += outputLength
            if processName == 0:
                #print(str(((currFrame - startFrame) / (endFrame - startFrame) * 100))+"% complete"+"\n", totalSubBatchOutputLength)           #print percentage
                pass
            
            if totalSubBatchOutputLength + outputLength > treshold:
                subBatchNumber += 1
                startFrame = currFrame + 1
                print("Batch exceeded max treshold")
                break

            if currFrame >= endFrame - 1: stop = True
                
        outputFile.write("\"STOP\"]")    #close batch array
        outputFile.write("\n}")    #close json
        outputFile.close()
    print("Process", processName + 1, "complete")
    print(totalSubBatchOutputLength, "Process:", processName, "\n")   


def flushJson(var, fileName, index):

    print("Length", len(var[2]), "Index", index)
    outputFile = open(os.path.join(outputDir, fileName+str(index))+".json", "w")
    object = json.dumps({"fps": var[0], "batchSize": var[1], "seq": var[2]})
    outputFile.write(object)


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

    ##############################################    intitate finalisation

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
            output = [fps, len(seq), list(seq)]
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
                        output = [fps, len(seq), list(seq)]
                        flushJson(output, fileName, currTotalFiles); currTotalFiles += 1
                        seq.clear()
                        totalBatchLength = 0

                os.remove(loadFileName + str(batchID)+"_"+str(subBatchID)+"_TEMP"); print("File deleted")
                subBatchID += 1
            except FileNotFoundError:
                print("File not found, next batch...")
                break

        batchID += 1
    print("Completed, total", currTotalFiles + 1, "files")