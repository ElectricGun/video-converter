import os
import time as t
import multiprocessing as mp
import cv2
import math as Math

outputDir = ""    #insert output dir
media = ""    #insert file
length = int(cv2.VideoCapture(media).get(cv2.CAP_PROP_FRAME_COUNT))
#length = 200    #length override
resize = 10 / 100    #resize percentage

cpuThreads = mp.cpu_count()
nprocesses = 1
framesPerProcess = Math.ceil(length / nprocesses)     #rough frames per process amount
treshold = 100000    #maximum array size per file

def scale(height, width, scale):    #define skips as array to work with a non integer resize factor
                            
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
    

def parseToFile(media, outputDir, batchLength, endFrame, processName):

    fileName = media.split("/")[-1]
    startFrame = processName * framesPerProcess
    

    subBatchNumber = 0
    stop = False
    while True:
        if stop == True: break

        outputFileName = os.path.join(outputDir, fileName+"_raw"+str(processName)+"_"+str(subBatchNumber)+".json")
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
            #fps = int(cap.get(cv2.CAP_PROP_FPS))
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

        process = mp.Process(target=parseToFile, args=(media, outputDir,  batchLength, realTotalLength, i))
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
    #intitate finalisation