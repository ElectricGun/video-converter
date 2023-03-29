import os
import time as t
import multiprocessing as mp
import cv2
import math as Math

outputDir = ""    #insert output dir
media = ""    #insert file
length = int(cv2.VideoCapture(media).get(cv2.CAP_PROP_FRAME_COUNT))
resize = 100 / 100    #resize percentage

cpuThreads = mp.cpu_count()
nprocesses = 12
framesPerProcess = Math.ceil(length / nprocesses)     #rough frames per process amount

def parseToFile(media, outputDir, batchLength, endFrame, processName):

    fileName = media.split("/")[-1]
    outputFileName = os.path.join(outputDir, fileName+"_raw"+str(processName)+".txt")
    outputFile = open(outputFileName, "w")

    startFrame = processName * framesPerProcess

    #print("Process:", processName, startFrame, endFrame, batchLength, "\n")
    for currFrame in range(startFrame, endFrame):
        outputFile.write("[")    #open array
        
        cap = cv2.VideoCapture(media)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.set(1, currFrame)                                               
        res, frame = cap.read()
        height, width, channels = frame.shape
                                   
        newHeight, newWidth = Math.ceil(height * resize), Math.ceil(width * resize)
        try:
            skipH, skipW =  height / newHeight, width / newWidth
        except ZeroDivisionError:
            skipH, skipW = 1, 1

        skipsHeight, skipsWidth = [], []
        remainder0, remainder1 = 0, 0

    ###############################################     define skips as array to work with a non integer resize factor

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
                    bright = Math.floor(sum(frame[pickerY][pickerX]) / 255)

                except IndexError:
                    pass
                outputY.append(str(bright))
                                                
        outputString = "],\n[".join(",".join(str(x[0]) for x in y) for y in output)
        outputFile.write("["+outputString+"]")
        outputFile.write("]\n")    #close array
        
        if processName == 0:
            print(str(((currFrame - startFrame) / (endFrame - startFrame) * 100))+"% complete"+"\n")           #print percentage
    print("Process", processName + 1, "complete")
    outputFile.close()



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