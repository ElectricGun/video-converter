import json
import os
import colours

class Compression:
    def __init__(self, treshold):
        self.treshold = treshold
        pass

    def packPixel(self, pixel, x, y):
        packedPixel = pixel
        packedPixel |= (x << 8)
        packedPixel |= (y << 16)
        return packedPixel

    def unpackPixel(self, packedPixel):
        pixel = (packedPixel & 0x000000FF)
        x = (packedPixel & 0x0000FF00) >> 8
        y = (packedPixel & 0x00FF0000) >> 16
        return [pixel, x, y]

    def compressFrame(self, frame, prevFrame, isRaw):
        treshold = self.treshold
        output = []
        length = 0
        keyframe = prevFrame == None
        frame = frame[::-1]
        if keyframe != True:
            prevFrame = prevFrame[::-1]
            for y in range(len(frame)):
                for x in range(len(frame[0])):
                    prevPixel = prevFrame[y][x]
                    currPixel = frame[y][x]
                    if (isRaw):
                        similarity = colours.compareCol(prevPixel, currPixel)
                    else:
                        similarity = currPixel == prevPixel
                    if (similarity > treshold):
                        pass
                    else:
                        output.append([currPixel, x, y])
                        length += 1
        else:
            for y in range(len(frame)):
                outputFrameX = []
                for x in range(len(frame[0])):
                    currPixel = frame[y][x]
                    output.append([currPixel, x, y])
                    length += 1
        return output, length