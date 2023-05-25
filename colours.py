import numpy as np
import math as Math

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

def compareCol(rgb1, rgb2):
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
    return 1 - dist

class Colours:
    def __init__(self, palHSV):
        self.paletteHSV = palHSV
        pass
        
    def colToResource(self, col1, palette, mode):
        similarityIndeces = []
        if mode == 0:
            for col2 in palette:
                similarityIndeces.append(self.compareCol(col1, col2))
                similar = similarityIndeces.index(max(similarityIndeces))
        elif mode == 1:
            for hsv2 in self.paletteHSV:
                hsv1 = self.getHSV(self, col1)
                similarityIndeces.append(self.compareColHSV(hsv1, hsv2))
                similar = similarityIndeces.index(max(similarityIndeces))
        elif mode == 2:
            for col2 in palette:
                similarityIndeces.append(self.barycentricDistance(col1, col2))
                similar = similarityIndeces.index(max(similarityIndeces))
        else:
            for col2 in palette:
                similarityIndeces.append(self.compareCol(col1, col2))
                similar = similarityIndeces.index(max(similarityIndeces))

        return similar