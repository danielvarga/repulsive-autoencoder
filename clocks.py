from PIL import ImageDraw, Image
import random, math
import numpy as np

antialiasFactor = 4
targetSizeOfImage = 28
sizeOfImage = targetSizeOfImage * antialiasFactor
numberOfHands = 2
handWidth = 10 * antialiasFactor

clockHandColor = 255
clockBorderColor = 128


def randomClockHandCoords(randomAngle):
    randomAngle = random.random() * 2 * math.pi
    handBegginingXCoord = (sizeOfImage / 2) if (sizeOfImage % 2 == 0) else ((sizeOfImage + 1) / 2)
    handBegginingYCoord = (sizeOfImage / 2) if (sizeOfImage % 2 == 0) else ((sizeOfImage + 1) / 2)
    handEndXCoord = math.cos(randomAngle) * (sizeOfImage * 0.5) + handBegginingXCoord
    handEndYCoord = math.sin(randomAngle) * (sizeOfImage * 0.5) + handBegginingYCoord

    return([(handBegginingXCoord, handBegginingYCoord), (handEndXCoord, handEndYCoord)])


def clock(params):
    antialiasFactor = 4
    img = Image.new("L", (sizeOfImage, sizeOfImage), 0)
    draw = ImageDraw.Draw(img)

    # draw bounding circle of clock
    # ellipseBoundingBox = [(0, 0), (sizeOfImage - 1, sizeOfImage - 1)]
    # draw.ellipse(ellipseBoundingBox, None, clockBorderColor)

    for indx, handAngle in enumerate(params):
        if indx==0:
            clockHandColor = 255
        else :
            clockHandColor = 127
        draw.line(randomClockHandCoords(handAngle), clockHandColor, handWidth)

    img = img.resize((targetSizeOfImage, targetSizeOfImage), Image.ANTIALIAS)

    return np.array(img)

def randomClock():
    return clock(np.random.uniform(0, 2*np.pi, size=(2, )))

def generateImages():
    for name in range(100000):
        randomClock()

if __name__ == "__main__":
    generateImages()
