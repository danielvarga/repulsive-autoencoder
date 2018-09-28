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
    #randomAngle = random.random() * 2 * math.pi
    handBegginingXCoord = (sizeOfImage / 2) if (sizeOfImage % 2 == 0) else ((sizeOfImage + 1) / 2)
    handBegginingYCoord = (sizeOfImage / 2) if (sizeOfImage % 2 == 0) else ((sizeOfImage + 1) / 2)
    handEndXCoord = math.cos(randomAngle) * (sizeOfImage * 0.5) + handBegginingXCoord
    handEndYCoord = math.sin(randomAngle) * (sizeOfImage * 0.5) + handBegginingYCoord

    return([(handBegginingXCoord, handBegginingYCoord), (handEndXCoord, handEndYCoord)])


def clock(params):
    assert len(params) <= 3, 'RGB image can hold up to 3 hands only.'

    #img = Image.new("RGB", (sizeOfImage, sizeOfImage), 0)
    #draw = ImageDraw.Draw(img)

    # for indx, handAngle in enumerate(params):
    #     color = [0, 0, 0]
    #     color[indx] = clockHandColor
    #     color = tuple(color)
    #     draw.line(randomClockHandCoords(handAngle), color, handWidth)

    # img = img.resize((targetSizeOfImage, targetSizeOfImage), Image.ANTIALIAS)
    # return np.array(img)

    rgb_img = np.zeros((targetSizeOfImage, targetSizeOfImage, 3), 'uint8')

    for i, handAngle in enumerate(params):
        img = Image.new("L", (sizeOfImage, sizeOfImage), 0)
        draw = ImageDraw.Draw(img)
        draw.line(randomClockHandCoords(handAngle), clockHandColor, handWidth)
        img = img.resize((targetSizeOfImage, targetSizeOfImage), Image.ANTIALIAS)
        rgb_img[:, :, i] = np.array(img)

    return rgb_img


    

    

def randomClock():
    return clock(np.random.uniform(0, 2*np.pi, size=(2, )))

def generateImages():
    for name in range(100000):
        randomClock()

if __name__ == "__main__":
    generateImages()
