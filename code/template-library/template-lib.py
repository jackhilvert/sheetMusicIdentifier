from PIL import Image
from PIL import ImageFilter
import numpy as np
# random number generator
import random

# for drawing text over image
from PIL import ImageDraw
import math


path = "C:\\Users\\colin\\iCloudDrive\\Desktop\\Spring 2022\\B457\\githubRepo\\a1-group3\\test-images\\"

headPath = path + "template1.png"           # 10 is the default height
quarterRestPath = path + "template2.png"    # 34 is the default height
eighthRestPath = path + "template3.png"     # 28 is the default height

destinationPath = "C:\\Users\\colin\\iCloudDrive\\Desktop\\Spring 2022\\B457\\githubRepo\\a1-group3\\code\\template-library\\"


def scaleImage(pInputImagePath, pNewHeight, pDestPath, pNoteType):
    """
    Scales the image 

    pInputImagePath     str     notes
    pNewHeight          number  the height in pixels of the new image
    pDestPath           str     Path without the file name
    pNoteType           str     The prefix of the destination file name
                                file will be named: pNoteType + str(image.height) + ".png"

    str, number, str -> None
    """
    lImage = Image.open(pInputImagePath)
    lPreWidth = lImage.width
    lPreHeight = lImage.height
    lNewHeight = pNewHeight

    # we scale the width proportionally
    percentShift =  float(lNewHeight) /lPreHeight 
    print(percentShift)
    lNewWidth = int(float(lPreWidth)*( float(percentShift)))

    image = lImage.resize((lNewWidth, lNewHeight), Image.ANTIALIAS)


    image.save((pDestPath + str(pNoteType)+ str(lNewHeight) + ".png"))


for i in range(6, 28):
    scaleImage(headPath, i, destinationPath + "head\\", "noteHead")