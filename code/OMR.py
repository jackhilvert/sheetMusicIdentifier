from PIL import Image
from PIL import ImageFilter
import numpy as np
# random number generator
import random
import sys
# for drawing text over image
from PIL import ImageDraw
import functions


# helper function to find the nearest staff to y
def find_nearest_staff(y, staves, half_step):
    """
    Given some y-value corresponding to a found note
    
    return the location of the top-most bar of the corresponding staff

    """
    best = np.inf
    result = 0
    for staff in staves:
        if staff < y:
            val = staff + (half_step * 8)  # bottom line of staff
        else:
            val = staff
        res = abs(y - val)
        if res < best:
            best = res
            result = staff
    return result


def find_note(y, half_step, trebleOrBass = "treble"):
    """
    for each detected note, find what note it is by doing the following:
    1. subtract its y-coordinate from the top of the staff that is closest to its position
           (we have these staff locations from the hough_transform method)
    2. take the result of 1. and divide it by 1/2 of the height of the scaled template (+ or - 4 or 5 pixels)

    :param y: the y value of the note in the image
    :param half_step: 1/2 of the size of the space in between the staff lines
    :return: a tuple containing the note identified along with some uncertainty value
        (note string, uncertanty +/- in pixels)
    """
    confidence = 0  # no, this is not a joke

    nearest_staff = find_nearest_staff(y) # this is a y value ABCDEFG

    y = y + half_step # account for the passed y value (here we want the center of the note, not the top corner)

    # nearest_staff = 30

    bassClef = ["A", "G", "F", "E", "D", "C", "B" ] # top, half-step_down, half-step_down etc
    trebleClef = [ "F", "E", "D", "C", "B", "A", "G"] # "  "    "     "     "   "     "   "

    lDelta = y - nearest_staff  #positive: move forwards through array
                                # negative: move backwards through the array

    chosenClef = trebleClef if trebleOrBass == "treble" else bassClef
    delNotePre = lDelta / half_step   
    delNote = int(round(delNotePre, 0))
    confidence += (abs(delNote  - delNotePre) * half_step) # any chopped off pixels increases our uncertanty value
    while(delNote > 6):
        delNote -= 7
    if(lDelta > 0):
        # we move forwards through our array
        note = chosenClef[delNote]

    else:
        lDelta = -1 * lDelta
        # we move backwards through the array
        note = chosenClef[-delNote]

def load_image(path):
    return Image.open(path, mode='r')


# 8
# (a) Load in a specified music image.
# (b) Detect all of the staves using step 6 above. In addition to giving you the staves, this also gives an
# estimate of the scale of the image – i.e. the size of the note heads – since the space between staff lines is
# approximately the height of a notehead.


# (c) Rescale the image so that the note head size in the image agrees with the size of your note head
# templates. (Alternatively, you can rescale the note templates so that they agree with the image, you can
# have a library of pre-defined templates of different scales and select the appropriate one dynamically.)

def pick_templates(headsize):
    '''
    int: eighth, headsize

    returns a list of image objects that are the templates for the given sizes 
    '''
    # print(headsize)
    eighth, quarter = scaleTemplateHeights(headsize)
    headLib = "./code/template-library/head/noteHead"
    headLib += str(int(headsize)) + ".png"
    quarterLib = "./code/template-library/quarter-rest/quarterRest"
    quarterLib += str(int(quarter)) + ".png"
    eighthLib = "./code/template-library/eighth-rest/eighthRest"
    eighthLib += str(int(eighth)) + ".png"
    hImage = Image.open(headLib)
    qImage = Image.open(quarterLib)
    eImage = Image.open(eighthLib)

    return [hImage, qImage, eImage]


# (d) Detect the notes and eighth and quarter rests in the image, using the approach of step 4, step 5,
# some combination, or a new technique of your own invention. The goal is to correctly find as many symbols
# as possible, with few false positives.
def draw_notes(image, hTemplate, qTemplate, eTemplate):
    # need note head height
    hLocations = functions.locate_symbol(image, hTemplate, (255, 0, 0))
    qLocations = functions.locate_symbol(image, qTemplate, (255,0,0))
    eLocations = functions.locate_symbol(image, eTemplate, (255,0, 0))
    htemplatearr = np.array(hTemplate, dtype='int64')
    qtemplatearr = np.array(qTemplate, dtype='int64')
    etemplatearr = np.array(eTemplate, dtype='int64')
    # print(hLocations)

    draw = ImageDraw.Draw(image)
    for i in range(len(hLocations)):
        point = hLocations[i]
        upperLeft = (point[0], point[1])
        lowerRight = (point[0] + len(htemplatearr[0]), point[1] + len(htemplatearr))
        draw.rectangle((upperLeft, lowerRight), outline=(255, 0,0))
    for i in range(len(qLocations)):
        point = qLocations[i]
        upperLeft = (point[0], point[1])
        lowerRight = (point[0] + len(qtemplatearr[0]), point[1] + len( qtemplatearr))
        draw.rectangle((upperLeft, lowerRight), outline=(0,255,0))

    for i in range(len(eLocations)):
        point = eLocations[i]
        upperLeft = (point[0], point[1])
        lowerRight = (point[0] + len(etemplatearr[0]), point[1] + len(etemplatearr))
        draw.rectangle((upperLeft, lowerRight), outline= (0, 0, 255))
    return image


# Your code should output two files:
# (a) detected.png: Visualization of which notes were detected (as in Fig 1(b)).
# (b) detected.txt: A text file indicating the detection results. The text file should have one line per
# detected symbol, with the following format for each line:



# page 4 of 5
# B457 Group Assignment 1 February 3, 2022
# < row >< col >< height >< width >< symbol type >< pitch >< confidence >
# where row and col are the coordinates of the upper-left corner of the bounding box, height and width
# are the dimensions of the bounding box, symbol type is one of filled note, eighth rest, or quarter rest, pitch
# is the note letter (A through G) of the note or is an underscore ( ) if the symbol is a rest, and confidence is
# a number that should be high if the program is relatively certain about the detected symbol and low if it is
# not too sure.

def scaleTemplateHeights(pNoteHeadHeight):
    """
    Given a height of the note head (or space between the staves +/- 2)

    returns sizes other two templates:

        (eigthRestHeight,  quarterRestHeight)
    
    number -> tuple
    """
    defaultSizes = (10, 28, 34)  # head, eighth, quarter
    scaleFactor = float(pNoteHeadHeight) / float(defaultSizes[0])
    return (int(scaleFactor * defaultSizes[1]), int(scaleFactor * defaultSizes[2]))


# def detect_notes(image):
#     # run edge detection on image. return the new image

#     spacing, stave_locations = functions.hough_transform(image)

#     half_step = spacing / 2

#     # get space between staff lines from above function as well as where each staff begins

#     # TODO   we need to make a library of scaled templates so that we can use the correctly -> Colin wrote this
#     # TODO   scaled template for each staff spacing we may encounter -> Colin also wrote this

#     # get the template that is scaled such that its height is the same as the spacing between staff lines (from hough)

#     for i in range(3):
#         template = None
#         color = None
#         # if i is 0:
#         #   template = quarter_rest (scaled)
#         #   color = (0,255,0)
#         # elif i is 1:
#         #   template = eighth_rest (need to scale)
#         #   color = (0,0,255)
#         # else:
#         #   template = note_head (scaled based on spacing of lines)
#         #   color = (255,0,0)

#         located, image = functions.locate_symbol(image, template,
                                    # color)  # gives array of tuples detailing the x,y positions for each found template
#         if i is 2:
#             for location in located:
#                 # here, y represents the y-value of the note that we have detected.
#                 note, uncertainty = find_note(located[0], half_step)  # may need to be located[1]

#                 # write note name in the output image

#     # here, y represents the y-value of the note that we have detected.
#     y = 0

#     note, uncertainty = find_note(y, half_step)

    # #write the note name on the image at location


def detection(path):
    image = load_image(path)

    stave_locations, note_height = functions.hough_transform(image)
    
    half_step = note_height / 2
    # print(note_height)
    templates = pick_templates(note_height)
    final = draw_notes(image, templates[0], templates[1], templates[2])
    # detect note value and then write it on the image
    return final






result = detection("./test-images/music1-cropped.png")
result.show()

#final code: 
# if (len(sys.argv)>2):
#     print("Incorrect # of args")
# else: 
#     result = detection(sys.argv[1])
#     result.show()

