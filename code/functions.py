from PIL import Image
from PIL import ImageFilter
import numpy as np
# random number generator
import random

# for drawing text over image
from PIL import ImageDraw
import math


# 3. Implement a function that convolves a greyscale image I with an arbitrary two-dimensional kernel H.
# (You can use a brute-force implementation – no need to use fancy tricks or Fourier Transforms, although
# you can if you want.) Make sure your code handles image boundaries in some reasonable way.
def get_value_matrix(pImage, pPosition, pKernel):
    """
    takes an image, a position in the image and a kernel and returns a 2D array corresponding to the values 
    in the image centered at the postion with the size of the kernel

    EXAMPLE:

    [[1 , 2 , 3],
     [4 , 5 , 6],
     [7 , 8 , 9] ]

    Image, tuple, List of Lists -> List of Lists
    """
    lKernelHeight = len(pKernel)
    lKernelWidth = len(pKernel[0])
    offset_height = int((lKernelHeight - 1) / 2)
    offset_width = int((lKernelWidth - 1) / 2)
    output = []  # initialize a matrix of the size of the kernel

    # we fill the output matrix with the values found in the image at the given position
    for i in range(-offset_height, offset_height + 1):  # for a 3X3 : range(-1 , 2) -> [-1, 0, 1]
        row = []
        for j in range(-offset_width, offset_width + 1):
            row.append(pImage.getpixel((pPosition[0] + j, pPosition[1] + i)))  # (width, height)
        # now we have the full row
        output.append(row)  # add the row to the output matrix

    return output


# function to compute the value that one iteration of a convolution produces
# pValues = matrix of
def compute_conv_val(pValues, pKernel):
    val = 0
    for i in range(len(pKernel)):
        for j in range(len(pKernel)):
            val += pValues[i][j] * pKernel[i][j]
    return int(val)  # Colin: Changed this to int()


def convolve_brute_force(pI, pH):
    # output image
    offset_height = int((len(pH) - 1) / 2)
    offset_width = int((len(pH[0]) - 1) / 2)
    output_image = Image.new(mode="L", size=(pI.width - (2 * offset_width), pI.height - (2 * offset_height)))
    position = [-1, -1]  # width, height
    for i in range(offset_height, pI.height - offset_height):
        position[1] += 1
        for j in range(offset_width, pI.width - offset_width):
            position[0] += 1
            # print("position: ", str(position))
            # print("i, j :", i, j)
            # now we're inside the bounds of the new image
            val_matrix = get_value_matrix(pI, tuple(position), pH)
            res = compute_conv_val(val_matrix, pH)
            output_image.putpixel((position[0], position[1]), res)
        position[0] = -1
    return output_image


# img = Image.open(
#     'C:\\Users\\colin\\iCloudDrive\\Desktop\\Spring 2022\\B457\\githubRepo\\a1-group3\\sample_code\\first_photograph.png')
# # img = Image.open('C:\\Users\\tatec\\PycharmProjects\\a1-group3\\sample_code\\first_photograph.png')
# kernel = [[0, 0, 0],
#           [0, 1, 0],
#           [0, 0, 0]]
# newImage = convolve_brute_force(img, kernel)
# print("width, height: ", newImage.width, newImage.height)
# newImage.save("C:\\Users\\colin\\iCloudDrive\\Desktop\\Spring 2022\\B457\\githubRepo\\a1-group3\\part3Testing_1.png")


# newImage.save('newImage.png')

# HELPER FUNCTIONS ------------------------------------------------------------------------------

# Convlolve RGB
def get_value_matrix_RGB(pImage, pPosition, pKernel):
    """
    takes an image, a position in the image and a kernel and returns a 2D array corresponding to the values
    in the image centered at the postion with the size of the kernel

    This particular implementation will return values in the form of tuples (r, g, b)

    EXAMPLE:

    [[(1,2,3) , (1,2,3) , (1,2,3)],
     [(1,2,3) , (1,2,3) , (1,2,3)],
     [(1,2,3) , (1,2,3) , (1,2,3)] ]

    Image, tuple, List of Lists -> List of Lists
    """
    lKernelHeight = len(pKernel)
    lKernelWidth = len(pKernel[0])
    offset_height = int((lKernelHeight - 1) / 2)
    offset_width = int((lKernelWidth - 1) / 2)
    output = []  # initialize a matrix of the size of the kernel

    # we fill the output matrix with the values found in the image at the given position
    for i in range(-offset_height, offset_height + 1):  # for a 3X3 : range(-1 , 2) -> [-1, 0, 1]
        row = []
        for j in range(-offset_width, offset_width + 1):
            # width, height
            # print("I, J : ", i, j)
            # print("pPos : ", pPosition)
            try:
                r, g, b = pImage.getpixel((pPosition[0] + j, pPosition[1] + i))  # assign values to r, g, and b
            except:
                r, g, b, a = pImage.getpixel((pPosition[0] + j, pPosition[1] + i))  # assign values to r, g, and b
            row.append((r, g, b))
        # now we have the full row
        output.append(row)  # add the row to the output matrix

    return output


# Colin: Updating Tate's function to handle RBG:
def compute_conv_val_RBG(pValues, pKernel):
    """

    list of lists, list of lists -> tuple
    """
    val = (0, 0, 0)  # now our value will have 3 parts: R, G, B
    for i in range(len(pKernel)):
        for j in range(len(pKernel)):
            # print(val)
            R = pValues[i][j][0] * pKernel[i][j] + val[0]
            G = pValues[i][j][1] * pKernel[i][j] + val[1]
            B = pValues[i][j][2] * pKernel[i][j] + val[2]
            val = (int(R), int(G), int(B))
    return val


def convolve_brute_force_RGB(pI, pH):
    """

    Image, list of lists -> Image
    """
    # output image
    offset_height = int((len(pH) - 1) / 2)
    offset_width = int((len(pH[0]) - 1) / 2)
    # intialize the image with RGB mode
    output_image = Image.new(mode="RGB", size=(pI.width - (2 * offset_width), pI.height - (2 * offset_height)))
    position = [-1, -1]  # width, height
    for i in range(offset_height, pI.height - offset_height):
        position[1] += 1
        for j in range(offset_width, pI.width - offset_width):
            position[0] += 1
            # print("position: ", str(position))
            # print("i, j :", i, j)
            # now we're inside the bounds of the new image
            # print("Debug in convolve_brute_force_RGB -> position = ",position)
            val_matrix = get_value_matrix_RGB(pI, tuple(position), pH)
            res = compute_conv_val_RBG(val_matrix, pH)
            output_image.putpixel((position[0], position[1]), res)
        position[0] = -1
    return output_image


# img = Image.open('C:\\Users\\colin\\iCloudDrive\\Desktop\\Spring 2022\\B457\\githubRepo\\a1-group3\\sample_code\\first_photograph.png')
# # img = Image.open('C:\\Users\\tatec\\PycharmProjects\\a1-group3\\sample_code\\first_photograph.png')
# kernel = [[0, 0, 0],
#           [0, 1, 0],
#           [0, 0, 0]]
# newImage = convolve_brute_force(img, kernel)
# print("width, height: ", newImage.width, newImage.height)
# newImage.save("C:\\Users\\colin\\iCloudDrive\\Desktop\\Spring 2022\\B457\\githubRepo\\a1-group3\\part3Testing_1.png")
# # newImage.save('newImage.png')


# 4. Implement a function that convolves a greyscale image I with a separable kernel H. Recall that a
# separable kernel is one such that H = hTx hy, where hx and hy are both column vectors. Implement efficient
# convolution by using two convolution passes (with hx and then hy), as we discussed in class. Make sure
# your code handles image boundaries in some reasonable way.

# helper function that returns the Hx and Hy vectors that are multiplied to create the kernel
# it will return as a list in the form [Hx, Hy], where Hx and Hy are both lists of numbers.
# Hx is the row vector, and Hy is the column vector.


def convolve_grayscale(image, H):
    I = image
    # gets the value matrix for the image
    value_matrix = get_grayscale_matrix(image)
    # get the column vectors for the kernel
    hx, hy = get_column_vectors(H)
    # setting up arrays for convolving along x then y axis
    convolved_rows = np.zeros((I.width, I.height))
    convolved_cols = np.zeros((I.height, I.width))
    # convolve along x axis
    for x in range(len(value_matrix)):
        convolve = np.convolve(value_matrix[x], hx, "same")
        convolved_rows[x] = convolve

    # transpose so we can loop thru the columns
    transposed = convolved_rows.transpose()
    # convolve along y axis
    for y in range(transposed.shape[0]):
        convolve = np.convolve(transposed[y], hy, "same")
        convolved_cols[y] = convolve

    # transpose again to get cols and rows in the same position
    output = convolved_cols.transpose()

    output_image = Image.new(mode="L", size=(I.width, I.height))
    for x in range(output.shape[0]):
        for y in range(output.shape[1]):
            res = int(output[x][y])
            output_image.putpixel((x, y), (res))

    return output_image


# def test_gray_convolve(image = Image.open(r'sample_code/first_photograph.png'), kernel = [[1,2,1],[0,0,0], [-1,-2,-1]], test = [[]] ):
#     #if the output is not provided, we will assume that we are testing for a kernel that returns the original image
#     output = convolve_grayscale(image,kernel)
#     value_matrix = get_grayscale_matrix(image)

#     output.show()
#     output.save('code/grayscale_test.png')
#######################################################################################
# HELPER FUNCTIONS:#
#######################################################################################
def get_column_vectors(kernel):
    """
    helper function that returns the Hx and Hy vectors that are multiplied to create the kernel
    it will return as a list in the form [Hx, Hy], where Hx and Hy are both lists of numbers.

    Hx is the row vector, and Hy is the column vector.
    """
    U, E, V = np.linalg.svd(kernel)

    if E[1] >= 0.05 or E[1] <= -0.05:
        return

    # Prints used to verify correct output:
    # print(E)
    # print(U[:, 0])
    # print(V[0, :])
    Hx = V[0, :] * -np.sqrt(E[0])
    Hy = U[:, 0] * -np.sqrt(E[0])

    return Hx, Hy


# kernel1 = [[1, 2, 3],
#            [2, 4, 6],
#            [3, 6, 9]]
# x = get_vectors(kernel1)
# print(x)

def get_grayscale_matrix(image):
    # gets the grayscale value matrix of an image

    imageHeight = image.height
    imageWidth = image.width
    matrix = np.zeros((imageWidth, imageHeight))
    for x in range(imageWidth):
        for y in range(imageHeight):
            coords = x, y
            matrix[x][y] = image.getpixel(coords)
    return matrix


#######################################################################################
#  END P4 HELPER FUNCTIONS:#
#######################################################################################
# 5. A main goal of OMR is to locate various musical symbols in the image. Suppose for each of these
# symbols, we have a black and white “template” – a small m ×n-pixel image containing just that symbol –
# with black pixels indicating the symbol and white pixels indicating background. Call this template T. Now
# we can consider each m ×n-pixel region in the image of a sheet of music, compute a score for how well
# each region matches the template, and then mark the highest-scoring ones as being the symbol of interest.
# In other words, we want to define some similarity function fTI (i,j) that evaluates how similar the region
# around coordinates (i,j) in image I is to the template.
# One could define this function f(·) in many different ways. One simple way of doing this is to simply
# count the number of pixels that disagree between the image and the template – i.e. the Hamming distance
# between the two binary images, FUNCTION
# This function needs to be computed for each m ×n-pixel neighborhood of I. Fortunately, with a small
# amount of algebra, this can performed using a convolution operation!
# Implement a routine to detect a given template by doing the convolution above.


def compute_hamming_distance(value_matrix, template):
    """
    This helper function assumes that value_matrix and template will always be of the same size
    """
    result = 0
    kSum = 0
    for i in range(len(value_matrix)):
        for j in range(len(value_matrix[0])):
            for k in range(3):
                kSum += np.abs(
                    value_matrix[i][j][k] - template[i][j][k])  # assuming that template is a 2d array of tuples
            result += kSum
            kSum = 0

    return result


# this function will take an image and a template and draw a square around the places that
# the template is found in the image. Color is a tuple of the form (R, G, B)
def locate_symbol(image, template, color):
    dictionary = {}
    template = np.array(template, dtype='int64')
    img = np.array(image, dtype='int64')
    result = []
    threshold = 13005

    for i in range(len(img) - len(template)):
        for j in range(len(img[0]) - len(template[0])):
            # calculating hamming distance for each value
            dist = 0
            tooBig = False
            sub = str(img[i:i + len(template), j:j + len(template)])
            if sub in dictionary:
                dist = dictionary.get(sub)
            else:
                for x in range(len(template)):
                    for y in range(len(template[0])):
                        pixel = img[i + x][j + y]
                        dist += np.abs(pixel[0] - template[x][y][0])
                        dist += np.abs(pixel[1] - template[x][y][1])
                        dist += np.abs(pixel[2] - template[x][y][2])

                        if dist > 15300:
                            tooBig = True
                            break
                        if tooBig:
                            break
                    if tooBig:
                        break
                dictionary[sub] = dist
            if dist <= threshold:
                result.append([j, i])
                # print("Found match at position", j, i)

    # draw = ImageDraw.Draw(image)
    # for i in range(len(result)):
    #     point = result[i]
    #
    #     upperLeft = (point[0], point[1])
    #     lowerRight = (point[0] + len(template[0]), point[1] + len(template))
    #     draw.rectangle((upperLeft, lowerRight), outline=color)
    # name = "symbol_located_" + str(threshold) + ".png"

    return result


# temp = Image.open("C:\\Users\\tatec\\PycharmProjects\\a1-group3\\Template\\template1.png", "r")
# img = Image.open("C:\\Users\\tatec\\PycharmProjects\\a1-group3\\test-images\\music1.png", "r")


# print(locate_symbol(img, temp))

# 6. An alternative approach is to define the template matching scoring function using edge maps, which
# tend to be less sensitive to background clutter and more forgiving of small variations in symbol appearance.
# To do this, first run an edge detector on the template and the input image. You can use the Sobel operator
# and your separable convolution routine above to do this. Then, implement a version of template matching
# that uses the following scoring function:
# FUNCTION
# where I and T here are assumed to be edge maps, having value 1 if a pixel is an edge and 0 otherwise, and
# γ(·) is a function that is 0 when its parameter is non-zero and is infinite when its parameter is 0.
# Note that computing this scoring function for every pixel in the image can be quite slow if implemented
# naively. Each of the min’s involves a nested loop, each summation involves a nested loop, so computing the
# score for every pixel (i, j) requires a sextuply-nested loop! However, we can once again use a simple instance
# of dynamic programming to speed up this calculation. Notice that the above equation can be re-written as
# FUNCTION
# and M × N are the dimensions of I. Notice that D(i, j) has an intuitive meaning: for every pixel (i, j), it
# tells you the distance (in pixels) to the closest edge pixel in I. More importantly, notice that re-writing the
# equations in this way reduces the number of nested loops needed to compute f I
# T from six to four, because
# D can be pre-computed. Computing D for all pixels requires four nested loops if implemented naively, but
# requires only quadratic time if you’re clever (not required for this assignment).


# SOBEL OPERATORS
Sx = [[-.125, 0, .125],
      [-.25, 0, .25],
      [-.125, 0, .125]]
Sy = [[.125, .25, .125],
      [0, 0, 0],
      [-.125, -.25, -.125]]
meanFilterKernel = [[1 / 9, 1 / 9, 1 / 9],
                    [1 / 9, 1 / 9, 1 / 9],
                    [1 / 9, 1 / 9, 1 / 9]]


# def sobel(image):
#     values = get_grayscale_matrix(image)
#     result = convolve_brute_force_RGB(image, Sy)
#     result = convolve_brute_force_RGB(result, Sx)
#     return result




# Scoring Function -----------------------------

# IMPORTANT PART 6 GLOBAL VARIABLES ----------------------------------------

# Scoring helper function
if (False):  # Colin's Paths
    part6Image = Image.open(
        'C:\\Users\\colin\\iCloudDrive\\Desktop\\Spring 2022\\B457\\githubRepo\\a1-group3\\test-images\\music1.png')
    part6Template = Image.open(
        'C:\\Users\\colin\\iCloudDrive\\Desktop\\Spring 2022\\B457\\githubRepo\\a1-group3\\test-images\\template1.png')
elif (False):  # Tate Path
    part6Image = Image.open('C:\\Users\\tatec\\PycharmProjects\\a1-group3\\test-images\\music1.png')
    part6Template = Image.open('C:\\Users\\tatec\\PycharmProjects\\a1-group3\\test-images\\template1.png')
elif (True):  # Jack Path
    part6Image = Image.open('./test-images/music1.png')
    part6Template = Image.open('./test-images/template1.png')
else:  # Matthew Path
    part6Image = Image.open('C:\\src\\a1-group3\\test-images\\music1.png')
    part6Template = Image.open('C:\\src\\a1-group3\\test-images\\template1.png')

# memoDictPart6 = dict()  # key= (i, j) : value= distance to nearest edge pixel

# Run edge detector on the template and the input image

# Convert to grayscale:
# part6Image = part6Image.convert("L")
# part6Template = part6Template.convert("L")

# p6ImgEDGE = convolve_grayscale(part6Image, Sx)
# p6TmpEDGE = convolve_grayscale(part6Template, Sx)


# p6ImgEDGE = convolve_brute_force_RGB(part6Image, Sx)
# p6TmpEDGE = convolve_brute_force_RGB(part6Template, Sx)


# End global variables -------------------------------------------------------
def dist(pPos1, pPos2):
    """
    A simple euclidian distance function

    pPos1    tuple   (x, y)
    pPos2    tuple   (x, y)

    tuple, tuple -> number
    """
    return math.sqrt((pPos1[0] - pPos2[0]) ** 2 + (pPos1[1] - pPos2[1]) ** 2)


def dFuncPreCompute(pImage):
    """
    This function performs the pre-computing for the dFunc

    Image -> None
    """
    global memoDictPart6
    whitePixelThreshold = 40  # what is the smallest value we consider a white pixel (an edge)

    lWidth = pImage.width
    lHeight = pImage.height
    distance = (math.inf, (0, 0))  # (minimized value,  (position width, height))
    for i in range(lWidth):
        for j in range(lHeight):
            # for each position, we grab the distance to the nearest edge pixel

            # Naive implementation:
            for k in range(lWidth):
                for l in range(lHeight):
                    # print(pImage.getpixel((k,l)))
                    if (pImage.getpixel(
                            (k, l)) > whitePixelThreshold):  # if (k,l) corresponds to an edge pixel we log the distance
                        lDist = dist((i, j), (k, l))
                        if (lDist > distance[0]):
                            distance = (lDist, (k, l))

            # we have checked all 
            memoDictPart6[(i, j)] = distance[0]
            distance = (math.inf, (0, 0))  # (minimized value,  (position width, height))


# dFuncPreCompute(p6ImgEDGE)  # do the computation for dFunc


def dFunc(i, j):
    """
    for every pixel i, j returns the distance in pixels to the closest
    edge pixel in I

    """  # TODO: Implement this function dynamically: can be pre-computed
    global memoDictPart6
    return memoDictPart6[(i, j)]


def tFunc(i, j):
    """
    returns a value 1 if the pixel is an edge, 0 otherwise

    number, number -> number
    """
    global part6ImgEDGE
    whitePixelThreshold = 40  # what is the smallest value we consider a white pixel (an edge)
    if (part6ImgEDGE.getpixel((i, j)) > whitePixelThreshold):
        return 1
    else:
        return 0


def scoreFunc(i, j):
    """

    T       Edge map
    dFunc   As is above

    number, number -> number
    """
    m, n = (0, 0)  # TODO: implement n and m (the dimensions of the image)
    score = 0  # intialize the value of the score
    for k in range(0, m):  # range from 0 -> m-1
        for l in range(0, n):  # range from 0 -> n-1
            score += tFunc(k, l) * dFunc(i + k, j + l)  # the score is a summation
    return score


# ----- End Scoring Function ------------------------
# test_gray_convolve()


# template = Image.open('C:\\Users\\colin\\iCloudDrive\\Desktop\\Spring 2022\\B457\\githubRepo\\a1-group3\\test-images\\template1.png')
# newTemplate = convolve_brute_force_RGB(template, Sy)
# newTemplate = convolve_brute_force_RGB(newTemplate, Sx)
# newTemplate.show()
# newTemplate = template.save("C:\\Users\\colin\\iCloudDrive\\Desktop\\Spring 2022\\B457\\githubRepo\\a1-group3\\part6SobelXTemplate.png")


# Testing code for part 6
# image = Image.open(
#     'C:\\Users\\colin\\iCloudDrive\\Desktop\\Spring 2022\\B457\\githubRepo\\a1-group3\\test-images\\music1.png')
# # apply gaussian (by approximation: use sobel operators)\
# image.save("C:\\Users\\colin\\iCloudDrive\\Desktop\\Spring 2022\\B457\\githubRepo\\a1-group3\\part6Testing_1b.png")
# newImage = convolve_brute_force_RGB(image, meanFilterKernel)
# print("Success!")
# newImage.save("C:\\Users\\colin\\iCloudDrive\\Desktop\\Spring 2022\\B457\\githubRepo\\a1-group3\\part6Testing_2b.png")

# newImage2 = convolve_brute_force_RGB(image, Sx)
# newImage2 = convolve_brute_force_RGB(newImage2, meanFilterKernel)

# newImage2 = convolve_brute_force_RGB(newImage2, Sy)
# newImage2 = convolve_brute_force_RGB(newImage2, meanFilterKernel)

# newImage2.save("C:\\Users\\colin\\iCloudDrive\\Desktop\\Spring 2022\\B457\\githubRepo\\a1-group3\\part6Testing_3b.png")
# kernel = [[0, 0, 0],
#           [0, 50, 0],
#           [0, 0, 0]]
# newImage3 = convolve_brute_force_RGB(newImage2, kernel)
# newImage3.save("C:\\Users\\colin\\iCloudDrive\\Desktop\\Spring 2022\\B457\\githubRepo\\a1-group3\\part6Testing_4b.png")


# tmp = Image.open('C:\\Users\\colin\\iCloudDrive\\Desktop\\Spring 2022\\B457\\githubRepo\\a1-group3\\test-images\\template1.png')
# # apply gaussian (by approximation: use sobel operators)\
# tmp.convert("RGB")
# tmp.save("C:\\Users\\colin\\iCloudDrive\\Desktop\\Spring 2022\\B457\\githubRepo\\a1-group3\\part6Testing_1c.png")
# newTmp = convolve_brute_force_RGB(tmp, meanFilterKernel)
# print("Success!")
# newTmp.save("C:\\Users\\colin\\iCloudDrive\\Desktop\\Spring 2022\\B457\\githubRepo\\a1-group3\\part6Testing_2c.png")

# newTmp2 = convolve_brute_force_RGB(tmp, Sx)
# #newTmp2 = convolve_brute_force_RGB(newTmp2, meanFilterKernel)

# newTmp2.save("C:\\Users\\colin\\iCloudDrive\\Desktop\\Spring 2022\\B457\\githubRepo\\a1-group3\\part6Testing_2.5c.png")

# newTmp2 = convolve_brute_force_RGB(newTmp2, Sy)
# #newTmp2 = convolve_brute_force_RGB(newTmp2, meanFilterKernel)

# newTmp2.save("C:\\Users\\colin\\iCloudDrive\\Desktop\\Spring 2022\\B457\\githubRepo\\a1-group3\\part6Testing_3c.png")
# kernel = [[0, 0, 0],
#           [0, 50, 0],
#           [0, 0, 0]]
# newTmp2 = convolve_brute_force_RGB(newTmp2, kernel)
# newTmp2.save("C:\\Users\\colin\\iCloudDrive\\Desktop\\Spring 2022\\B457\\githubRepo\\a1-group3\\part6Testing_4c.png")


# 7
# The sample image and template we described above were carefully designed so that the size of the
# template exactly matched the size of the objects appearing in the image. In practice, we won’t know the
# scale ahead of time and we’ll have to infer it from an image. Fortunately, if we can find the staff lines, then
# we can estimate the note head size, since the distance between staff lines is approximately the height of a
# note head. To find the staves, one could first find horizontal lines using Hough transforms and then try to
# find groups of five equally-spaced lines, but this two-step approach introduces the possibility of failure: if a
# line is not detected properly, an entire staff might not be found. A better approach is to apply the Hough
# transform to find the groups of five lines directly. Implement a Hough transform to do this. Assume that
# the lines in the staves are perfectly horizontal, perfectly parallel, and evenly spaced (but we do not know
# the spacing ahead of time). Then the Hough voting space has two dimensions: the row-coordinate of the
# first line of the staff, and the spacing distance between the staff lines. Each pixel in the image then “votes”
# for a set of row-coordinates and spacing parameters. Each peak in this voting space then corresponds to
# the row-coordinate and spacing of a staff line, which in turn tells us where each of the five lines of the staff
# is located.

def hough_transform(image):

    # Run edge detection on input image
    newImage2 = convolve_brute_force_RGB(image, meanFilterKernel)
    newImage2 = convolve_brute_force_RGB(newImage2, Sy)

    kernel = [[0, 0, 0],
              [0, 50, 0],
              [0, 0, 0]]
          
    image = convolve_brute_force_RGB(newImage2, kernel)

    x_max = image.height
    y_max = image.width

    space_min = 1
    space_max = 20

    hough_space = np.zeros((x_max, space_max))

    # Each pixel will update hough space for each set of coordinates and line spacing values that
    # would create a line on the current one
    for x in range(x_max):
        for y in range(y_max):
            coords = y, x
            if image.getpixel(coords)[0] == 0: continue
            for ispace in range(space_max - space_min):
                ispace += 1
                for row in range(x - 1):
                    for line in range(4):
                        line += 1
                        if (row + (ispace * line)) == x:
                            hough_space[row, ispace] += 1
                if (x + (ispace * 4)) < x_max:
                    hough_space[x, ispace] += 1

    # first local max
    max = 0
    max_row = 0
    max_spacing = 0

    for x in range(x_max):
        for y in range(space_max):
            if hough_space[x, y] > max:
                max = hough_space[x, y]
                max_row = x
                max_spacing = y

    # second local max
    second_max = 0
    second_row = 0
    second_spacing = 0

    for x in range(x_max):
        for y in range(space_max):
            if hough_space[x, y] > second_max:
                if (abs(max_row - x) > 10):
                    second_max = hough_space[x, y]
                    second_row = x
                    second_spacing = y

    # row_coordinates = []

    # # add row coordinate values to array
    # for x in range(2):
    #     if x == 0: 
    #         row = max_row
    #         spaceing = max_spacing
    #     else: 
    #         row = second_row
    #         spaceing = second_spacing
    #     for line in range(5):
    #         row_coordinates.append(row + (line * spaceing))

    # Create tuple for storing row values as another tuple and the spacing
    staves = (max_row, second_row), max_spacing

    return staves

# image = Image.open("C:\\src\\a1-group3\\test-images\\music1.png")
# print(hough_transform(image))
