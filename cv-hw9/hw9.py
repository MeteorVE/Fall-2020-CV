import numpy as np
import matplotlib.pyplot as plt
import cv2

import math

"""
CV-2020 Hw9 General Edge Detection
Author: MeteorV

You are to implement following edge detectors with thresholds :
(a) Robert's Operator: 12
(b) Prewitt's Edge Detector: 24
(c) Sobel's Edge Detector: 38
(d) Frei and Chen's Gradient Operator: 30
(e) Kirsch's Compass Operator: 135
(f) Robinson's Compass Operator: 43
(g) Nevatia-Babu 5x5 Operator: 12500

You can use any programing language to implement homework, however, you'll get zero point if you just call existing library.
Threshold Values listed above are for reference, you can choose your own values if you want.
"""


def extend_padding(img, ext_size):
    h, w = img.shape
    new_img = np.full( (w+ext_size*2, h+ext_size*2), 0, dtype=np.uint8 )
    new_img[ ext_size:w+ext_size, ext_size:h+ext_size ] = img

    new_img[ 0: ext_size, 0:ext_size] = np.full((ext_size, ext_size), img[0][0], dtype=np.uint8)
    new_img[ 0: ext_size, h+ext_size:h+ext_size*2] = np.full((ext_size, ext_size), img[0][h-1], dtype=np.uint8)
    new_img[ w+ext_size:w+ext_size*2, 0: ext_size] = np.full((ext_size, ext_size), img[w-1][0], dtype=np.uint8)
    new_img[ w+ext_size:w+ext_size*2, h+ext_size:h+ext_size*2] = np.full((ext_size, ext_size), img[w-1][h-1], dtype=np.uint8)

    for i in range(ext_size):
        new_img[ ext_size:ext_size+h, i ] = img[ 0:h, 0]
        new_img[ ext_size:ext_size+h, i+w+ext_size ] = img[ 0:h, w-1]
        new_img[ i, ext_size:ext_size+w ] = img[ 0, 0:w]
        new_img[ i+h+ext_size, ext_size:ext_size+w ] = img[ w-1, 0:w]

    return new_img

def get_roberts_operator(img, threshold):
    """
    :type img: Image(numpy 2d)
    :type threshold: int
    :return type: Image 
    """

    h, w = img.shape
    new_img = img.copy()

    for x in range( h ):
        for y in range( w ):
            x1, y1 = x+1, y+1            
            if x1 >= w:
                x1 = w-1
            if y1 >= h:
                y1 = h-1

            r1 = -int(img[x][y])+img[x1][y1]
            r2 = -int(img[x1][y])+img[x][y1]
            magitude = int(math.sqrt(r1**2 + r2**2))

            if magitude >= threshold :
                new_img[x][y] = 0
            else:
                new_img[x][y] = 255

    return new_img

def get_prewitt_operator(img, threshold):
    """
    :type img: Image(numpy 2d)
    :type threshold: int
    :return type: Image 
    """

    h, w = img.shape
    new_img = img.copy()
    img = extend_padding(img, 1)

    for x in range( 1,h+1 ):
        for y in range( 1,w+1 ):

            # x1 
            #    x
            #      x2
            x1, y1 = x-1, y-1
            x2, y2 = x+1, y+1

            s1 = -int(img[x1][y1]) - 1*int(img[x1][y]) -int(img[x1][y2]) + int(img[x2][y1]) + 1*int(img[x2][y]) + int(img[x2][y2])
            s2 = -int(img[x1][y1]) - 1*int(img[x][y1]) -int(img[x2][y1]) + int(img[x1][y2]) + 1*int(img[x][y2]) + int(img[x2][y2])
            magitude = int(math.sqrt(s1**2 + s2**2))

            if magitude >= threshold :
                new_img[x-1][y-1] = 0
            else:
                new_img[x-1][y-1] = 255

    return new_img

def get_sobel_operator(img, threshold):
    """
    :type img: Image(numpy 2d)
    :type threshold: int
    :return type: Image 
    """

    h, w = img.shape
    new_img = img.copy()
    img = extend_padding(img, 1)

    for x in range( 1,h+1 ):
        for y in range( 1,w+1 ):

            # x1 
            #    x
            #      x2
            x1, y1 = x-1, y-1
            x2, y2 = x+1, y+1

            s1 = -int(img[x1][y1]) - 2*int(img[x1][y]) -int(img[x1][y2]) + int(img[x2][y1]) + 2*int(img[x2][y]) + int(img[x2][y2])
            s2 = -int(img[x1][y1]) - 2*int(img[x][y1]) -int(img[x2][y1]) + int(img[x1][y2]) + 2*int(img[x][y2]) + int(img[x2][y2])
            magitude = int(math.sqrt(s1**2 + s2**2))

            if magitude >= threshold :
                new_img[x-1][y-1] = 0
            else:
                new_img[x-1][y-1] = 255

    return new_img

def get_frei_chen_operator(img, threshold):
    """
    :type img: Image(numpy 2d)
    :type threshold: int
    :return type: Image 
    """

    h, w = img.shape
    new_img = img.copy()
    img = extend_padding(img, 1)

    for x in range( 1,h+1 ):
        for y in range( 1,w+1 ):

            # x1 
            #    x
            #      x2
            x1, y1 = x-1, y-1
            x2, y2 = x+1, y+1

            s1 = -int(img[x1][y1]) - math.sqrt(2)*int(img[x1][y]) -int(img[x1][y2]) + int(img[x2][y1]) + math.sqrt(2)*int(img[x2][y]) + int(img[x2][y2])
            s2 = -int(img[x1][y1]) - math.sqrt(2)*int(img[x][y1]) -int(img[x2][y1]) + int(img[x1][y2]) + math.sqrt(2)*int(img[x][y2]) + int(img[x2][y2])
            magitude = int(math.sqrt(s1**2 + s2**2))

            if magitude >= threshold :
                new_img[x-1][y-1] = 0
            else:
                new_img[x-1][y-1] = 255

    return new_img

def get_kirsch_operator(img, threshold):
    """
    :type img: Image(numpy 2d)
    :type threshold: int
    :return type: Image 
    """

    h, w = img.shape
    new_img = img.copy()
    img = extend_padding(img, 1)

    for x in range( 1,h+1 ):
        for y in range( 1,w+1 ):

            # x1               x1y1   x1y   x1y2
            #    x             x y1   x y   x y2
            #      x2          x2y1   x2y   x2y2 
            x1, y1 = x-1, y-1
            x2, y2 = x+1, y+1

            coordinate = np.array( [int(img[x1][y1]), int(img[x1][y]), int(img[x1][y2]), int(img[x][y1]), int(img[x][y2]), int(img[x2][y1]), int(img[x2][y]), int(img[x2][y2])] )
            k0 = np.dot(np.array( [-3, -3, 5, -3, 5, -3, -3, 5] ), coordinate)
            k1 = np.dot(np.array( [-3, 5, 5, -3, 5, -3, -3, -3] ), coordinate)
            k2 = np.dot(np.array( [5, 5, 5, -3, -3, -3, -3, -3] ), coordinate)
            k3 = np.dot(np.array( [5, 5, -3, 5, -3, -3, -3, -3] ), coordinate)
            k4 = np.dot(np.array( [5, -3, -3, 5, -3, 5, -3, -3] ), coordinate)
            k5 = np.dot(np.array( [-3, -3, -3, 5, -3, 5, 5, -3] ), coordinate)
            k6 = np.dot(np.array( [-3, -3, -3, -3, -3, 5, 5, 5] ), coordinate)
            k7 = np.dot(np.array( [-3, -3, -3, -3, 5, -3, 5, 5] ), coordinate)

            magitude = max(k0, k1, k2, k3, k4, k5, k6, k7)
            
            if magitude >= threshold :
                new_img[x-1][y-1] = 0
            else:
                new_img[x-1][y-1] = 255

    return new_img

def get_robinson_operator(img, threshold):
    """
    :type img: Image(numpy 2d)
    :type threshold: int
    :return type: Image 
    """

    h, w = img.shape
    new_img = img.copy()
    img = extend_padding(img, 1)

    for x in range( 1,h+1 ):
        for y in range( 1,w+1 ):

            # x1               x1y1   x1y   x1y2
            #    x             x y1   x y   x y2
            #      x2          x2y1   x2y   x2y2 
            x1, y1 = x-1, y-1
            x2, y2 = x+1, y+1

            coordinate = np.array( [int(img[x1][y1]), int(img[x1][y]), int(img[x1][y2]), int(img[x][y1]), int(img[x][y2]), int(img[x2][y1]), int(img[x2][y]), int(img[x2][y2])] )
            k0 = np.dot(np.array( [-1, 0, 1, -2, 2, -1, 0, 1] ), coordinate)
            k1 = np.dot(np.array( [0, 1, 2, -1, 1, -2, -1, 0] ), coordinate)
            k2 = np.dot(np.array( [1, 2, 1, 0, 0, -1, -2, -1] ), coordinate)
            k3 = np.dot(np.array( [2, 1, 0, 1, -1, 0, -1, -2] ), coordinate)
            k4 = np.dot(np.array( [1, 0, -1, 2, -2, 1, 0, -1] ), coordinate)
            k5 = np.dot(np.array( [0, -1, -2, 1, -1, 2, 1, 0] ), coordinate)
            k6 = np.dot(np.array( [-1, -2, -1, 0, 0, 1, 2, 1] ), coordinate)
            k7 = np.dot(np.array( [-2, -1, 0, -1, 1, 0, 1, 2] ), coordinate)

            magitude = max(k0, k1, k2, k3, k4, k5, k6, k7)

            if magitude >= threshold :
                new_img[x-1][y-1] = 0
            else:
                new_img[x-1][y-1] = 255

    return new_img

def get_nevatia_babu_operator(img, threshold):
    """
    :type img: Image(numpy 2d)
    :type threshold: int
    :return type: Image 
    """

    h, w = img.shape
    new_img = img.copy()
    img = extend_padding(img, 2)

    for x in range( 2,h+2 ):
        for y in range( 2,w+2 ):

            # x1                 x1y1   x1y2   x1y   x1y3   x1y4
            #    x2              x2y1   x2y2   x2y   x2y3   x2y4
            #       x            x y1   x y2   x y   x y3   x y4
            #         x3         x3y1   x3y2   x3y   x3y3   x3y4
            #            x4      x4y1   x4y2   x4y   x4y3   x4y4

            x1, y1 = x-2, y-2
            x2, y2 = x-1, y-1
            x3, y3 = x+1, y+1
            x4, y4 = x+2, y+2

            coordinate = np.array( [int(img[x1][y1]), int(img[x1][y2]), int(img[x1][y]), int(img[x1][y3]), int(img[x1][y4]), int(img[x2][y1]), int(img[x2][y2]), int(img[x2][y]), int(img[x2][y3]), int(img[x2][y4]),  int(img[x][y1]), int(img[x][y2]), int(img[x][y]), int(img[x][y3]), int(img[x][y4]), int(img[x3][y1]), int(img[x3][y2]), int(img[x3][y]), int(img[x3][y3]), int(img[x3][y4]), int(img[x4][y1]), int(img[x4][y2]), int(img[x4][y]), int(img[x4][y3]), int(img[x4][y4])  ])
            n0 = np.dot(np.array( [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 0, 0, 0, 0, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100] ), coordinate)
            n1 = np.dot(np.array( [100, 100, 100, 100, 100, 100, 100, 100, 78, -32, 100, 92, 0, -92, -100, 32, -78, -100, -100, -100, -100, -100, -100, -100, -100] ), coordinate)
            n2 = np.dot(np.array( [100, 100, 100, 32, -100, 100, 100, 92, -78, -100, 100, 100, 0, -100, -100, 100, 78, -92, -100, -100, 100, -32, -100, -100, -100] ), coordinate)
            n3 = np.dot(np.array( [-100, -100, 0, 100, 100, -100, -100, 0, 100, 100, -100, -100, 0, 100, 100, -100, -100, 0, 100, 100, -100, -100, 0, 100, 100] ), coordinate)
            n4 = np.dot(np.array( [-100, 32, 100, 100, 100, -100, -78, 92, 100, 100, -100, -100, 0, 100, 100, -100, -100, -92, 78, 100, -100, -100, -100, -32, 100] ), coordinate)
            n5 = np.dot(np.array( [100, 100, 100, 100, 100, -32, 78, 100, 100, 100, -100, -92, 0, 92, 100, -100, -100, -100, -78, 32, -100, -100, -100, -100, -100] ), coordinate)
            

            magitude = max(n0, n1, n2, n3, n4, n5)

            if magitude >= threshold :
                new_img[x-2][y-2] = 0
            else:
                new_img[x-2][y-2] = 255

    return new_img

def main():
    img = cv2.imread('lena.bmp',0)

    cv2.imwrite("1-roberts.bmp", get_roberts_operator(img, 12))
    cv2.imwrite("2-prewitt.bmp", get_prewitt_operator(img, 24))
    cv2.imwrite("3-sobel.bmp", get_sobel_operator(img, 38))
    cv2.imwrite("4-frei.bmp", get_frei_chen_operator(img, 30))
    cv2.imwrite("5-kirsch.bmp", get_kirsch_operator(img, 135))
    cv2.imwrite("6-robinson.bmp", get_robinson_operator(img, 43))
    cv2.imwrite("7-nevatia_babu.bmp", get_nevatia_babu_operator(img, 12500))

    # === TEST AREA === #
    #img = np.array([ [1,2,3],[4,5,6],[7,8,9] ])
    #img = np.array([ [169,146,153,145,137,151,112,98],[104,104,97,100,115,40,42,63],[130,120,95,120,130,212,115,128],[124,157,162,45,87,77,75,101],[124,201,177,176,136,113,150,137],[162,155,193,46,52,87,126,203],[141,149,38,54,155,145,132,57],[87,64,156,161,180,210,99,79]])



if __name__ == '__main__':
    main()

