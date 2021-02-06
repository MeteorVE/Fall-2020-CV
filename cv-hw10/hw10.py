import numpy as np
import matplotlib.pyplot as plt
import cv2

import math

"""
CV-2020 Hw10 Zero Crossing Edge Detection
Author: MeteorV

Implement 2 Laplacian Mask, Minimum Variance Laplacian, Laplacian of Gaussian, and Difference of Gaussian(inhibitory sigma=3, excitatory sigma=1, kernel size 11x11).
Please list the kernels and the thresholds(for zero crossing) you used.
Threshold Values listed below are for reference:
(僅供參考，同學可自己找出 Edge Image 品質最佳的門檻值threshold value)
(a) Laplace Mask1 (0, 1, 0, 1, -4, 1, 0, 1, 0): 15
(b) Laplace Mask2 (1, 1, 1, 1, -8, 1, 1, 1, 1)
(c) Minimum variance Laplacian: 20
(d) Laplace of Gaussian: 3000
(e) Difference of Gaussian: 1

- You can use any programing language to implement homework, however, you'll get zero point if you just call existing library.
"""


def extend_padding(img, ext_size):
    h, w = img.shape
    new_img = np.full( (w+ext_size*2, h+ext_size*2), 0, dtype=np.int16 )
    new_img[ ext_size:w+ext_size, ext_size:h+ext_size ] = img

    new_img[ 0: ext_size, 0:ext_size] = np.full((ext_size, ext_size), img[0][0], dtype=np.int16)
    new_img[ 0: ext_size, h+ext_size:h+ext_size*2] = np.full((ext_size, ext_size), img[0][h-1], dtype=np.int16)
    new_img[ w+ext_size:w+ext_size*2, 0: ext_size] = np.full((ext_size, ext_size), img[w-1][0], dtype=np.int16)
    new_img[ w+ext_size:w+ext_size*2, h+ext_size:h+ext_size*2] = np.full((ext_size, ext_size), img[w-1][h-1], dtype=np.int16)

    for i in range(ext_size):
        new_img[ ext_size:ext_size+h, i ] = img[ 0:h, 0]
        new_img[ ext_size:ext_size+h, i+w+ext_size ] = img[ 0:h, w-1]
        new_img[ i, ext_size:ext_size+w ] = img[ 0, 0:w]
        new_img[ i+h+ext_size, ext_size:ext_size+w ] = img[ w-1, 0:w]

    return new_img


def get_laplacian1_operator(img, threshold):
    """
    :type img: Image(numpy 2d)
    :type threshold: int
    :return type: Image (0, 1) 
    """

    h, w = img.shape
    new_img = img.copy().astype('int16') 
    img = extend_padding(img, 1)

    for x in range( 1,h+1 ):
        for y in range( 1,w+1 ):

            # x1               x1y1   x1y   x1y2
            #    x             x y1   x y   x y2
            #      x2          x2y1   x2y   x2y2 
            x1, y1 = x-1, y-1
            x2, y2 = x+1, y+1

            coordinate = np.array( [int(img[x1][y1]), int(img[x1][y]), int(img[x1][y2]), int(img[x][y1]), int(img[x][y]), int(img[x][y2]), int(img[x2][y1]), int(img[x2][y]), int(img[x2][y2])] )
            magitude = np.dot(np.array( [0, 1, 0, 1, -4, 1, 0, 1, 0] ), coordinate)

            if magitude >= threshold :
                new_img[x-1][y-1] = 1
            elif magitude <= threshold*(-1):
                new_img[x-1][y-1] = -1
            else:
                new_img[x-1][y-1] = 0

    return new_img

def get_laplacian2_operator(img, threshold):
    """
    :type img: Image(numpy 2d)
    :type threshold: int
    :return type: Image (0, 1) 
    """

    h, w = img.shape
    new_img = img.copy().astype('int16') 
    img = extend_padding(img, 1)

    for x in range( 1,h+1 ):
        for y in range( 1,w+1 ):

            # x1               x1y1   x1y   x1y2
            #    x             x y1   x y   x y2
            #      x2          x2y1   x2y   x2y2 
            x1, y1 = x-1, y-1
            x2, y2 = x+1, y+1

            coordinate = np.array( [int(img[x1][y1]), int(img[x1][y]), int(img[x1][y2]), int(img[x][y1]), int(img[x][y]), int(img[x][y2]), int(img[x2][y1]), int(img[x2][y]), int(img[x2][y2])] )
            magitude = np.dot(np.array( [1, 1, 1, 1, -8, 1, 1, 1, 1] ), coordinate)*(1/3)
            
            if magitude >= threshold :
                new_img[x-1][y-1] = 1
            elif magitude <= threshold*(-1):
                new_img[x-1][y-1] = -1
            else:
                new_img[x-1][y-1] = 0

    return new_img

def get_minvar_laplacian_operator(img, threshold):
    """
    :type img: Image(numpy 2d)
    :type threshold: int
    :return type: Image (0, 1) 
    """

    h, w = img.shape
    new_img = img.copy().astype('int16') 
    img = extend_padding(img, 1)

    for x in range( 1,h+1 ):
        for y in range( 1,w+1 ):

            # x1               x1y1   x1y   x1y2
            #    x             x y1   x y   x y2
            #      x2          x2y1   x2y   x2y2 
            x1, y1 = x-1, y-1
            x2, y2 = x+1, y+1

            coordinate = np.array( [int(img[x1][y1]), int(img[x1][y]), int(img[x1][y2]), int(img[x][y1]), int(img[x][y]), int(img[x][y2]), int(img[x2][y1]), int(img[x2][y]), int(img[x2][y2])] )
            magitude = np.dot(np.array( [2, -1, 2, -1, -4, -1, 2, -1, 2] ), coordinate)*(1/3)
            
            if magitude >= threshold :
                new_img[x-1][y-1] = 1
            elif magitude <= threshold*(-1):
                new_img[x-1][y-1] = -1
            else:
                new_img[x-1][y-1] = 0

    return new_img

def get_gauss_laplacian_operator(img, threshold):
    """
    :type img: Image(numpy 2d)
    :type threshold: int
    :return type: Image (0, 1) 
    """

    h, w = img.shape
    new_img = img.copy().astype('int16') 
    img = extend_padding(img, 5)

    for x in range( 5,h+5 ):
        for y in range( 5,w+5 ):

            # x1               x1y1   x1y   x1y2
            #    x             x y1   x y   x y2
            #      x2          x2y1   x2y   x2y2 
            x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 = x-5, y-5, x-4, y-4, x-3, y-3, x-2, y-2, x-1, y-1
            x6, y6, x7, y7, x8, y8, x9, y9, x10, y10 = x+1, y+1, x+2, y+2, x+3, y+3, x+4, y+4, x+5, y+5

            coordinate = np.array( [int(img[x1][y1]), int(img[x1][y2]), int(img[x1][y3]), int(img[x1][y4]), int(img[x1][y5]), int(img[x1][y]), int(img[x1][y6]), int(img[x1][y7]), int(img[x1][y8]), int(img[x1][y9]), int(img[x1][y10]),\
                                    int(img[x2][y1]), int(img[x2][y2]), int(img[x2][y3]), int(img[x2][y4]), int(img[x2][y5]), int(img[x2][y]), int(img[x2][y6]), int(img[x2][y7]), int(img[x2][y8]), int(img[x2][y9]), int(img[x2][y10]),\
                                    int(img[x3][y1]), int(img[x3][y2]), int(img[x3][y3]), int(img[x3][y4]), int(img[x3][y5]), int(img[x3][y]), int(img[x3][y6]), int(img[x3][y7]), int(img[x3][y8]), int(img[x3][y9]), int(img[x3][y10]),\
                                    int(img[x4][y1]), int(img[x4][y2]), int(img[x4][y3]), int(img[x4][y4]), int(img[x4][y5]), int(img[x4][y]), int(img[x4][y6]), int(img[x4][y7]), int(img[x4][y8]), int(img[x4][y9]), int(img[x4][y10]),\
                                    int(img[x5][y1]), int(img[x5][y2]), int(img[x5][y3]), int(img[x5][y4]), int(img[x5][y5]), int(img[x5][y]), int(img[x5][y6]), int(img[x5][y7]), int(img[x5][y8]), int(img[x5][y9]), int(img[x5][y10]),\
                                    int(img[x ][y1]), int(img[x ][y2]), int(img[x ][y3]), int(img[x ][y4]), int(img[x ][y5]), int(img[x ][y]), int(img[x ][y6]), int(img[x ][y7]), int(img[x ][y8]), int(img[x ][y9]), int(img[x ][y10]),\
                                    int(img[x6][y1]), int(img[x6][y2]), int(img[x6][y3]), int(img[x6][y4]), int(img[x6][y5]), int(img[x6][y]), int(img[x6][y6]), int(img[x6][y7]), int(img[x6][y8]), int(img[x6][y9]), int(img[x6][y10]),\
                                    int(img[x7][y1]), int(img[x7][y2]), int(img[x7][y3]), int(img[x7][y4]), int(img[x7][y5]), int(img[x7][y]), int(img[x7][y6]), int(img[x7][y7]), int(img[x7][y8]), int(img[x7][y9]), int(img[x7][y10]),\
                                    int(img[x8][y1]), int(img[x8][y2]), int(img[x8][y3]), int(img[x8][y4]), int(img[x8][y5]), int(img[x8][y]), int(img[x8][y6]), int(img[x8][y7]), int(img[x8][y8]), int(img[x8][y9]), int(img[x8][y10]),\
                                    int(img[x9][y1]), int(img[x9][y2]), int(img[x9][y3]), int(img[x9][y4]), int(img[x9][y5]), int(img[x9][y]), int(img[x9][y6]), int(img[x9][y7]), int(img[x9][y8]), int(img[x9][y9]), int(img[x9][y10]),\
                                    int(img[x10][y1]), int(img[x10][y2]), int(img[x10][y3]), int(img[x10][y4]), int(img[x10][y5]), int(img[x10][y]), int(img[x10][y6]), int(img[x10][y7]), int(img[x10][y8]), int(img[x10][y9]), int(img[x10][y10]),\
                                    ])
            magitude = np.dot(np.array( [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0, \
                                         0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0, \
                                         0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0, \
                                        -1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1, \
                                        -1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1, \
                                        -2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2, \
                                        -1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1, \
                                        -1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1, \
                                         0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0, \
                                         0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0, \
                                         0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0 \
                                    ] ), coordinate)
            
            if magitude >= threshold :
                new_img[x-5][y-5] = 1
            elif magitude <= threshold*(-1):
                new_img[x-5][y-5] = -1
            else:
                new_img[x-5][y-5] = 0

    return new_img

def get_diff_gauss_laplacian_operator(img, threshold):
    """
    :type img: Image(numpy 2d)
    :type threshold: int
    :return type: Image (0, 1) 
    """

    h, w = img.shape
    new_img = img.copy().astype('int16') 
    img = extend_padding(img, 5)

    for x in range( 5,h+5 ):
        for y in range( 5,w+5 ):

            # x1               x1y1   x1y   x1y2
            #    x             x y1   x y   x y2
            #      x2          x2y1   x2y   x2y2 
            x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 = x-5, y-5, x-4, y-4, x-3, y-3, x-2, y-2, x-1, y-1
            x6, y6, x7, y7, x8, y8, x9, y9, x10, y10 = x+1, y+1, x+2, y+2, x+3, y+3, x+4, y+4, x+5, y+5

            coordinate = np.array( [int(img[x1][y1]), int(img[x1][y2]), int(img[x1][y3]), int(img[x1][y4]), int(img[x1][y5]), int(img[x1][y]), int(img[x1][y6]), int(img[x1][y7]), int(img[x1][y8]), int(img[x1][y9]), int(img[x1][y10]),\
                                    int(img[x2][y1]), int(img[x2][y2]), int(img[x2][y3]), int(img[x2][y4]), int(img[x2][y5]), int(img[x2][y]), int(img[x2][y6]), int(img[x2][y7]), int(img[x2][y8]), int(img[x2][y9]), int(img[x2][y10]),\
                                    int(img[x3][y1]), int(img[x3][y2]), int(img[x3][y3]), int(img[x3][y4]), int(img[x3][y5]), int(img[x3][y]), int(img[x3][y6]), int(img[x3][y7]), int(img[x3][y8]), int(img[x3][y9]), int(img[x3][y10]),\
                                    int(img[x4][y1]), int(img[x4][y2]), int(img[x4][y3]), int(img[x4][y4]), int(img[x4][y5]), int(img[x4][y]), int(img[x4][y6]), int(img[x4][y7]), int(img[x4][y8]), int(img[x4][y9]), int(img[x4][y10]),\
                                    int(img[x5][y1]), int(img[x5][y2]), int(img[x5][y3]), int(img[x5][y4]), int(img[x5][y5]), int(img[x5][y]), int(img[x5][y6]), int(img[x5][y7]), int(img[x5][y8]), int(img[x5][y9]), int(img[x5][y10]),\
                                    int(img[x ][y1]), int(img[x ][y2]), int(img[x ][y3]), int(img[x ][y4]), int(img[x ][y5]), int(img[x ][y]), int(img[x ][y6]), int(img[x ][y7]), int(img[x ][y8]), int(img[x ][y9]), int(img[x ][y10]),\
                                    int(img[x6][y1]), int(img[x6][y2]), int(img[x6][y3]), int(img[x6][y4]), int(img[x6][y5]), int(img[x6][y]), int(img[x6][y6]), int(img[x6][y7]), int(img[x6][y8]), int(img[x6][y9]), int(img[x6][y10]),\
                                    int(img[x7][y1]), int(img[x7][y2]), int(img[x7][y3]), int(img[x7][y4]), int(img[x7][y5]), int(img[x7][y]), int(img[x7][y6]), int(img[x7][y7]), int(img[x7][y8]), int(img[x7][y9]), int(img[x7][y10]),\
                                    int(img[x8][y1]), int(img[x8][y2]), int(img[x8][y3]), int(img[x8][y4]), int(img[x8][y5]), int(img[x8][y]), int(img[x8][y6]), int(img[x8][y7]), int(img[x8][y8]), int(img[x8][y9]), int(img[x8][y10]),\
                                    int(img[x9][y1]), int(img[x9][y2]), int(img[x9][y3]), int(img[x9][y4]), int(img[x9][y5]), int(img[x9][y]), int(img[x9][y6]), int(img[x9][y7]), int(img[x9][y8]), int(img[x9][y9]), int(img[x9][y10]),\
                                    int(img[x10][y1]), int(img[x10][y2]), int(img[x10][y3]), int(img[x10][y4]), int(img[x10][y5]), int(img[x10][y]), int(img[x10][y6]), int(img[x10][y7]), int(img[x10][y8]), int(img[x10][y9]), int(img[x10][y10]),\
                                    ])
            magitude = np.dot(np.array( [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1, \
                                         -3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3, \
                                         -4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4, \
                                         -6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -1, \
                                         -7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7, \
                                         -8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8, \
                                         -7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7, \
                                         -6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -1, \
                                         -4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4, \
                                         -3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3, \
                                         -1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1 \
                                    ] ), coordinate)
            
            if magitude >= threshold :
                new_img[x-5][y-5] = 1
            elif magitude <= threshold*(-1):
                new_img[x-5][y-5] = -1
            else:
                new_img[x-5][y-5] = 0

    return new_img

def zero_crossing_edge_detecotr(img):
    """
    :type img: Image(numpy 2d) (0, 1)
    :return type: Image (0~255)
    """

    h, w = img.shape
    new_img = img.copy().astype('int16') 
    img = extend_padding(img, 1)

    for x in range( 1,h+1 ):
        for y in range( 1,w+1 ):

            if img[x][y] == 1:
                flag = 1
                for a in range(-1, 2):
                    for b in range(-1, 2):
                        if img[x+a][y+b] == -1:
                            flag = 0
                if flag == 1:

                    new_img[x-1][y-1] = 255
                else:
                    new_img[x-1][y-1] = 0
            else:
                new_img[x-1][y-1] = 255

    return new_img

def main():
    img = cv2.imread('lena.bmp',0)

    cv2.imwrite("1-laplacian1.bmp", zero_crossing_edge_detecotr(get_laplacian1_operator(img, 15)))
    cv2.imwrite("2-laplacian2.bmp", zero_crossing_edge_detecotr(get_laplacian2_operator(img, 15)))
    cv2.imwrite("3-minvar_laplacian.bmp", zero_crossing_edge_detecotr(get_minvar_laplacian_operator(img, 20)))
    cv2.imwrite("4-gauss_laplacian.bmp", zero_crossing_edge_detecotr(get_gauss_laplacian_operator(img, 3000)))
    cv2.imwrite("5-diff_gauss_laplacian.bmp", zero_crossing_edge_detecotr(get_diff_gauss_laplacian_operator(img, 1)))

    # === TEST AREA === #
    #img = np.array([ [153,166,85,90,96,102],[161,164,161,166,171,173],[103,105,97,97,97,103],[126,123,117,115,145,86],[135,129,110,114,107,113],[134,128,130,134,139,129]
    #   ])
    #print(zero_crossing_edge_detecotr(get_laplacian1_operator(img, 15)))



if __name__ == '__main__':
    main()

