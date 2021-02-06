import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
CV-2020 Hw5 Mathematical Morphology - Gray Scaled Morphology
Author: MeteorV

Write programs which do gray-scale morphology on a gray-scale image(lena.bmp):
(a) Dilation
(b) Erosion
(c) Opening
(d) Closing

- Please use the octogonal 3-5-5-5-3 kernel. (which is actually taking the local maxima or local minima respectively).
- 4 images should be included in your report: Dilation, Erosion, Opening and Closing.
- You can use any programing language to implement homework, however, you'll get zero point if you just call existing library.
"""

def add_padding(img, radius_pair, fill_value=0):
    """
    :Usage: add padding to image
    :type img: Image (2D Array)
    :type radius_pair: Pair(int, int) = (kernel_width/2, kernel_height/2)
    :return type: 2D Array
    """  
    w, h = img.shape
    new_img = np.full( (w+radius_pair[0]*2, h+radius_pair[1]*2), fill_value, dtype=np.uint8 )
    new_img[ radius_pair[0]:radius_pair[0]+w, radius_pair[1]:radius_pair[1]+h ] = img
    return new_img

def dilation(img, kernel):
    padding_img = add_padding(img, (2,2), 0)
    img_result  = img.copy()

    # check the max of kernel shape
    w, h = img.shape
    kw, kh = 2, 2
    for x in range(2, h+2):
        for y in range(2, w+2):
            local_max = 0
            for a, b in kernel:
                if local_max < padding_img[x+a, y+b]:
                    local_max = padding_img[x+a, y+b]
            img_result[x-2, y-2] = local_max

    return img_result

def erosion(img, kernel):
    padding_img = add_padding(img, (2,2), 255)
    img_result  = img.copy() #add_padding(img, (2,2), 255)

    # check the max of kernel shape
    w, h = img.shape
    kw, kh = 2, 2
    for x in range(2, h+2):
        for y in range(2, w+2):
            
            local_min = 255
            for a, b in kernel:
                if local_min > padding_img[x+a, y+b]:
                    local_min = padding_img[x+a, y+b]

            img_result[x-2, y-2] = local_min

    return img_result

def main():

    k =  [(-2,-1),(-2,0),(-2,1),(-1,-2),(-1,-1),(-1,0),(-1,1),(-1,2),(0,-2),(0,-1),(0,0),(0,1),(0,2),(1,-2),(1,-1),(1,0),(1,1),(1,2),(2,-1),(2,0),(2,1)] 
    img = cv2.imread('lena.bmp',0)

    cv2.imwrite('dilation.bmp', dilation(img, k))
    cv2.imwrite('erosion.bmp', erosion(img, k))
    cv2.imwrite('opening.bmp', dilation(erosion(img, k),k))
    cv2.imwrite('closing.bmp', erosion(dilation(img, k),k))

main()