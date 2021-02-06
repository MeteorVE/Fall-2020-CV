import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

"""
CV-2020 Hw2
Author: MeteorV

Write a program to generate:
(a) a binary image (threshold at 128)
(b) a histogram
(c) connected components(regions with + at centroid, bounding box)

"""

def two_pass_algorithm(img):

    mark = np.full( img.shape, -1, dtype=np.int16 )
    background_color = 0
    disjoint_set = {}
    color_num = 0

    h, w = img.shape
    # first scan
    for x in range(h):
        for y in range(w):
            if img[x][y] != background_color:

                # 1. check if 4 direction are background
                neighbors_color = []
                if x-1 != -1 and img[x-1][y] != background_color:
                        neighbors_color.append(mark[x-1][y])
                if y-1 != -1 and img[x][y-1] != background_color:
                        neighbors_color.append(mark[x][y-1])

                neighbors_color = sorted(neighbors_color)
                if len(neighbors_color) == 0:
                    # all neighbor = background
                    mark[x][y] = color_num
                    disjoint_set[int(color_num)] = set()
                    color_num += 1
                else:
                    mark[x][y] = neighbors_color[0]
                    for child in neighbors_color[1:]:
                        # two way record
                        disjoint_set[neighbors_color[0]].add(child)
                        disjoint_set[child].add(neighbors_color[0])

    root = [ -1 for i in range(color_num)]

    # disjoint set
    for father in range(color_num):
        if root[father] == -1:
            root[father] = father
        else:
            continue

        stack = list(disjoint_set[father])
        while stack:
            child = stack.pop()
            if root[child] == -1:
                disjoint_set[father].add(child)
                root[child] = father
                stack += list(disjoint_set[child])


    # second pass
    information = {}
    for i in root:
        if i == root[i]:
            information[i] = { 'left':w, 'right':0, 'up':h, 'down':0, 'child':0 }

    for x in range(h):
        for y in range(w):
            if img[x][y] != background_color:
                mark[x][y] = root[mark[x][y]]
                
                information[mark[x][y]]['child'] += 1
                if information[mark[x][y]]['left'] > x:
                    information[mark[x][y]]['left'] = x
                if information[mark[x][y]]['right'] < x:
                    information[mark[x][y]]['right'] = x
                if information[mark[x][y]]['up'] > y:
                    information[mark[x][y]]['up'] = y
                if information[mark[x][y]]['down'] < y:
                    information[mark[x][y]]['down'] = y
    
    return information

def draw(img, information):

    for el in information:
        if information[el]['child'] > 500:
            left, right, up, down = information[el]['left'], information[el]['right'], information[el]['up'], information[el]['down']
            img[left:right, up] = (255,45,45)
            img[left:right, down] = (255,45,45)
            img[left, up:down] = (255,45,45)
            img[right, up:down] = (255,45,45)

            x_center = int((left + right)/2)
            y_center = int((up + down)/2)

            img[x_center:x_center + 5, y_center] = (45,45,255)
            img[x_center - 5:x_center, y_center] = (45,45,255)
            img[x_center, y_center-5:y_center  ] = (45,45,255)
            img[x_center, y_center:y_center + 5] = (45,45,255)

    return img

def generate_hist(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # [0]:通道, None: 沒有使用 mask, [256]:HistSize 多少個直方柱, [0, 256]: 能表示像素值從 0~256 
    plt.title('Lena histogram'); plt.ylabel('number'); plt.xlabel('pixel value')
    plt.hist(img.ravel(),256,[0,256]); plt.savefig('Lena Histogram'); 
    #plt.show() # for debug

def binarilization(img, threshold):
    h, w = img.shape
    for x in range(w):
        for y in range(h):
            if img[x][y] >= threshold:
                img[x][y] = 255
            else:
                img[x][y] = 0

    return img

def main():
    
    img = cv2.imread('lena.bmp',0)
    
    # (b) a histogram
    generate_hist(img)

    # (a) a binary image (threshold at 128)
    binary_image = binarilization(img, 128)
    cv2.imwrite('binarize.bmp', binary_image)


    # (c) connected components(regions with + at centroid, bounding box)
    info = two_pass_algorithm(binary_image)

    img_rgb = cv2.imread('binarize.bmp') # read as 3 tunnel 
    result = draw(img_rgb, info)
    cv2.imwrite('connect_4.bmp', result)

    # test area
    test = np.array([
        [0,255,  0,255,255,  0],
        [0,255,255,255,255,  0],
        [0,  0,  0,  0,  0,  0],
        [0,255,  0,255,255,  0],
        [0,255,  0,255,255,  0]
        ]) # use this to help you debug !
    #print(two_pass_algorithm(test))


main()