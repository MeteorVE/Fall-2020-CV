import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
CV-2020 Hw4
Author: MeteorV

Write programs which do binary morphology on a binary image:
(a) Dilation
(b) Erosion
(c) Opening
(d) Closing
(e) Hit-and-miss transform

- Binarize Lena with the threshold 128 (0-127,128-255).
- Please use the octogonal 3-5-5-5-3 kernel.
- Please use the "L" shaped kernel (same as the text book) to detect the upper-right corner for hit-and-miss transform.
- Please process the white pixels (operating on white pixels).
- 5 images should be included in your report: Dilation, Erosion, Opening, Closing, and Hit-and-Miss.
"""

"""
Notice: More flexible way of Dilation、Erosion write in hw5.
"""

# sel = struct element = kernel
# g_val = gray value

def show(img, show_grid=True, show_ticks=False):
    """Plot the given image"""
    width, height = img.shape
    axes = plt.gca()
    if show_ticks:
        axes.set_xticks(np.arange(-.5, width, 1))
        axes.set_yticks(np.arange(-.5, height, 1))
        axes.set_xticklabels(np.arange(0, width + 1, 1))
        axes.set_yticklabels(np.arange(0, height + 1, 1))
    else:
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
    plt.grid(show_grid)
    plt.imshow(img, cmap=plt.cm.binary_r)
    plt.show()
    return plt.imshow(img, cmap=plt.cm.binary_r)

def all_elemtent_and(a,b):
    """all elemtents of two-dim array will operate AND to each other.""" 
    """input: two 2-dim arr, output: np.array (2-dim)"""
    if a.shape != b.shape:
        print("WARNING !!!!!!!!!!!")
        print(a.shape, b.shape)
        return

    w, h = a.shape
    ret = np.zeros((w, h), dtype=np.uint8)
    #ret = [[0 for i in range(w)]  for j in range(h)]
    for i in range(w):
        for j in range(h) :
            ret[i][j] = a[i][j] or b[i][j]
    return ret

def add_padding(img, radius_pair):
    """ radius_pair: w,h of selem/2  """
    w, h = img.shape
    new_img = np.zeros((w+radius_pair[0]*2, h+radius_pair[1]*2), dtype=np.uint8)
    new_img[ radius_pair[0]:radius_pair[0]+w, radius_pair[1]:radius_pair[1]+h ] = img
    return new_img

def compare_contain(img, sel, g_val):
    """g_val means gray value"""

    if img.shape != sel.shape:
        print("WARNING !!!!!!!!!!!")
        print(img.shape, sel.shape)

    idx_selem = np.where(sel.ravel() == g_val)[0]
    idx_img = np.where(img.ravel() == g_val)[0]

    idx_neg =  np.where(sel.ravel() == -1)[0]
    idx_zero = np.where(img.ravel() == 0)[0]
    #print(idx_neg)
    return (set(idx_selem) <= set(idx_img)) and (set(idx_neg) <= set(idx_zero))

def dilation(img, sel, g_val):

    # 1. create a padding added img, with all zero or contain src img.
    sw, sh = sel.shape
    img_result = add_padding(img, (int(sw/2),int(sh/2)))

    # 2. go through all of nodes of image.

    w,h = img.shape
    for x in range(h):
        for y in range(w):
            if img[x][y] == g_val:
                # copy the sele to new img (with AND)
                # use AND because we can't just overwrite the new_img, it contains replacment of other coordinate.

                img_result[x-int(sw/2):x+int(sw/2)+1, y-int(sh/2):y+int(sh/2)+1] = all_elemtent_and(sel, img_result[x-int(sw/2):x+int(sw/2)+1, y-int(sh/2):y+int(sh/2)+1])

    # get the result img.
    # show(img_result, show_ticks=True, show_grid=True)
    return img_result

def erosion(img, sel, g_val):
    sw, sh = sel.shape

    # 1. create a empty image, we will record something on that.
    img_result = np.zeros(img.shape, dtype=np.uint8)

    # img_result = add_padding(np.zeros(img.shape, dtype=np.uint8), (int(sw/2),int(sh/2)))


    # 2. check every "block", if src image's block contain the selem, record 1.
    w, h = img.shape
    for x in range(w-int(sh)+1):
        for y in range(h-int(sw)+1):
            if compare_contain(img[x:x+sw, y:y+sh] , sel, g_val):
                img_result[x+int(sw/2)][y+int(sh/2)] = g_val
    
    #show(img_result/g_val, show_ticks=True, show_grid=True)
    return img_result

def opening(img, sel, g_val):
    # erosion -> dilation
    return dilation(erosion(img, sel, g_val), sel, g_val)

def closing(img, sel, g_val):
    # dilation -> erosion
    return erosion(dilation(img, sel, g_val), sel, g_val)

def get_rotate_sel(sel):
    """rotate the sel 90deg, 180deg, 270deg"""
    mat90 = np.rot90(sel, 1) # rorate 90 <left> anti-clockwise
    mat180 = np.rot90(sel, 2) # rorate 180 <left> anti-clockwise
    mat270 = np.rot90(sel, 3) # rorate 270 <left> anti-clockwise
    return [sel, mat90, mat180, mat270]

def complement_img(img, g_val):
    result = np.zeros(img.shape, dtype=np.uint8)
    w,h = img.shape
    for x in range(h):
        for y in range(w):
            result[x][y] = 0 if img[x][y] == g_val else g_val
    return result

def intersection_2d(a, b):
    if a.shape != b.shape:
        print("WARNING !!! ", a.shape, b.shape)

    w, h = a.shape
    result = np.zeros(a.shape, dtype=np.uint8)
    for x in range(h):
        for y in range(w):
            result[x][y] = a[x][y] and b[x][y]
    return result

def union_2d(a, b):
    if a.shape != b.shape:
        print("WARNING !!! ", a.shape, b.shape)

    w, h = a.shape
    result = np.zeros(a.shape, dtype=np.uint8)
    for x in range(h):
        for y in range(w):
            result[x][y] = a[x][y] or b[x][y]
    return result

def hit_and_miss(img, sel, miss_kernel, g_val):
    """miss_kernel: a kernel bigger than sel, and it is a complement of sel."""

    # 1. get sel and a compoment bigger than sel named M
    sel_arr = get_rotate_sel(sel)
    mk_arr = get_rotate_sel(miss_kernel)
    img_c = complement_img(img, g_val)

    # 2. get eorse(img, sel)
    # 3. get eorse(complement of img, M)
    # Do 4 times eorse.

    hit_arr =[] # array of erose(img, sel)
    miss_arr = [] # array of erose(img_complement, miss_kernel)

    for i in range(len(sel_arr)):
        hit_arr.append( erosion(img, sel_arr[i], g_val) ) 
        miss_arr.append( erosion(img_c, mk_arr[i], g_val) ) 

    # 4. get intersection of 2. and 3.
    # there are 4 corner, so we have 4 image of intersection
   
    hit_and_miss_array = []
    for i in range(len(hit_arr)):
        hit_and_miss_array.append( intersection_2d(hit_arr[i], miss_arr[i]) )

    # 5. get OR of all hit and miss transformation.

    result = hit_and_miss_array[0]
    for i in range(len(hit_and_miss_array)-1):
        result = union_2d(result, hit_and_miss_array[i+1])

    return result


def hit_or_miss_upper_right(img, sel, miss_kernel, g_val):

    # 1. get sel and a compoment bigger than sel named M
    hk = np.rot90(sel, 1) # hit kernel
    mk = np.rot90(miss_kernel, 1) # miss kernel
    img_c = complement_img(img, g_val)

    # 2. get eorse(img, sel)
    # 3. get eorse(complement of img, M)

    h_img = erosion(img, hk, g_val)
    m_img = erosion(img_c, mk, g_val)

    return intersection_2d(h_img, m_img)


kernel = np.array([[0,1,1,1,0],
                   [1,1,1,1,1],
                   [1,1,1,1,1],
                   [1,1,1,1,1],
                   [0,1,1,1,0]], dtype=np.uint8)

L = np.array([[0,1,0],
              [1,1,0],
              [0,0,0]], dtype=np.uint8)

M = np.array([[ 0,-1, 0],
              [-1,-1, 1],
              [ 0, 1, 1]], dtype=np.int8)

#img = cv2.imread('j.png',0)
img = cv2.imread('binarize.bmp',0)

cv2.imwrite('erosion.bmp', erosion(img, kernel*255, 255))
cv2.imwrite('dilation.bmp', dilation(img, kernel*255, 255))
cv2.imwrite('opening.bmp', opening(img, kernel*255, 255))
cv2.imwrite('closing.bmp', closing(img, kernel*255, 255))
cv2.imwrite('hm_ur.bmp', hit_or_miss_upper_right(img, L*255, M*255, 255)) # hit_and_miss(img, sel, miss_kernel, g_val)


#print(img[0][0]) # 255 = 白 = 1
# 0 是黑的，但如果整張圖都是 1 那就會是整張圖都黑的 ...
#show(img, show_ticks=True, show_grid=True)