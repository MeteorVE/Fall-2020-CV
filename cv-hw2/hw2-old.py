import copy

import numpy as np
import matplotlib.pyplot as plt
import cv2


def main():
    # usage: python ./main.py
    # load lena.bmp into a 2D numpy.array
    img = cv2.imread('lena.bmp', 0)

    # TODO: binarize the image (threshold at 128)
    img_bin = np.zeros(img.shape, np.int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] >= 128:
                img_bin[i][j] = 255

    cv2.imwrite('binarize.bmp', img_bin)

    
    dist = np.zeros(256, np.int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            dist[img[i][j]] += 1

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # [0]:通道, None: 沒有使用 mask, [256]:HistSize 多少個直方柱, [0, 256]: 能表示像素值從 0~256 
    plt.title('Lena histogram'); plt.ylabel('number'); plt.xlabel('pixel value')
    plt.hist(img.ravel(),256,[0,256]); plt.show()

    # TODO: connected components (regions with + at centroid, bounding box)
    group = np.zeros(img.shape, np.int)
    k = 1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img_bin[i][j] > 0:
                group[i][j] = k
                k += 1
    group_backup = copy.deepcopy(group)

    while True:
        # top-down pass ...
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if i == 0:
                    if j == 0:
                    # up-left corner
                        if group[i][j] > 0:
                            # check all of its neighbors for minimum id
                            min_id = group[i][j]
                            if group[i][j + 1] > 0 and group[i][j + 1] < min_id:
                                min_id = group[i][j + 1]
                            if group[i + 1][j] > 0 and group[i + 1][j] < min_id:
                                min_id = group[i + 1][j]
                            # replace all of its neighbors with the minimum id
                            if group[i][j + 1] > 0:
                                group[i][j + 1] = min_id
                            if group[i + 1][j] > 0:
                                group[i + 1][j] = min_id
                    elif j == img.shape[1] - 1:
                    # up-right corner
                        if group[i][j] > 0:
                            min_id = group[i][j]
                            if group[i][j - 1] > 0 and group[i][j - 1] < min_id:
                                min_id = group[i][j - 1]
                            if group[i + 1][j] > 0 and group[i + 1][j] < min_id:
                                min_id = group[i + 1][j]
                            if group[i][j - 1] > 0:
                                group[i][j - 1] = min_id
                            if group[i + 1][j] > 0:
                                group[i + 1][j] = min_id
                    else:
                    # first row except the above two cases
                        if group[i][j] > 0:
                            min_id = group[i][j]
                            if group[i][j - 1] > 0 and group[i][j - 1] < min_id:
                                min_id = group[i][j - 1]
                            if group[i][j + 1] > 0 and group[i][j + 1] < min_id:
                                min_id = group[i][j + 1]
                            if group[i + 1][j] > 0 and group[i + 1][j] < min_id:
                                min_id = group[i + 1][j]
                            if group[i][j - 1] > 0:
                                group[i][j - 1] = min_id
                            if group[i][j + 1] > 0:
                                group[i][j + 1] = min_id
                            if group[i + 1][j] > 0:
                                group[i + 1][j] = min_id
                elif i == img.shape[0] - 1:
                    if j == 0:
                    # bottmo-left corner
                        if group[i][j] > 0:
                            min_id = group[i][j]
                            if group[i - 1][j] > 0 and group[i - 1][j] < min_id:
                                min_id = group[i - 1][j]
                            if group[i][j + 1] > 0 and group[i][j + 1] < min_id:
                                min_id = group[i][j + 1]
                            if group[i - 1][j] > 0:
                                group[i - 1][j] = min_id
                            if group[i][j + 1] > 0:
                                group[i][j + 1] = min_id
                    elif j == img.shape[1] - 1:
                    # bottom-right corner
                        if group[i][j] > 0:
                            min_id = group[i][j]
                            if group[i - 1][j] > 0 and group[i - 1][j] < min_id:
                                min_id = group[i - 1][j]
                            if group[i][j - 1] > 0 and group[i][j - 1] < min_id:
                                min_id = group[i][j + 1]
                            if group[i - 1][j] > 0:
                                group[i - 1][j] = min_id
                            if group[i][j - 1] > 0:
                                group[i][j - 1] = min_id
                    else:
                    # last row except the above two cases
                        if group[i][j] > 0:
                            min_id = group[i][j]
                            if group[i - 1][j] > 0 and group[i - 1][j] < min_id:
                                min_id = group[i - 1][j]
                            if group[i][j - 1] > 0 and group[i][j - 1] < min_id:
                                min_id = group[i][j - 1]
                            if group[i][j + 1] > 0 and group[i][j + 1] < min_id:
                                min_id = group[i][j + 1]
                            if group[i - 1][j] > 0:
                                group[i - 1][j] = min_id
                            if group[i][j - 1] > 0:
                                group[i][j - 1] = min_id
                            if group[i][j + 1] > 0:
                                group[i][j + 1] = min_id
                else:
                    if j == 0:
                        if group[i][j] > 0:
                            min_id = group[i][j]
                            if group[i - 1][j] > 0 and group[i - 1][j] < min_id:
                                min_id = group[i - 1][j]
                            if group[i][j + 1] > 0 and group[i][j + 1] < min_id:
                                min_id = group[i][j + 1]
                            if group[i + 1][j] > 0 and group[i + 1][j] < min_id:
                                min_id = group[i + 1][j]
                            if group[i - 1][j] > 0:
                                group[i - 1][j] = min_id
                            if group[i][j + 1] > 0:
                                group[i][j + 1] = min_id
                            if group[i + 1][j] > 0:
                                group[i + 1][j] = min_id
                    elif j == img.shape[1] - 1:
                        if group[i][j] > 0:
                            min_id = group[i][j]
                            if group[i - 1][j] > 0 and group[i - 1][j] < min_id:
                                min_id = group[i - 1][j]
                            if group[i][j - 1] > 0 and group[i][j - 1] < min_id:
                                min_id = group[i][j - 1]
                            if group[i + 1][j] > 0 and group[i + 1][j] < min_id:
                                min_id = group[i + 1][j]
                            if group[i - 1][j] > 0:
                                group[i - 1][j] = min_id
                            if group[i][j - 1] > 0:
                                group[i][j - 1] = min_id
                            if group[i + 1][j] > 0:
                                group[i + 1][j] = min_id
                    else:
                        if group[i][j] > 0:
                            min_id = group[i][j]
                            if group[i - 1][j] > 0 and group[i - 1][j] < min_id:
                                min_id = group[i - 1][j]
                            if group[i][j - 1] > 0 and group[i][j - 1] < min_id:
                                min_id = group[i][j - 1]
                            if group[i][j + 1] > 0 and group[i][j + 1] < min_id:
                                min_id = group[i][j + 1]
                            if group[i + 1][j] > 0 and group[i + 1][j] < min_id:
                                min_id = group[i + 1][j]
                            if group[i - 1][j] > 0:
                                group[i - 1][j] = min_id
                            if group[i][j - 1] > 0:
                                group[i][j - 1] = min_id
                            if group[i][j + 1] > 0:
                                group[i][j + 1] = min_id
                            if group[i + 1][j] > 0:
                                group[i + 1][j] = min_id
        # bottom-up pass ...
        for i in range(img.shape[0] - 1, -1, -1): # for lena.bmp: 511, 510, 509, ..., 2, 1, 0
            for j in range(img.shape[1]):
                if i == 0:
                    if j == 0:
                    # up-left corner
                        if group[i][j] > 0:
                            min_id = group[i][j]
                            if group[i][j + 1] > 0 and group[i][j + 1] < min_id:
                                min_id = group[i][j + 1]
                            if group[i + 1][j] > 0 and group[i + 1][j] < min_id:
                                min_id = group[i + 1][j]
                            if group[i][j + 1] > 0:
                                group[i][j + 1] = min_id
                            if group[i + 1][j] > 0:
                                group[i + 1][j] = min_id
                    elif j == img.shape[1] - 1:
                    # up-right corner
                        if group[i][j] > 0:
                            min_id = group[i][j]
                            if group[i][j - 1] > 0 and group[i][j - 1] < min_id:
                                min_id = group[i][j - 1]
                            if group[i + 1][j] > 0 and group[i + 1][j] < min_id:
                                min_id = group[i + 1][j]
                            if group[i][j - 1] > 0:
                                group[i][j - 1] = min_id
                            if group[i + 1][j] > 0:
                                group[i + 1][j] = min_id
                    else:
                    # first row except the above two cases
                        if group[i][j] > 0:
                            min_id = group[i][j]
                            if group[i][j - 1] > 0 and group[i][j - 1] < min_id:
                                min_id = group[i][j - 1]
                            if group[i][j + 1] > 0 and group[i][j + 1] < min_id:
                                min_id = group[i][j + 1]
                            if group[i + 1][j] > 0 and group[i + 1][j] < min_id:
                                min_id = group[i + 1][j]
                            if group[i][j - 1] > 0:
                                group[i][j - 1] = min_id
                            if group[i][j + 1] > 0:
                                group[i][j + 1] = min_id
                            if group[i + 1][j] > 0:
                                group[i + 1][j] = min_id
                elif i == img.shape[0] - 1:
                    if j == 0:
                    # bottom-left corner
                        if group[i][j] > 0:
                            min_id = group[i][j]
                            if group[i - 1][j] > 0 and group[i - 1][j] < min_id:
                                min_id = group[i - 1][j]
                            if group[i][j + 1] > 0 and group[i][j + 1] < min_id:
                                min_id = group[i][j + 1]
                            if group[i - 1][j] > 0:
                                group[i - 1][j] = min_id
                            if group[i][j + 1] > 0:
                                group[i][j + 1] = min_id
                    elif j == img.shape[1] - 1:
                    # bottom-right corner
                        if group[i][j] > 0:
                            min_id = group[i][j]
                            if group[i - 1][j] > 0 and group[i - 1][j] < min_id:
                                min_id = group[i - 1][j]
                            if group[i][j - 1] > 0 and group[i][j - 1] < min_id:
                                min_id = group[i][j + 1]
                            if group[i - 1][j] > 0:
                                group[i - 1][j] = min_id
                            if group[i][j - 1] > 0:
                                group[i][j - 1] = min_id
                    else:
                    # last row except the above two cases
                        if group[i][j] > 0:
                            min_id = group[i][j]
                            if group[i - 1][j] > 0 and group[i - 1][j] < min_id:
                                min_id = group[i - 1][j]
                            if group[i][j - 1] > 0 and group[i][j - 1] < min_id:
                                min_id = group[i][j - 1]
                            if group[i][j + 1] > 0 and group[i][j + 1] < min_id:
                                min_id = group[i][j + 1]
                            if group[i - 1][j] > 0:
                                group[i - 1][j] = min_id
                            if group[i][j - 1] > 0:
                                group[i][j - 1] = min_id
                            if group[i][j + 1] > 0:
                                group[i][j + 1] = min_id
                else:
                    if j == 0:
                        if group[i][j] > 0:
                            min_id = group[i][j]
                            if group[i - 1][j] > 0 and group[i - 1][j] < min_id:
                                min_id = group[i - 1][j]
                            if group[i][j + 1] > 0 and group[i][j + 1] < min_id:
                                min_id = group[i][j + 1]
                            if group[i + 1][j] > 0 and group[i + 1][j] < min_id:
                                min_id = group[i + 1][j]
                            if group[i - 1][j] > 0:
                                group[i - 1][j] = min_id
                            if group[i][j + 1] > 0:
                                group[i][j + 1] = min_id
                            if group[i + 1][j] > 0:
                                group[i + 1][j] = min_id
                    elif j == img.shape[1] - 1:
                        if group[i][j] > 0:
                            min_id = group[i][j]
                            if group[i - 1][j] > 0 and group[i - 1][j] < min_id:
                                min_id = group[i - 1][j]
                            if group[i][j - 1] > 0 and group[i][j - 1] < min_id:
                                min_id = group[i][j - 1]
                            if group[i + 1][j] > 0 and group[i + 1][j] < min_id:
                                min_id = group[i + 1][j]
                            if group[i - 1][j] > 0:
                                group[i - 1][j] = min_id
                            if group[i][j - 1] > 0:
                                group[i][j - 1] = min_id
                            if group[i + 1][j] > 0:
                                group[i + 1][j] = min_id
                    else:
                        if group[i][j] > 0:
                            min_id = group[i][j]
                            if group[i - 1][j] > 0 and group[i - 1][j] < min_id:
                                min_id = group[i - 1][j]
                            if group[i][j - 1] > 0 and group[i][j - 1] < min_id:
                                min_id = group[i][j - 1]
                            if group[i][j + 1] > 0 and group[i][j + 1] < min_id:
                                min_id = group[i][j + 1]
                            if group[i + 1][j] > 0 and group[i + 1][j] < min_id:
                                min_id = group[i + 1][j]
                            if group[i - 1][j] > 0:
                                group[i - 1][j] = min_id
                            if group[i][j - 1] > 0:
                                group[i][j - 1] = min_id
                            if group[i][j + 1] > 0:
                                group[i][j + 1] = min_id
                            if group[i + 1][j] > 0:
                                group[i + 1][j] = min_id
        # stop iterating if no change
        if False not in (group == group_backup):
            break
        # keep track of latest group
        group_backup = copy.deepcopy(group)

    # For the connected components, please use 500 pixels as a threshold.
    # Omit regions that have a pixel count less than 500.
    count_pixel = np.zeros(np.max(group) + 1, np.int)
    for i in range(group.shape[0]):
        for j in range(group.shape[1]):
            if group[i][j] > 0:
                count_pixel[group[i][j]] += 1
    for i in range(group.shape[0]):
        for j in range(group.shape[1]):
            if count_pixel[group[i][j]] < 500:
                group[i][j] = 0
    # count the number of regions ...
    n_regions = np.sum(count_pixel >= 500)

    # plot the bounding box ...
    for i in range(1, count_pixel.shape[0]):
        if count_pixel[i] >= 500:
            # i is the region id
            ind_set = np.array(np.where(group == i)).T
            # locate the up-left point and the bottom-right point
            ind_up_left = np.array([np.min(ind_set[:, 0]), np.min(ind_set[:, 1])])
            ind_bottom_right = np.array([np.max(ind_set[:, 0]), np.max(ind_set[:, 1])])
            # draw 4 lines with pixel value = 128 ...
            img_bin[ind_up_left[0]:ind_bottom_right[0] + 1, ind_up_left[1]] = 128
            img_bin[ind_up_left[0]:ind_bottom_right[0] + 1, ind_bottom_right[1]] = 128
            img_bin[ind_up_left[0], ind_up_left[1]:ind_bottom_right[1] + 1] = 128
            img_bin[ind_bottom_right[0], ind_up_left[1]:ind_bottom_right[1] + 1] = 128

            x_center = int((ind_up_left[0]+ind_bottom_right[0])/2)
            y_center = int((ind_up_left[1]+ind_bottom_right[1])/2)

            img_bin[x_center:x_center + 5, y_center] = 128
            img_bin[x_center - 5:x_center, y_center] = 128
            img_bin[x_center, y_center-5:y_center  ] = 128
            img_bin[x_center, y_center:y_center + 5] = 128

            print(ind_up_left[0], ind_bottom_right[0], ind_up_left[1],ind_bottom_right[1])

    cv2.imwrite('connect_4.bmp', img_bin)


if __name__ == '__main__':
    main()