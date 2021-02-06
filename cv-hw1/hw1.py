import numpy as np
import cv2

"""
CV-2020 Hw1
Author: MeteorV

Part1. Write a program to do the following requirement.
(a) upside-down lena.bmp
(b) right-side-left lena.bmp
(c) diagonally flip lena.bmp
Part2. Write a program or use software to do the following requirement.
(d) rotate lena.bmp 45 degrees clockwise
(e) shrink lena.bmp in half
(f) binarize lena.bmp at 128 to get a binary image
"""

def upside_down_flip(img):
    """
    :type img: Image (3D Array)
    :return type: 3D Array
    """   
	for i in range(int(len(img)/2)):
		# upside-down
		img[i], img[len(img)-i-1] = img[len(img)-i-1], img[i].copy()
	
	return img

def right_side_left(img):
	"""
    :type img: Image (3D Array)
    :return type: 3D Array
    """   
	for i in range(int(len(img))):
		# right-side-left
		row = img[i]
		for c in range(int(len(row)/2)):
			row[c], row[len(row)-c-1] = row[len(row)-c-1], row[c].copy()
	return img

def diagonally_flip(img):
	"""
    :type img: Image (3D Array)
    :return type: 3D Array
    """   
	return right_side_left(upside_down_flip(img))

def rotate(image, angle, center=None, scale=1.0):
    """
    :type img: Image (3D Array)
    :return type: 3D Array
    """       
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated

def binarilization(img, threshold=128):
    """
    :type img: Image (3D Array)
    :return type: 3D Array
    """   
    h, w = img.shape
    for x in range(h):
        for y in range(w):
            if img[x][y] >= threshold:
                img[x][y] = 255
            else:
                img[x][y] = 0
    return img

def main():
	try:
		img = cv2.imread('lena.bmp')
	except:
		print("open img error")

	cv2.imwrite("1-upside_down_flip.bmp", upside_down_flip(img.copy()))
	cv2.imwrite("2-right_side_left.bmp", right_side_left(img.copy()))
	cv2.imwrite("3-diagonally_flip.bmp", diagonally_flip(img.copy()))
	cv2.imwrite("4-rotate_45_degrees.bmp", rotate(img.copy(), angle=45, scale=1.0))
	cv2.imwrite("5-shrink_half.bmp", cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation = cv2.INTER_AREA))
	cv2.imwrite('6-binarilization.bmp', binarilization(cv2.imread('lena.bmp', 0).copy()))


def show(name,img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

main()
