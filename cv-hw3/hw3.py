import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

"""
CV-2020 Hw3
Author: MeteorV

Write a program to generate images and histograms:
(a) original image and its histogram
(b) image with intensity divided by 3 and its histogram
(c) image after applying histogram equalization to (b) and its histogram
"""


def divide_three(img):
	"""
	:Usage: all pixel value divide 3
	:type img: Image (3D Array)
	:return type: 3D Array
	"""   
	for i in range(int(len(img))):
		row = img[i] # 第 i 個 row
		for c in range(int(len(row))):
			# row[c] 是一個 tuple 
			row[c] = (row[c]/3).astype(np.uint8)
	return img

def histogram_equalization(img):
	"""
	:Usage: histogram equalization
	:type img: Image (3D Array)
	:return type: 3D Array
	"""  
	img = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).flatten()
	
	# compute appear times
	prob = [0 for i in range(256)] 
	print(img[0])
	for i in img:
		prob[int(i)] += 1
	
	cumsum = np.cumsum(prob)
	cdf_max, cdf_min = max(cumsum), min(cumsum)
	
	for i in range(len(prob)):
		prob[i] = ((cumsum[i]-cdf_min)*255/(cdf_max - cdf_min)).astype(np.uint8)

	for i in range(len(img)):
		img[i] = prob[img[i]]
	
	img = np.reshape(img, (-1, 512))
	return img

def show(name,img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def main():
	print("usage :\npython hw2.py [function number]\n1: get a binary image, 2: get a img of intensity divided by 3,\n 3: applying histogram equalization\n\n")
	assert len(sys.argv) == 2
	try:
		img = cv2.imread('lena.bmp')
	except:
		print("open img error")

	if sys.argv[1] == '1':

		hist = cv2.calcHist([img], [0], None, [256], [0, 256])
		# [0]:通道, None: 沒有使用 mask, [256]:HistSize 多少個直方柱, [0, 256]: 能表示像素值從 0~256 
		
		plt.title('Lena origin histogram'); plt.ylabel('number'); plt.xlabel('pixel value')
		plt.hist(img.ravel(),256,[0,256]); 
		plt.savefig('origin_histogram') # plt.show()

	if sys.argv[1] == '2':
		img = divide_three(img)
		cv2.imwrite('divide_three.bmp', img)

		hist = cv2.calcHist([img], [0], None, [256], [0, 256])
		plt.title('Lena intensity/3 histogram'); plt.ylabel('number'); plt.xlabel('pixel value')
		plt.hist(img.ravel(),256,[0,256]); 
		plt.savefig('divide_three_histogram') # plt.show()

	if sys.argv[1] == '3':
		img = histogram_equalization(img)
		cv2.imwrite('histogram_equalization.bmp', img)


		hist = cv2.calcHist([img], [0], None, [256], [0, 256])
		plt.title('Lena histogram equalization'); plt.ylabel('number'); plt.xlabel('pixel value')
		plt.hist(img.ravel(),256,[0,256]); 
		plt.savefig('histogram_equalization_histogram') # plt.show()

main()
