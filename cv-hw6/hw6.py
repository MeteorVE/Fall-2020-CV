import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
CV-2020 Hw6 Yokoi Connectivity Number
Author: MeteorV

Write a program which counts the Yokoi connectivity number on a downsampled image(lena.bmp).

- Downsampling Lena from 512x512 to 64x64:
  - Binarize the benchmark image lena as in HW2, then using 8x8 blocks as a unit, take the topmost-left pixel as the downsampled data.
- Count the Yokoi connectivity number on a downsampled lena using 4-connected.
- Result of this assignment is a 64x64 matrix.
- You can use any programing language to implement homework, however, you'll get zero point if you just call existing library.
"""

# x7  x2  x6
# x3  x0  x1
# x8  x4  x5

# h(x0, x1, x6, x2)
# h(x0, x2, x7, x3)
# h(x0, x3, x8, x4)
# h(x0, x4, x5, x1)


def h_function_of_Yokoi(b, c, d, e):
	"""
	:type b, c, d, e: pixel number
	:return type: Char ('q' or 'r' or 's')
	"""

	if b == c and ( d!=b or e!= b ):
		return 'q'
	elif b == c and (d ==b and e == b):
		return 'r'
	elif b != c:
		return 's'
	else:
		print("WARNING!")
		return 's'

def f_function_of_Yokoi(a1, a2, a3, a4):
	"""
	:type a1~a4: 'q' or 'r' or 's' 
	:return type: Int (0~5)
	"""

	ret = 0
	if a1 == 'q':
		ret += 1
	if a2 == 'q':
		ret += 1
	if a3 == 'q':
		ret += 1
	if a4 == 'q':
		ret += 1
	if a1 == a2 and a2 == a3 and a3 == a4 and a4 == 'r':
		return 5
	return  ret

def down_sampling(src_img, scale):
	"""
	:type src_img: Image (read by Numpy)
	:type scale: Int
	:return type: Image (Numpy)
	"""

	h, w = src_img.shape
	print(src_img.shape, int(w/scale), int(h/scale))
	new_img = np.zeros((int(w/scale), int(h/scale)), dtype=np.uint8)
	for x in range( new_img.shape[0] ):
		for y in range ( new_img.shape[1]):
			new_img[x][y] = src_img[x*scale][y*scale]
	return new_img

def add_padding(img, radius_pair):
	"""
	:type img: Image (read by Numpy)
	:type radius_pair: Pair (w,h which need to expand)
	"""
	
	h, w = img.shape
	#123 is the frame pixel value, just let it not 0 or 255
	new_img = np.full((w+radius_pair[0]*2, h+radius_pair[1]*2), 123 ,dtype=np.uint8) 
	new_img[ radius_pair[0]:radius_pair[0]+w, radius_pair[1]:radius_pair[1]+h ] = img
	return new_img

def binarilization(img, threshold):
	h, w = img.shape
	for x in range(h):
		for y in range(w):
			if img[x][y] >= threshold:
				img[x][y] = 255
			else:
				img[x][y] = 0
	cv2.imwrite('binarilization.bmp', img)

	return img

def get_Yokoi_number(img, scale):
	"""
	:type src_img: Image (read by Numpy)
	:type scale: Int
	:return type: Image (Numpy)
	"""

	new_img = down_sampling(img, scale)
	cv2.imwrite('down_sampling.bmp', new_img)

	new_img = add_padding(new_img, (1, 1)) # prevent the frame just contain 6 pixel or 4 pixel

	result = [['0' for x in range(64)] for y in range(64)]
	for x in range( 1, new_img.shape[0]-1 ):
		for y in range ( 1, new_img.shape[1] -1):

			if new_img[x][y] == 0:
				result[x-1][y-1] = ' '
				continue

			x0, x1, x2, x3, x4 = new_img[x][y], new_img[x+1][y], new_img[x][y-1], new_img[x-1][y], new_img[x][y+1]
			x5, x6, x7, x8 = new_img[x+1][y+1], new_img[x+1][y-1], new_img[x-1][y-1], new_img[x-1][y+1]
			a1 = h_function_of_Yokoi(x0, x1, x6, x2)
			a2 = h_function_of_Yokoi(x0, x2, x7, x3)
			a3 = h_function_of_Yokoi(x0, x3, x8, x4)
			a4 = h_function_of_Yokoi(x0, x4, x5, x1)

			result[x-1][y-1] = f_function_of_Yokoi(a1, a2, a3, a4)
			
			#print(x0, x1, x2, x3, x4,x5, x6, x7, x8, a1, a2, a3, a4, result[x-1][y-1]);input()
	return result


def main():
	
	img = cv2.imread('lena.bmp',0)
	ret = get_Yokoi_number(binarilization(img, 128), 8)

	with open('output.txt', 'w' , encoding='UTF-8') as f:
		for x in range(64):
			for y in range(64):
				if ret[x][y] !=0:
					f.write(str(ret[x][y]))
				else:
					f.write(' ')
			f.write('\n')
