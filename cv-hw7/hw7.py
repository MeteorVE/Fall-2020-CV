import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
CV-2020 Hw7 Thinning
Author: MeteorV

Write a program which does thinning on a downsampled image (lena.bmp).

- Downsampling Lena from 512x512 to 64x64:
  - Binarize the benchmark image lena as in HW2, then using 8x8 blocks as a unit, take the topmost-left pixel as the downsampled data.
- You have to use 4-connected neighborhood detection.
- You can use any programing language to implement homework, however, you'll get zero point if you just call existing library.
- More detail about this homework. Download here--> thinning-operator.pdf
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

# PRO = Pair Relationship Operator
def h_function_of_PRO(a, m):
    if a == m:
        return 1
    else:
        return 0

# CSO = Connected_Shrink_Operator
def h_function_of_CSO(b, c, d, e):
    if b == c and (d != b or e != b):
        return 1
    else:
        return 0

def down_sampling(src_img, scale):
    """
    :type src_img: Image (read by Numpy)
    :type scale: Int
    :return type: Image (Numpy)
    """

    w, h = src_img.shape
    new_img = np.zeros((int(w/scale), int(h/scale)), dtype=np.uint8)
    for x in range( new_img.shape[0] ):
        for y in range ( new_img.shape[1]):
            new_img[x][y] = src_img[x*scale][y*scale]
    return new_img

def add_padding(img, radius_pair=(1,1), fill_number=123):
    """
    :type img: Image (read by Numpy, ndarray OR 2D-list)
    :type radius_pair: Pair (w,h which need to expand)
    :return type: Same type of img
    """
    
    w, h = len(img), len(img[0])
    #123 is the frame pixel value, just let it not 0 or 255
    if type(img) == np.ndarray:
        new_img = np.full((w+radius_pair[0]*2, h+radius_pair[1]*2), fill_number ,dtype=np.uint8) 
        new_img[ radius_pair[0]:radius_pair[0]+w, radius_pair[1]:radius_pair[1]+h ] = img
    else:
        new_img = [[fill_number for i in range(w+radius_pair[0]*2) ] for j in range(h+radius_pair[1]*2)]
        for i in range(w):
            new_img[i+1][1:-1] = img[i]
    return new_img



def binarilization(img, threshold=128):
    h, w = img.shape
    for x in range(h):
        for y in range(w):
            if img[x][y] >= threshold:
                img[x][y] = 255
            else:
                img[x][y] = 0
    cv2.imwrite('binarilization.bmp', img)

    return img

def get_Yokoi_graph(img):
    """
    :type src_img: Image (read by Numpy)
    :return type: Image (2D Array , Size: 64*64)
    """

    new_img = add_padding(img, (1, 1)) # prevent the frame just contain 6 pixel or 4 pixel

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
            
    return result

def get_pair_relationship_operator_graph(img, edge_number=1):
    """
    :type img: Image (Numpy array, processed by Yokoi)
    :type edge_number: Int (the edge number or pixel value, default is 1 in Yokoi table)
    :return type: 2D Array (Pair_Relationship_Operator result)
    """

    height, width = len(img), len(img[0])
    padding_img = add_padding(img, (1, 1))
    pro_table = [[ None for x in range(width)] for y in range(height) ] # PRO = Pair_Relationship_Operator
    
    for x in range(1,height+1):
        for y in range(1,width+1):
            h = 0
            if padding_img[x][y] == ' ':
                pro_table[x-1][y-1] = ' '
                continue
            
            h += h_function_of_PRO(padding_img[x][y], padding_img[x-1][y])
            h += h_function_of_PRO(padding_img[x][y], padding_img[x+1][y])
            h += h_function_of_PRO(padding_img[x][y], padding_img[x][y-1])
            h += h_function_of_PRO(padding_img[x][y], padding_img[x][y+1])
            
            if h >= 1 and padding_img[x][y] == edge_number:
                pro_table[x-1][y-1] = 'p'
            elif h < 1 or padding_img[x][y] != edge_number:
                pro_table[x-1][y-1] = 'q'
            else:
                pro_table[x-1][y-1] = ' '
                #print(x,y,pro_table[x-1][y-1])


    return pro_table

def get_connected_shrink_operator_graph(pro_table, ones_table):
    """
    :type img: Image (2D Array, processed by Pair_Relationship_Operator)
    :return type: 2D Array (Pair_Relationship_Operator result)
    """   

    # Notice : detect 'p' to decide which need to compute, but when computing, the compared element is read from ones_table.

    result_table = add_padding(ones_table, (1, 1), 0)
    height, width = len(pro_table), len(pro_table[0])

    for x in range(height):
        for y in range(width):
            if pro_table[x][y] == 'p':
                h = 0
                x+=1
                y+=1
                x0, x1, x2, x3, x4 = result_table[x][y], result_table[x+1][y], result_table[x][y-1], result_table[x-1][y], result_table[x][y+1]
                x5, x6, x7, x8 = result_table[x+1][y+1], result_table[x+1][y-1], result_table[x-1][y-1], result_table[x-1][y+1]
                h += h_function_of_CSO(x0, x1, x6, x2)
                h += h_function_of_CSO(x0, x2, x7, x3)
                h += h_function_of_CSO(x0, x3, x8, x4)
                h += h_function_of_CSO(x0, x4, x5, x1)

                if h == 1:
                    result_table[x][y] = 0
                x-=1
                y-=1

    # delete padding.
    for i in range(1, width+1):
        result_table[i] = result_table[i][1:-1]
    del result_table[0], result_table[-1]

    return result_table


def get_thinning_operator_graph(img):

    iteration = 1
    while True:
        print("iteration:", iteration)

        # Prepare for run Yokoi
        yokoi_table = get_Yokoi_graph(img)

        # Prepare for run Pair_Relationship_Operator
        pro_table = get_pair_relationship_operator_graph(yokoi_table, 1)
        writing('pro_table.txt', pro_table)

        # Prepare the image for Connected_Shrink_Operator
        h, w = len(pro_table), len(pro_table[0])
        ones_table = [[ None for x in range(w)] for y in range(h) ] # CSO = Connected_Shrink_Operator 
        for x in range(h):
            for y in range(w):
                if pro_table[x][y] != ' ' and pro_table[x][y] != None:
                    ones_table[x][y] = 1
                else:
                    ones_table[x][y] = 0

        writing('ones_table.txt', ones_table)
        cso_table = get_connected_shrink_operator_graph( pro_table, ones_table)

        # Compare the CSO_table and After Processing Table
        writing('cso_table.txt', cso_table)

        img = np.array(cso_table)*255
        cv2.imwrite(str(iteration)+'-iteration-thin.bmp', img)
        
        if cso_table == ones_table:
            break
        else:
            iteration += 1
            #input() # for debug
        
        
    return img


def writing(filename, arr):
    with open(filename, 'w' , encoding='UTF-8') as f:
        for x in range(64):
            for y in range(64):
                if arr[x][y] == None or arr[x][y] == 123:
                    f.write(' ')
                else:
                    f.write(str(arr[x][y]))
            f.write('\n')


def main():
    img = cv2.imread('lena.bmp',0)
    result = get_thinning_operator_graph( down_sampling(binarilization(img, 128), 8) )
    cv2.imwrite('thin.bmp', result)


if __name__ == '__main__':
    main()


# === TESTING AREA === #

#img = np.array([[1,1,0,0,0],[0,1,1,0,0],[0,0,1,1,0],[0,0,1,1,0],[0,0,1,1,0]])
#ret = get_thinning_operator_graph( img*255 )
#ans = np.array([[1,1,0,0,0],[0,1,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]])*255
#if ret != ans:
#    print("WA")