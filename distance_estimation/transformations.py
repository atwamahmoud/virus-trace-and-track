import numpy as np
import warping
import time
from ctypes import *

"""

    The following functions are adapted from OpenCV C++ Library

"""

def get_perspective_transform(src_rect, dest_rect):
    """ 
    Calculates coefficients of perspective transformation
    which maps (xi,yi) to (ui,vi), (i=1,2,3,4):
    
         c00*xi + c01*yi + c02
    ui = ---------------------
         c20*xi + c21*yi + c22
    
         c10*xi + c11*yi + c12
    vi = ---------------------
         c20*xi + c21*yi + c22
    
    Coefficients are calculated by solving linear system:
    | x0 y0  1  0  0  0 -x0*u0 -y0*u0 | |c00| |u0|
    | x1 y1  1  0  0  0 -x1*u1 -y1*u1 | |c01| |u1|
    | x2 y2  1  0  0  0 -x2*u2 -y2*u2 | |c02| |u2|
    | x3 y3  1  0  0  0 -x3*u3 -y3*u3 |.|c10|=|u3|,
    |  0  0  0 x0 y0  1 -x0*v0 -y0*v0 | |c11| |v0|
    |  0  0  0 x1 y1  1 -x1*v1 -y1*v1 | |c12| |v1|
    |  0  0  0 x2 y2  1 -x2*v2 -y2*v2 | |c20| |v2|
    |  0  0  0 x3 y3  1 -x3*v3 -y3*v3 | |c21| |v3|
    
    where:
      cij - matrix coefficients, c22 = 1

    """
    # Start by constructing the matrix
    A = np.zeros((8,8), dtype=np.double)
    b = np.zeros((8,1), dtype=np.double)
    
    x0,y0 = src_rect[0]
    x1,y1 = src_rect[1]
    x2,y2 = src_rect[2]
    x3,y3 = src_rect[3]
    u0,v0 = dest_rect[0]
    u1,v1 = dest_rect[1]
    u2,v2 = dest_rect[2]
    u3,v3 = dest_rect[3]

    A[:][0] = np.array([x0, y0, 1, 0, 0, 0, -x0*u0, -y0*u0])
    A[:][1] = np.array([x1, y1, 1, 0, 0, 0, -x1*u1, -y1*u1])
    A[:][2] = np.array([x2, y2, 1, 0, 0, 0, -x2*u2, -y2*u2])
    A[:][3] = np.array([x3, y3, 1, 0, 0, 0, -x3*u3, -y3*u3])
    A[:][4] = np.array([0, 0, 0, x0, y0, 1, -x0*v0, -y0*u0])
    A[:][5] = np.array([0, 0, 0, x1, y1, 1, -x1*v1, -y1*v1])
    A[:][6] = np.array([0, 0, 0, x2, y2, 1, -x2*v2, -y2*v2])
    A[:][7] = np.array([0, 0, 0, x3, y3, 1, -x3*v3, -y3*v3])
    
    b = np.array([
        [u0],[u1],[u2],[u3],[u0],[v1],[v2],[v3]])

    x = np.linalg.solve(A, b)

    # construct perspective transform matrix
    M = np.zeros((3,3), dtype=np.double)

    M[0,0] = x[0][0]
    M[0,1] = x[1][0]
    M[0,2] = x[2][0]
    M[1,0] = x[3][0]
    M[1,1] = x[4][0]
    M[1,2] = x[5][0]
    M[2,0] = x[6][0]
    M[2,1] = x[7][0]
    M[2,2] = 1.
    return M

def __warp_perspective_cython(img, transformation_matrix, image_dimensions):
    return warping.warp_perspective(img, transformation_matrix, image_dimensions)

def warp_perspective(img, transformation_matrix, image_dimensions):
    ## Uses Cython's implementation to speed things up
    res = __warp_perspective_cython(img, transformation_matrix, image_dimensions)
    return res

def __warp_perspective(img, transformation_matrix, image_dimensions):
    """ 
    Transforms an image using a transformation matrix
    which maps (xi,yi) to (ui,vi), (i=1,2,3,4):
    
         c00*xi + c01*yi + c02
    ui = ---------------------
         c20*xi + c21*yi + c22
    
         c10*xi + c11*yi + c12
    vi = ---------------------
         c20*xi + c21*yi + c22
    
    Transformation Matrix:
    |c00 c01 c02|
    |c10 c11 c12|
    |c20 c21 c22|
    
    where:
      cij - matrix coefficients, c22 = 1

    """
    height,width = image_dimensions
    transformed = np.zeros(img.shape, dtype=np.uint8)
    x = (img).nonzero()
    print(x)
    for i in range(0, height):
        for j in range(0, width):
            points = np.array([i,j,1]).dot(transformation_matrix.T)
            z = points[2]
            y = points[0]
            x = points[1]

            z = 1 if z == 0 else z
            x = int(x/z)
            y = int(y/z)
            if x >= width or y >= height:
                continue
            else:
                transformed[y,x] = img[i,j]
    return transformed

