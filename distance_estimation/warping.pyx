import numpy as np

def warp_perspective(img, transformation_matrix, image_dimensions):
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

