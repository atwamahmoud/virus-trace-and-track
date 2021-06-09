import numpy as np
# import base64

## Constants
POSITIVE_CLASSIFICATION = 1
NEGATIVE_CLASSIFICATION = 0
NUM_THREADS = 8


## Helper Functions

def get_integral_image(original_image):
  integral_image = np.zeros(original_image.shape)
  row_sum = np.zeros(original_image.shape)
  for y in range(len(original_image)):
    for x in range(len(original_image[y])):
      if y > 0:
        row_sum[y][x] = row_sum[y-1][x]
      
      row_sum[y][x] = row_sum[y][x] + original_image[y][x]

      if x > 0:
        integral_image[y][x] = integral_image[y][x-1]
      

      integral_image[y][x] = integral_image[y][x] + row_sum[y][x]
  return integral_image

def create_rectangular_region(x, y, width, height):
  return {
      "x": x,
      "y": y,
      "width": width,
      "height": height
  }

def coumpute_feature(rectangular_region, integral_image):
  x,y,width,height = [rectangular_region[k] for k in ('x', 'y', 'width', 'height')]
  bottom_left = integral_image[y+height][x]
  top_right = integral_image[y][x+width]
  bottom_right = integral_image[y+height][x+width]
  top_left = integral_image[y][x]
  return top_left + bottom_right - top_right - bottom_left 


def get_applied_feature(integral_image, feature):
    pos_regions, neg_regions = feature
    sum_pos = sum([coumpute_feature(pos_region, integral_image) for pos_region in pos_regions])
    sum_neg = sum([coumpute_feature(neg_region, integral_image) for neg_region in neg_regions])
    return sum_pos - sum_neg

def is_b64_string_img(img_string):
  try:
    # TODO: Add base64 format validation...
    return img_string.startswith("data:image")
  except Exception as e:
    print(e)
    return False
