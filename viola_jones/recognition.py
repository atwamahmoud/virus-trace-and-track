from ViolaJones import ViolaJones, WeakClassifier
from Image import Image
import math
from skimage import io, draw, filters
import pickle
from Morphology import Morphology	
import numpy as np
import cv2


class Classifier:
	def __init__(self, model_path, scales=[1]):
		self.__viola_jones = ViolaJones.load(model_path)
		self.__scales = scales

	def set_model(self, model_path):
		self.__viola_jones = ViolaJones.load(model_path)

	# def __get_resized_images(self, image_data):
	# 	resized_images = []
	# 	image = Image(image_data)
	# 	print("started resizing")
	# 	for scale in self.__scales:
	# 		resized_images.append((scale, image.resize(scale)))
	# 		print("resized image to scale %.1f" % (scale))
	# 	print("finished resizing")
	# 	return resized_images

	def is_face(self, image):
		w,h = image.get_dimensions()
		W,H = ViolaJones.INPUT_DIMENSIONS()
		y = 0
		x = 0
		faces = []
		data = image.get_greyscale().get_data()
		confidence = self.__viola_jones.classify(data)
		print(confidence)
		return confidence >= 0.4

img = io.imread("side.jpg")



image_object = Image(img).resize(0.2)
segmented_image = image_object.segment_skin()

binary_mask = segmented_image.get_binary_mask()

# morphology = Morphology(kernel = np.ones((7,7),np.uint8));


binary_mask.show()

# opened = morphology.close(binary_mask)
# opened = morphology.open(opened)
opened = filters.gaussian(binary_mask.get_data(), sigma=2.5)
for j in range(len(opened)):
	for i in range(len(opened[j])):
		opened[j,i] = 1 if opened[j,i] >= 0.4 else 0

opened = Image(opened)		

opened.show()
# opened = morphology.dilute(opened)
# # opened = binary_mask
# opened.show()

# # opened = morphology.open(opened)
# # opened = morphology.open(opened)
# opened.show()


masked = image_object.apply_binary_mask(binary_mask)


data_masked = np.uint8(opened.get_data())

## Crop Non-black stuff...


# print(len())

# data_masked = masked.get_data()
# with open("./training.pkl", "rb") as file:
# 	test = pickle.load(file)
# 	img,classification = test[20]
# 	print(img)
# 	Image(img).show_greyscale()

# # print(test)


def get_line(xmin, ymin, xmax, ymax, c_len, r_len):
	rr, cc = draw.line(xmin, ymin, xmax, ymax)
	rr = list(filter(lambda r: r < r_len, rr));
	cc = list(filter(lambda c: c < c_len, cc));
	_min = min(len(rr), len(cc))
	return rr[0:_min],cc[0:_min]

crops = []
cnts, hierarchy = cv2.findContours(data_masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
classifier = Classifier("./num_feat_200.pkl")
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    crops.append({
    	"image": image_object.crop(x, x+w, y, y+h),
    	"x": x,
    	"w": w,
    	"y": y,
    	"h": h
    })


original_data = image_object.get_data()
print(len(crops))
for crop in crops:
	image = crop.get("image")
	data = image.get_data()
	h,w,_ = data.shape
	resized = image.resize_custom(19/w, 19/h)
	resized_shape = [x for x in resized.get_data().shape]
	if len(resized_shape) != 3 or resized_shape[0] != 19 or resized_shape[1] != 19:
		continue
	if not classifier.is_face(resized):
		# resized.show()
		continue
	x = crop.get("x")
	w = crop.get("w")
	y = crop.get("y")
	h = crop.get("h")
	rr1, cc1 = get_line(x, y, x+w, y, len(original_data), len(original_data[0]))
	rr2, cc2 = get_line(x, y, x, y+h, len(original_data), len(original_data[0]))
	rr3, cc3 = get_line(x+w, y+h, x+w, y, len(original_data), len(original_data[0]))
	rr4, cc4 = get_line(x+w, y+h, x, y+h, len(original_data), len(original_data[0]))
	original_data[cc1,rr1] = [255, 0, 0]
	original_data[cc2,rr2] = [255, 0, 0]
	original_data[cc3,rr3] = [255, 0, 0]
	original_data[cc4,rr4] = [255, 0, 0]


Image(original_data).show()

# faces = classifier.get_faces(data_masked)
# print(faces)
# print(img.shape)


# for face in faces:
# 	scale = face.get('scale');
# 	print(face.get('xmin'), face.get('ymin'), face.get('xmax'), face.get('ymax'))



# Image(data_masked).show()


# img = cv2.imread('faces.jpg')
# img = cv2.medianBlur(img,5)

# # cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
# _img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# circles = cv2.HoughCircles(_img,cv2.HOUGH_GRADIENT,1,60,
#                             param1=1,param2=2,minRadius=0,maxRadius=0)


# circles = np.uint16(np.around(circles))
# for i in circles[0,:]:
#     # draw the outer circle
#     cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
#     # draw the center of the circle
#     cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

# cv2.imshow('detected circles',img)


