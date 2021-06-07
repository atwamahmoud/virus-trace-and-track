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

	def is_face(self, image):
		w,h = image.get_dimensions()
		W,H = ViolaJones.INPUT_DIMENSIONS()
		y = 0
		x = 0
		data = image.get_greyscale().get_data()
		confidence = self.__viola_jones.classify(data)
		return math.ceil(confidence * 100) / 100 >= 0.45

	def calc_circularity(self, w,h):
		return (w * h) / ((w * 2 + h * 2) ** 2)

	def get_crops(self, thresholded_image, img_object, scale):
		crops = []
		data = thresholded_image.to_uint8().get_data()
		cnts, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for c in cnts:
			x,y,w,h = cv2.boundingRect(c)
			#calculate circulatiy
			c = self.calc_circularity(w,h)
			if c <= 0.05:
				continue
			crops.append({
	    	"image": img_object.crop(x, x+w, y, y+h),
	    	"x": x,
	    	"w": w,
	    	"y": y,
	    	"h": h,
	    	"scale": scale
	    })
		return crops

	def is_face_crop(self, crop):
		image = crop.get("image")
		data = image.get_data()
		h,w,_ = data.shape
		resized = image.resize_custom(19/w, 19/h)
		resized_shape = [x for x in resized.get_data().shape]
		if len(resized_shape) != 3 or resized_shape[0] != 19 or resized_shape[1] != 19:
			return False
		return self.is_face(resized)

	def get_line(self, xmin, ymin, xmax, ymax, c_len, r_len):
		rr, cc = draw.line(xmin, ymin, xmax, ymax)
		rr = list(filter(lambda r: r < r_len, rr));
		cc = list(filter(lambda c: c < c_len, cc));
		_min = min(len(rr), len(cc))
		return rr[0:_min],cc[0:_min]

	def get_faces(self, img_data, scale):
		image_object = Image(img_data).resize(scale)
		segmented_image = image_object.segment_skin()
		binary_mask = segmented_image.get_binary_mask()
		blurred = binary_mask.apply_gaussian_binary(sigma = 2.5)
		# binary_mask.show()
		# blurred.show()
		thresholded = blurred.threshold_float(thresh=0.5)
		masked = image_object.apply_binary_mask(thresholded)
		# thresholded.show()
		crops = self.get_crops(thresholded, masked, scale)
		face_crops = []
		for crop in crops:
			if self.is_face_crop(crop):
				face_crops.append(crop)
		return face_crops

	def show_faces(self, img_data, scale = 1):
		faces = self.get_faces(img_data, scale)
		for face in faces:
			x = math.floor(face.get("x") / scale)
			w = math.floor(face.get("w") / scale)
			y = math.floor(face.get("y") / scale)
			h = math.floor(face.get("h") / scale)
			rr1, cc1 = self.get_line(x, y, x+w, y, len(img_data), len(img_data[0]))
			rr2, cc2 = self.get_line(x, y, x, y+h, len(img_data), len(img_data[0]))
			rr3, cc3 = self.get_line(x+w, y+h, x+w, y, len(img_data), len(img_data[0]))
			rr4, cc4 = self.get_line(x+w, y+h, x, y+h, len(img_data), len(img_data[0]))
			img_data[cc1,rr1] = [255, 0, 0]
			img_data[cc2,rr2] = [255, 0, 0]
			img_data[cc3,rr3] = [255, 0, 0]
			img_data[cc4,rr4] = [255, 0, 0]
		# Image(img_data).show()
