import numpy as np
import math
from Image import Image
import cv2

class Morphology:
	def __init__(self, kernel):
		self.kernel = np.array(kernel)


	def get_padding_width(self):
		l,w = self.kernel.shape
		pad_width = (l - 1) / 2
		return int(pad_width)


	def apply_padding(self, image):
		padding_width = self.get_padding_width()
		return Image(np.pad(image.get_data(), padding_width))

	def is_full_contained(self, padded_image, c, r):
		alpha = self.get_padding_width()
		for j in range(len(self.kernel)):
			for i in range(len(self.kernel[j])):
				color = padded_image[c - alpha + j, r - alpha + i]
				if color == 0 and self.kernel[j,i] == 1:
					return False
		return True

	def is_partially_contained(self, padded_image, c, r):
		alpha = self.get_padding_width()
		for j in range(len(self.kernel)):
			for i in range(len(self.kernel[j])):
				color = padded_image[c - alpha + j, r - alpha + i]
				if color != 0 and self.kernel[j,i] == 1:
					return True
		return False

	def erode(self, image):
		padded_image = self.apply_padding(image)
		padding_width = self.get_padding_width()
		data = padded_image.get_data()
		data_clone = np.copy(data)
		img_data = image.get_data()
		for c in range(len(img_data)):
			for r in range(len(img_data[c])):
				if not self.is_full_contained(data, c, r):
					data_clone[c,r] = 0
				else:
					data_clone[c,r] = 1
		unpadded_image = data_clone[padding_width:len(data) - padding_width, padding_width:len(data[0]) - padding_width]
		return Image(unpadded_image)
		# return Image(cv2.erode(image.get_data(),self.kernel,iterations = 1))

	def dilute(self, image):
		# return Image(cv2.dilate(image.get_data(),self.kernel,iterations = 1))
		padded_image = self.apply_padding(image)
		padding_width = self.get_padding_width()
		data = padded_image.get_data()
		data_clone = np.copy(data)
		img_data = image.get_data()
		for c in range(len(img_data)):
			for r in range(len(img_data[c])):
				if self.is_partially_contained(data, c, r):
					data_clone[c,r] = 1
					
		unpadded_image = data_clone[padding_width:len(data) - padding_width, padding_width:len(data[0]) - padding_width]
		return Image(unpadded_image)

	def open(self, image):
		return self.dilute(self.erode(image))
	
	def close(self, image):
		return self.erode(self.dilute(image))



