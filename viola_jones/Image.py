import numpy as np
import matplotlib.pyplot as plt
import math

class Image:
	def __init__(self, image):
		self.image = np.array(image);

	def resize_nearest_neighbor(self, factor, original):
		new_len = math.floor(len(original) * factor)
		distributed = []
		for i in range(1, new_len+1):
			distributed.append(i)

		i = 0
		while i < len(distributed):
			idx = math.ceil(distributed[i] / factor)
			if factor < 1:
				idx = math.floor(distributed[i] / factor)
			distributed[i] = original[idx-1]
			i += 1

		return distributed


	def resize_custom(self, factor_x, factor_y):
		new_image = self.resize_nearest_neighbor(factor_y, self.image)
		for i in range(len(new_image)):
			new_image[i] = self.resize_nearest_neighbor(factor_x, new_image[i])

		return Image(new_image);


	def resize(self, factor):
		return self.resize_custom(factor, factor)

	def get_binary_mask(self):
		c,r,_ = self.get_data().shape
		clone = np.zeros((c,r))
		for c_idx in range(len(clone)):
			for r_idx in range(len(clone[c_idx])):
				r,g,b = self.image[c_idx,r_idx]
				_sum = r+g+b
				clone[c_idx, r_idx] = 0 if _sum == 0 else 1
		return Image(clone)

	def apply_binary_mask(self, mask):
		mask_data = mask.get_data()
		clone = np.array(np.copy(self.get_data()))
		for c_idx in range(len(clone)):
			for r_idx in range(len(clone[c_idx])):
				r,g,b = clone[c_idx,r_idx]
				_sum = r+g+b
				if mask_data[c_idx,r_idx] == 0:
					clone[c_idx, r_idx][0] = 0
					clone[c_idx, r_idx][1] = 0
					clone[c_idx, r_idx][2] = 0
		return Image(clone)

	def get_greyscale(self):
		r, g, b = self.image[:,:,0], self.image[:,:,1], self.image[:,:,2]
		_r = [0.2990 * __r for __r in r]
		_g = [0.5870 * __g for __g in g]
		_b = [0.1140 * __b for __b in b]
		gray = [(__r + __g + __b) for __r, __g, __b in zip(_r,_g,_b)]
		return Image(gray)

	def get_normalized_rgb(self):
		clone = np.array(np.copy(self.image), dtype=np.single)
		for c_idx in range(len(clone)):
			for r_idx in range(len(clone[c_idx])):
				r,g,b = clone[c_idx,r_idx]
				_sum = r+g+b
				_r = r / _sum if _sum > 0 else 0
				_g = g / _sum if _sum > 0 else 0
				_b = b / _sum if _sum > 0 else 0
				clone[c_idx,r_idx][0] = _r
				clone[c_idx,r_idx][1] = _g
				clone[c_idx,r_idx][2] = _b
		return Image(clone)

	def __get_hue(self, r,g,b,v, _min):
		if r == v:
			return 60 * (g-b) / (v - _min)
		elif g == v:
			return 2 + (60 * (b-r) / (v - _min))
		else:
			return 4 + (60 * (r-g) / (v - _min))

	def get_hsv(self):
		normal = self.get_normalized_rgb().get_data();
		clone = np.array(np.copy(normal), dtype=np.single)
		for c_idx in range(len(clone)):
			for r_idx in range(len(clone[c_idx])):
				r,g,b = clone[c_idx,r_idx]
				v = max(r,g,b)
				_min = min(r,g,b)
				s = v - min(r,g,b) if v > 0 else 0
				_sum = r+g+b
				h = self.__get_hue(r,g,b,v,_min)
				clone[c_idx,r_idx][0] = h
				clone[c_idx,r_idx][1] = s
				clone[c_idx,r_idx][2] = v
		return Image(clone)

	def get_ycbcr(self):
		clone = np.array(np.copy(self.image), dtype=np.single)
		for c_idx in range(len(clone)):
			for r_idx in range(len(clone[c_idx])):
				r,g,b = clone[c_idx,r_idx]
				_sum = r+g+b
				y = 0.299 * r + 0.587 * g + 0.114 * b
				cr = 128 + (0.5 * r) - (0.418688 * g) - (0.081312 * b)
				cb = 128 - (0.168736 * r) - (0.331264 * g) - (0.5 * b)
				clone[c_idx,r_idx][0] = y
				clone[c_idx,r_idx][1] = cb
				clone[c_idx,r_idx][2] = cr
		return Image(clone)


	def segment_skin(self):
		print("Started getting YCbCr")
		ycbcr = self.get_ycbcr().get_data()
		print("Finished getting YCbCr")
		print("Started getting HSV")
		hsv = self.get_hsv().get_data()
		print("Finished getting HSV")
		print("Started getting Normal RGB")
		normal = self.get_normalized_rgb().get_data()
		print("Finished getting Normal RGB")
		print("Started Segmenting")
		clone = np.array(np.copy(self.image), dtype=np.uint8)
		for c_idx in range(len(clone)):
			for r_idx in range(len(clone[c_idx])):
				r,g,b = normal[c_idx,r_idx]
				H,S,V = hsv[c_idx,r_idx]
				Y,Cb,Cr = ycbcr[c_idx,r_idx]
				if not (((r / g) > 1.185) and
					((H >= 0) and (H <= 25)) or
					((H >= 335) and (H <= 360)) and
					((S >= 0.2) and (S <= 0.6)) and
					((Cb > 77) and (Cb < 127)) and
					((Cr > 133) and (Cr < 173))):
					clone[c_idx,r_idx][0] = 0
					clone[c_idx,r_idx][1] = 0
					clone[c_idx,r_idx][2] = 0
		print("Finished Segmenting")
		return Image(clone)

	def get_dimensions(self):
		h,w,_ = self.image.shape
		return w,h

	def get_data(self):
		return self.image


	def show_greyscale(self):
		plt.figure("Grey scale")
		plt.imshow(self.get_greyscale().image, cmap='gray', vmin=0, vmax=255)
		plt.show()

	def crop(self, xmin, xmax, ymin, ymax):
		img = self.image[ymin:ymax+1, xmin:xmax+1]
		return Image(img)

	def show(self):
		plt.figure("RGB")
		plt.imshow(self.image, cmap='gray')
		plt.show()




