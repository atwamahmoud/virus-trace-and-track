from Classifier import Classifier
from ViolaJones import ViolaJones, WeakClassifier
from Image import Image
import math
from skimage import io, draw, filters
import pickle
from Morphology import Morphology	
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join


path = "./Test Images/WIDER_VALIDATION/celebration"

files = [f for f in listdir(path) if isfile(join(path, f))]

classifier = Classifier("./num_feat_200.pkl")


for file in files:
	img = io.imread(join(path, file))
	obj = Image(img)
	w,h = obj.get_dimensions() 
	classifier.show_faces(obj.resize_custom(256/w, 256/h).get_data(), scale=1)

# img = io.imread("ult.jpg")




