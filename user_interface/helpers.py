import cv2
import base64
import requests

CIRCLE_RADIUS = 4
CALIBRATION_MATRIX = [[180,162],[618,0],[552,540],[682,464]]

FACE_DET_LINK = 'http://localhost:8000/faces'
DISTANCE_LINK = 'http://localhost:8080/coordinates'

def get_persons_in_frame(frame_num, ground_truth):
	persons = []
	found = False
	while True:
		if len(ground_truth) == 0:
			break;
		person = ground_truth[0]
		if person["frame_num"] == frame_num:
			persons.append(ground_truth.pop(0))
			if not found:
				found = True
		elif found:
			return persons
		
	return persons


def format_line(line):
	return {
		"person_num": int(line[0]),
		"frame_num": int(line[1]),
		"x1": int(float(line[-4])),
		"y1": int(float(line[-3])),
		"x2": int(float(line[-2])),
		"y2": int(float(line[-1])),
	}


def get_formatted_ground_truth(file_path):
	f = open(file_path)
	trimmed_lines = list(map(lambda x: x.strip(), f.readlines()))
	splitted_lines = list(map(lambda x: x.split(","), trimmed_lines))
	structured_lines = list(map(format_line, splitted_lines))
	return structured_lines

def crop_recatngle(img, x1, y1, x2, y2):
	return img[y1:y2, x1:x2]

def get_b64_open_cv_img(img):
	retval, buffer_img= cv2.imencode('.jpg', img)
	data = base64.b64encode(buffer_img).decode('ascii')
	return data

def get_face_det_uri(rect):
	b64 = get_b64_open_cv_img(rect)
	return FACE_DET_LINK + "?img=" + b64

def get_faces_in_rectangle(img, x1, y1, x2, y2):
	try:
		rect = crop_recatngle(img, x1, y1, x2, y2)
		# cv2.imshow('frame', rect)
		r = requests.get(get_face_det_uri(rect))
		if r.status_code != 200:
			return []
		json_data = r.json()
		if json_data["success"] != True:
			return []
		return json_data["results"]
	except:
		return []

def format_person(person):
	return {
		"x1": person["x1"],
		"y1": person["y1"],
		"x2": person["x2"],
		"y2": person["y2"]
	}

def get_distance_payload(img, persons):
	formatted_bboxes = list(map(format_person, persons))
	return {
		"img": get_b64_open_cv_img(img),
		"calibration_matrix": CALIBRATION_MATRIX,
		"circle_radius": CIRCLE_RADIUS,
		"bounding_boxes": formatted_bboxes
	}

def get_distance_parameters(img, persons):
	try:
		# cv2.imshow('frame', rect)
		r = requests.post(DISTANCE_LINK, json=get_distance_payload(img,persons))
		if r.status_code != 200:
			return []
		json_data = r.json()
		if json_data["success"] != True:
			return []
		return json_data["results"]
	except:
		return []


def get_person_ellipse_color(distances, i):
	distances.pop(i)
	if len(distances) == 0:
		return (0, 255, 0, 0.5) 
	_min = min(distances)
	if _min <= 25:
		return (0, 0, 255, 0.5)
	return (0, 255, 0, 0.5)
