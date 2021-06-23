import cv2
import time
import helpers
import numpy as np
VID_WIDTH = 1280
VID_HEIGHT = 720


DATA_PATH = "./OxfordTownCentre"
VIDEO_NAME = "TownCentreXVID.mp4"
GROUND_TRUTH_FILE_NAME = "TownCentre-groundtruth.top"

def get_file_path(file_name):
	return DATA_PATH + "/" + file_name

cap = cv2.VideoCapture(get_file_path(VIDEO_NAME))

ground_truth = helpers.get_formatted_ground_truth(get_file_path(GROUND_TRUTH_FILE_NAME))


if cap.isOpened() == False:
	print("Error opening video stream or file")
	exit()

frame_num = 0
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

height, width = frame_height // 2, frame_width // 2

while(cap.isOpened()):
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here\
    frame = cv2.resize(frame,(width, height), interpolation=cv2.INTER_LINEAR)
    rgba_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
	
    rgba_frame[:, :, 3] = 1

    resized_frame = cv2.resize(frame,(480,270),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    persons = helpers.get_persons_in_frame(frame_num, ground_truth)

        # break;

    # exit()

    # for person in persons:
    # 	x1 = int(person["x1"] / 4)
    # 	x2 = int(person["x2"] / 4)
    # 	y1 = int(person["y1"] / 4)
    # 	y2 = int(person["y2"] / 4)
    # 	faces = helpers.get_faces_in_rectangle(resized_frame, x1, y1, x2, y2)
    # 	max_conf_face = None
    # 	for face in faces:
    # 		f_x1 = (face["x"] + x1) * 2
    # 		f_y1 = (face["y"] + y1) * 2
    # 		f_x2 = (f_x1 + face["w"] * 2) 
    # 		f_y2 = (f_y1 + face["h"] * 2)
	   #  	confidence = face["confidence"]
	   #  	cv2.rectangle(rgba_frame, (f_x1, f_y1), (f_x2, f_y2), (0,0,255, 1), thickness=1)
	   #  # get distance params...

    distance_params = helpers.get_distance_parameters(frame, persons)
    # Bird eye view...
    bird_eye_img = np.zeros((height,width, 3))

    for i in range(0, len(distance_params["bboxes"])):
        bbox = distance_params["bboxes"][i]
        person = persons[i]
        center_bird = (bbox["bird_centroid"][0], bbox["bird_centroid"][1])
        cv2.circle(bird_eye_img, center_bird, 4, (0,255,0), thickness=-1)
        cv2.putText(bird_eye_img, str(person["person_num"]), center_bird, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 1), 1, cv2.LINE_AA)
        
        ellipse_params = bbox["transformed_ellipse_params"]
        center_ellipse = (ellipse_params["r"] // 2, ellipse_params["c"] // 2)
        axes_length = (int(ellipse_params["r_radius"]), int(ellipse_params["c_radius"]))

        rgba_frame = cv2.ellipse(rgba_frame, center_ellipse, axes_length,
           0, 0, 360, helpers.get_person_ellipse_color(bbox["distances"], i), thickness=-1)


    # Doesn't combine loops to account for overlapping bounding boxes (will affect face det.)
    for person in persons:
    	x1 = int(person["x1"] / 2)
    	x2 = int(person["x2"] / 2)
    	y1 = int(person["y1"] / 2)
    	y2 = int(person["y2"] / 2)
    	cv2.rectangle(rgba_frame, (x1, y1), (x2, y2), (255,0,0, 1), thickness=1)
    	cv2.putText(rgba_frame, str(person["person_num"]), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0, 1), 1, cv2.LINE_AA)
		    

    # exit()

    # Display the resulting frame
    rgba_frame = cv2.resize(rgba_frame,(640,360),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    bird_eye_img = cv2.resize(bird_eye_img,(640,360),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('frame',  rgba_frame)
    cv2.imshow('bird eye', bird_eye_img)
    if cv2.waitKey(1) == ord('q'):
        break

    frame_num += 1


cap.release()
cv2.destroyAllWindows()