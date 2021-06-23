import falcon
from helpers import from_b64
import json
import distance



# Should be taken from sensors or reference points in our system...
calibration = [[180,162],[618,0],[552,540],[682,464]]


# The service should also provide 

class CoordinatesResource:
    
    def on_post(self, req, resp):
        """Handles POST requests"""
        body = req.get_media()
        img_b64 = body.get('img')
        bounding_boxes = body.get('bounding_boxes')
        calibration = body.get('calibration_matrix')
        circle_radius = body.get('circle_radius')
        formatted_bounding_boxes = []
        for bounding_box in bounding_boxes:
            print(bounding_box)
            x1 = bounding_box.get('x1')
            y1 = bounding_box.get('y1')
            x2 = bounding_box.get('x2')
            y2 = bounding_box.get('y2')
            formatted_bounding_boxes.append([x1, y1, x2, y2])
        img = from_b64(img_b64)
        results = distance.get_coordinates_with_distances(img, calibration, formatted_bounding_boxes, circle_radius)

        # Should save bounding box locations in a db....

        # Validate
        resp.status = falcon.HTTP_200  # This is the default status
        resp.text = json.dumps({
            "success": True,
            "results": results
        }, ensure_ascii=False)