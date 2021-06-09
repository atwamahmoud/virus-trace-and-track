import falcon
from Classifier import Classifier
from Image import Image
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
        img = body.get('img')
        bounding_boxes = body.get('bounding_boxes')
        calibration = body.get('calibration_matrix')
        formatted_bounding_boxes = []
        for bounding_box in bounding_boxes:
            """
                {
                    x: int,
                    y: int,
                    w: int,
                    h: int,
                }
            """
            x = bounding_box.get('x')
            y = bounding_box.get('y')
            w = bounding_box.get('w')
            h = bounding_box.get('h')
            formatted_bounding_boxes.append([x, y, x+w, y+h])
        img = from_b64(img_b64)
        results = distance.get_results(img, calibration, formatted_bounding_boxes)

        # Should save bounding box locations in a db....

        # Validate
        resp.status = falcon.HTTP_200  # This is the default status
        resp.text = json.dumps({
            "success": True,
            "results": results
        }, ensure_ascii=False)