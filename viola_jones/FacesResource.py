import falcon
from Classifier import Classifier
import json

class FacesResource:
    
    def on_get(self, req, resp):
        """Handles GET requests"""
        img_b64 = req.params.get("img")
        if img_b64 == None or img_b64 == "":
            resp.status = falcon.HTTP_400
            resp.text = json.dumps({
                "success": False,
                "message": "Couldn't find img parameter"
            }, ensure_ascii=False)
            return
        img = Image.from_b64(img_b64)
        classifier = Classifier("./num_feat_200.pkl")
        results = classifier.get_faces_data(img.get_data())
        print(results)
        # Validate
        resp.status = falcon.HTTP_200  # This is the default status
        resp.text = json.dumps({
            "success": True,
            "results": results
        }, ensure_ascii=False)