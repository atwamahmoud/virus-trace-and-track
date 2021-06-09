from wsgiref.simple_server import make_server
import os
from contextlib import redirect_stderr
import falcon
from FacesResource import FacesResource
from ViolaJones import ViolaJones
from WeakClassifier import WeakClassifier



app = falcon.App()

app.add_route('/faces', FacesResource())

# with redirect_stderr(open(os.devnull, "w")):
with make_server('', 8000, app) as httpd:
	httpd.serve_forever()

