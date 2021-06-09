from wsgiref.simple_server import make_server
import os
from contextlib import redirect_stderr
import falcon
from CoordinatesResource import CoordinatesResource



app = falcon.App()

app.add_route('/coordinates', CoordinatesResource())

# with redirect_stderr(open(os.devnull, "w")):
with make_server('', 8000, app) as httpd:
	httpd.serve_forever()

