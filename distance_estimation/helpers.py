from skimage import io
import base64
def from_b64(base64_str):
	try:
		if isinstance(base64_str, bytes):
			base64_str = base64_str.decode("utf-8")
		 
		base64_str = base64_str.replace(' ', '+')
		imgdata = base64.b64decode(base64_str)
		img = io.imread(imgdata, plugin='imageio')
		return img
	except Exception as e:
		print(e)
		return None
