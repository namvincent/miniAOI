import base64
from io import BytesIO
from PIL import Image

def encode_image_to_base64(filepath):
    with Image.open(filepath) as image:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")  # Adjust format as needed
        return base64.b64encode(buffered.getvalue()).decode()

