from capture_image import take_picture_ocr

import cv2
import pytesseract
from PIL import Image

take_picture_ocr()


# Function to perform OCR on an image
def ocr_on_image(image, angle):
    image.save(f'ocr_{angle}.jpg')
    return pytesseract.image_to_string(image,nice=10, lang='eng')


#img = cv2.imread('ocr.jpg', cv2.IMREAD_GRAYSCALE) 
#cv2.imwrite('ocr.jpg',img)
#thresh, bw = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)  # Apply automatic binary thresholding.

# Load your image

#thresh += 10
#thresh, bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
#cv2.imwrite('ocr.jpg', bw) 
# Dictionary to store OCR results
ocr_results = {}
# Load your image
image_path = 'ocr.jpg'  # replace with your image path
image = Image.open(image_path)
image1 = cv2.imread('ocr.jpg')
# Rotate the image every 10 degrees and perform OCR
# for angle in range(0, 360, 90):
for angle in range(0, 360, 90):
    rotated_image = image.rotate(angle)
    rotated_image_1 = image1.copy()
    rotated_image_1 = cv2.rotate(image1, angle, rotated_image_1)
    text = ocr_on_image(rotated_image,angle)
   
    print('-------------------------------------------------------------------------------------------------')
    ocr_results[angle] = text
    print(angle)
    print(text)
    print('----------------------------------------------------------------------------------------------')
# Finding the best result (customize this part as per your criteria)
# Here, the longest extracted string is considered the best result
best_angle = max(ocr_results, key=lambda k: len(ocr_results[k]))
best_result = ocr_results[best_angle]

print(f"Best result at {best_angle} degrees: \n{best_result}")
