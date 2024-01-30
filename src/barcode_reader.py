import cv2
import pyzbar.pyzbar as pyzbar
from capture_image import capture_frame


def decode(image):
    # decodes all barcodes from an image
    decoded_objects = pyzbar.decode(image)
    for obj in decoded_objects:
        # draw the barcode
        print("detected barcode:", obj)
        image = draw_barcode(obj, image)
        # print barcode type & data
        print("Type:", obj.type)
        print("Data:", obj.data)
        print("---------------------------------------")

    return image


def draw_barcode(decoded, image):
    # n_points = len(decoded.polygon)
    # for i in range(n_points):
    #     image = cv2.line(image, decoded.polygon[i], decoded.polygon[(i+1) % n_points], color=(0, 255, 0), thickness=5)
    # uncomment above and comment below if you want to draw a polygon and not a rectangle
    image = cv2.rectangle(
        image,
        (decoded.rect.left, decoded.rect.top),
        (
            decoded.rect.left + decoded.rect.width,
            decoded.rect.top + decoded.rect.height,
        ),
        color=(0, 255, 0),
        thickness=5,
    )

    return image


from glob import glob

cap = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow("Frame", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()

barcodes = glob("captured_*.jpg")
i = 0
for barcode_file in barcodes:
    # load the image to opencv
    img = cv2.imread(barcode_file)
    # decode detected barcodes & get the image
    # that is drawn
    img = decode(img)
    cv2.imwrite(f"code{i}.jpg", img)
    i += 1
    # show the image
    cv2.imshow("img", img)
    cv2.waitKey(0)
