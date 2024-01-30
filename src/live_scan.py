import cv2
from pyzbar.pyzbar import decode

def decode_barcodes(frame):
    barcodes = decode(frame)
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type
        text = f"{barcode_data} ({barcode_type})"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Initialize the video stream
cap = cv2.VideoCapture(0)# Adjust the capture device if needed
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 100)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            #print('No video output')
            continue

        frame = decode_barcodes(frame)
        cv2.imshow('Barcode Scanner', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
