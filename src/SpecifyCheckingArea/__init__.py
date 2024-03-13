import cv2
from datetime import datetime

# Initialize variables to store the coordinates of the selected area
top_left = None
bottom_right = None
drawing = False
original_image = None
done = False
qty: int = 0
areas = []
# Get the current date and time
current_datetime = datetime.now()

# Convert it to a string
current_datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
def draw_rectangle(event, x, y, flags, param):
    global top_left, bottom_right, drawing, original_image, done, image, qty, areas, temp

    if event == cv2.EVENT_LBUTTONDOWN:
       
        image = original_image.copy()
        temp = image.copy()
        if not drawing:
            top_left = (x, y)
            drawing = True
            cv2.rectangle(temp, top_left, top_left, (0, 255, 0), 2)
            cv2.imshow(current_datetime_str, temp)
        if drawing:
            bottom_right = (x,y)
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            cv2.imshow(current_datetime_str, image)   

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp = image.copy()
            bottom_right = (x, y)
            cv2.rectangle(temp, top_left, bottom_right, (0, 255, 0), 2)
            cv2.imshow(current_datetime_str, temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        original_image = temp.copy()
        bottom_right = (x, y)
        cv2.rectangle(original_image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.imshow(current_datetime_str, original_image)
        if top_left == bottom_right:
            done = True
            return

        print(f"Selected area coordinates: Top Left = {top_left}, Bottom Right = {bottom_right}")

        if top_left is not None:
            areas.append([top_left, bottom_right])
        qty = qty + 1
        top_left = None
        # return top_left, bottom_right
        # If you want to print the coordinates of all pixels within the area
        # print("Coordinates of all pixels within the selected area:")
        # for y in range(top_left[1], bottom_right[1]):
        #     for x in range(top_left[0], bottom_right[0]):
        #         print(f"(x, y) = ({x}, {y})")

def doNothing():
    print("test")

global first

first = True

def select(image_path):
    global top_left, bottom_right, drawing, original_image, done, image, first
    done = False
    # Read the image
    import cv2
    original_image = cv2.imread('Sources/source_image.jpg')
    image = cv2.imread('captured_image.jpg')
    # cv2.imshow(current_datetime_str, image)
    # cv2.waitKey(0)
    # Check if the window is not created, then create it
    #cv2.namedWindow(current_datetime_str, cv2.WINDOW_FULLSCREEN)
       
    cv2.startWindowThread()        
    cv2.namedWindow(current_datetime_str)

    cv2.setMouseCallback(current_datetime_str,draw_rectangle)    
    while True:
        cv2.imshow(current_datetime_str,original_image)
        if cv2.waitKey(0) & 0xFF == ord('q') or done == True:
            break
    
    first = False
  
    cv2.destroyAllWindows()
    return areas
