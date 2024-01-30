import base64
import json
import random
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageFont, ImageDraw
import requests
from spellchecker import SpellChecker
from capture_image import capture_frame

class ResultObject:
    def __init__(self, roi, check_type, result, content=None, text=""):
        self.top_left = roi[0]
        self.bottom_right = roi[1]
        self.check_type = check_type
        self.result = result
        self.content = content
        self.text = text


def load_image(image_path):
    img = cv2.imread(image_path)
    source = cv2.imread(f"Sources/source_image.jpg")
    return img, source


def load_image_hsv(image_path):
    img = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imwrite(f"hsv_captured_image.jpg", hsv_image)

    return hsv_image


def sharpen_image(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


def correct_color(image):
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    except:
        hsv = cv2.cvtColor(cv2.cvtColor(
            image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # Adjust the saturation and value channels
    h = cv2.equalizeHist(h)
    s = cv2.equalizeHist(s)
    v = cv2.equalizeHist(v)

    enhanced_hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)


def process_image(image_path, image=None, alpha=1.5, beta=-50.0, rotate=0):
    global resized

    # Load the image
    if image is None:
        pil_img = Image.open(image_path)
        rotated_img = pil_img.rotate(rotate, expand=True)
        image = np.array(rotated_img)
    else:
        pil_img = Image.fromarray(np.array(image))
        rotated_img = pil_img.rotate(rotate, expand=True)
        image = np.array(rotated_img)

        # Median Blur
    median_blur = cv2.medianBlur(image, 1)

    try:
        gray_image = cv2.cvtColor(median_blur, cv2.COLOR_BGR2GRAY)
    except:
        gray_image = image
    # Apply edge detection (using Canny edge detector as an example)

    sharpened = sharpen_image(gray_image)

    adjusted = cv2.convertScaleAbs(sharpened, alpha=3, beta=100)

    # adjusted = gray_image
    _, thresh = cv2.threshold(adjusted, 150, 255, cv2.THRESH_BINARY)

    if thresh.shape[0] > thresh.shape[1] and thresh.shape[1] < 300:
        zoom = (300 / thresh.shape[1]) + 1
        resized = cv2.resize(thresh, (int(adjusted.shape[1] * zoom), int(thresh.shape[0] * zoom)),
                             interpolation=cv2.INTER_LINEAR_EXACT)

    if thresh.shape[1] > thresh.shape[0] and thresh.shape[0] < 300:
        zoom = (300 / adjusted.shape[0]) + 1
        resized = cv2.resize(thresh, (int(adjusted.shape[1] * zoom), int(thresh.shape[0] * zoom)),
                             interpolation=cv2.INTER_LINEAR_EXACT)

    # Gaussian Blur
    gaussian_blur = cv2.GaussianBlur(resized, (1, 1), 0)

    # Median Blur
    median_blur = cv2.medianBlur(resized, 5)

    sharpened = sharpen_image(median_blur)

    # Display the original and processed images
    # cv2.imshow('Original Image', adjusted)
    # cv2.waitKey(0)
    # cv2.imshow('Grayscale Image', gray_image)
    # cv2.waitKey(0)
    # cv2.imshow('Edge Detection', sharpened)
    # cv2.waitKey(0)
    # cv2.imshow('Threshold', thresh)
    # cv2.waitKey(0)
    # cv2.imshow('Resized', resized)
    # cv2.waitKey(0)
    # cv2.imshow('Gaussian Blur', gaussian_blur)
    # cv2.waitKey(0)
    # cv2.imshow('Median Blur', median_blur)

    # Wait for a key press and then close all windows
    # cv2.destroyAllWindows()
    return median_blur, sharpened, gray_image


def read_out_locations_need_to_be_checked(coordinate_file_path):
    areas = []
    with open(coordinate_file_path, "r") as file:
        for line in file:
            check_type = line.strip().split(",")[0]
            top_left_x: int = int(line.strip().split(",")[1])
            top_left_y: int = int(line.strip().split(",")[2])
            bottom_right_x: int = int(line.strip().split(",")[3])
            bottom_right_y: int = int(line.strip().split(",")[4])
            top_left = (top_left_x, top_left_y)
            bottom_right = (bottom_right_x, bottom_right_y)
            print((top_left_x, top_left_y))
            print((bottom_right_x, bottom_right_y))
            areas.append((check_type, [top_left, bottom_right]))
    return areas


def add_special_words_to_dictionary():
    global spell
    spell = SpellChecker(language='en', case_sensitive=False)
    spell.word_frequency.load_text_file('dictionary.txt')
    return spell


def load_partial_image(image, top_left, bottom_right):
    return image[top_left[1]: bottom_right[1], top_left[0]: bottom_right[0]]


def calculate_average_color(image):
    return np.mean(image, axis=(0, 1))


def compare_color_and_save_mask(image, source, roi, threshold=90):
    if roi is None:
        roi = ((0, 0), (image.shape[1], image.shape[0]))
    else:
        roi = roi
    top_left, bottom_right = roi
    region_width = bottom_right[0] - top_left[0]
    region_height = bottom_right[1] - top_left[1]

    # template_resized = cv2.resize(source, (region_width, region_height))
    try:
        template_hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    except:
        template_hsv = cv2.cvtColor(cv2.cvtColor(
            source, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)

    try:
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    except:
        image_hsv = cv2.cvtColor(cv2.cvtColor(
            image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)

    main_image_hsv = image_hsv[top_left[1]                               : bottom_right[1], top_left[0]: bottom_right[0]]

    avg_color_template = calculate_average_color(template_hsv)
    avg_color_main_image = calculate_average_color(main_image_hsv)
    diff = avg_color_main_image - avg_color_template

    # avg_color_template = np.uint8([[avg_color_template]])
    # avg_color_template = cv2.cvtColor(avg_color_template, cv2.COLOR_HSV2BGR)
    # avg_color_main_image = np.uint8([[avg_color_main_image]])
    # avg_color_main_image = cv2.cvtColor(avg_color_main_image, cv2.COLOR_HSV2BGR)
    # color_difference = np.linalg.norm(avg_color_template - avg_color_main_image)
    # Create a blank mask
    color_difference = np.linalg.norm(diff)

    mask = np.zeros(image.shape[:2], dtype="uint8")
    result = mask.copy()
    # If color difference is significant, fill the ROI in the mask
    if color_difference > threshold:
        result[top_left[1]: bottom_right[1],
               top_left[0]: bottom_right[0]] = 255
        cv2.rectangle(result, top_left, bottom_right, (0, 0, 255), 1)
    else:
        result[top_left[1]: bottom_right[1], top_left[0]: bottom_right[0]] = 0
        cv2.rectangle(result, top_left, bottom_right, (0, 255, 0), 1)

    mask_path = f"Results/color_{roi}.jpg"

    # Save the mask
    cv2.imwrite(mask_path, diff)

    return color_difference > threshold, roi


def check_position(image, template, area, original_image):
    full_image = original_image.copy()
    original_top_left, original_bottom_right = area
    captured_image = original_image[original_top_left[1]: original_bottom_right[1],
                                    original_top_left[0]: original_bottom_right[0]]
    # template = cv2.imread(template_path, 0)
    target_image_gray = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)
    try:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        image_gray = image

    try:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    except:
        template_gray = template

    w, h = target_image_gray.shape[::-1]
    methods = [
        "cv2.TM_CCOEFF_NORMED",
        "cv2.TM_CCORR_NORMED",
        "cv2.TM_SQDIFF_NORMED",
        "cv2.TM_CCORR",
        "cv2.TM_CCOEFF",
        "cv2.TM_SQDIFF",
    ]
    for meth in methods:
        method = eval(meth)
        result = False
        res = cv2.matchTemplate(image_gray, template_gray, method)
        _, max_val, max_loc, min_loc = cv2.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        threshold = 0.85
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            # cv2.rectangle(
            #     template, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1
            # )
            outer_top_left = (top_left[0] + pt[0], top_left[1] + pt[1])
            bottom_right = (
                outer_top_left[0] + w,
                outer_top_left[1] + h
            )

            template_crop = template[outer_top_left[1]:bottom_right[1], outer_top_left[0]:bottom_right[0]]

            similar_area = captured_image[
                outer_top_left[1]:bottom_right[1], outer_top_left[0]:bottom_right[0]
            ]

            # similar_area = cv2.cvtColor(similar_area, cv2.COLOR_GRAY2BGR)

            edge_difference, edges = compare_features(
                template_crop, similar_area, detect_edges
            )
            corner_difference, corners = compare_features(
                template_crop, similar_area, detect_corners
            )

            if edge_difference < 5 and corner_difference < 8:
                wrong_color, roi = compare_color_and_save_mask(
                    similar_area, template_crop, None, 90
                )
                if wrong_color:
                    color = (244, 187, 88)
                else:
                    color = (0, 255, 0)
            else:
                wrong_color, roi = compare_color_and_save_mask(
                    similar_area, template_crop, None, 90
                )
                if wrong_color:
                    color = (0, 0, 255)
                else:
                    color = (244, 187, 88)

            if edge_difference < 5 and corner_difference < 10 and not wrong_color:
                result = True
                cv2.imwrite(
                    f"Results/similar_area_{meth}_{area}.jpg", similar_area)
                cv2.rectangle(
                    full_image, original_top_left, original_bottom_right, color, 1
                )
                cv2.imwrite(f"Results/full_image_{area}.jpg", full_image)
                break
        if result:
            return False

        cv2.rectangle(
            full_image, original_top_left, original_bottom_right, (
                0, 0, 255), 1
        )
        cv2.imwrite(f"Results/full_image_{area}.jpg", full_image)
        return True


def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def detect_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    good_corners = cv2.goodFeaturesToTrack(gray, 200, 0.01, 5)
    good_corners = np.intp(good_corners)
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    return good_corners, corners


def compare_features(image1, image2, feature_detector):
    if feature_detector == detect_edges:
        # Create images to visualize the edges and corners
        features1 = feature_detector(image1)
        features2 = feature_detector(image2)
        image = image2.copy()
        image[features2 > 0.01 * features2.max()] = [0, 0, 0]
    else:
        features1, corners1 = feature_detector(image1)
        features2, corners2 = feature_detector(image2)
        # Create images to visualize the edges and corners
        image = image2.copy()
        image[corners2 > 0.01 * corners2.max()] = [0, 0, 0]
    # Here you can add your logic to compare the features
    # For simplicity, just comparing the count of features
    if len(features1) == 0:
        difference = 555
    else:
        total_f1 = len(features1)
        difference = abs(total_f1 - len(features2))

    return difference, image


def find_best_ocr_result(ocr_results):
    best_result = None
    best_score = float("-inf")
    for result in ocr_results:
        words = result.strip().split()
        misspelled = spell.unknown(words)
        correct_words = len(words) - len(misspelled)
            # Score can be calculated as the number of correct words
        if correct_words > best_score:
            best_score = correct_words
            best_result = result
    return best_result, best_score


def find_best_ocr(list_ocr):
    best_result = None
    best_score = float("-inf")
    line_result = ''
    line_best_score = 0
    for result in filter(lambda x: x != '', list_ocr):
        line_result = ''
        line_best_score = 0
        for words in result:
            misspelled = spell.unknown(words)
            correct_words = len(words) - len(misspelled)
            line_best_score = line_best_score + correct_words
            line_result = f'{line_result}{correct_words}'
    # Score can be calculated as the number of correct words
    if line_best_score > best_score:
        best_score = line_best_score
        best_result = line_result
    print(f'{best_result}:{best_score}')
    return best_result


def find_best_of_best(bests):
    best_result = None
    best_score = float("-inf")

    for best, score in filter(lambda x: x[0] is not None, bests):
        correct_words = 0
        for line in best:
            words = line.strip()
            misspelled = spell.unknown(words)
            correct_words = len(words) - len(misspelled)

            # Score can be calculated as the number of correct words
        if correct_words > best_score:
            best_score = correct_words
            # best_result = spell.correction(best)
    return best_result


def write_text_on_image(
    img,
    text,
    text_position,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1,
    font_color=None,
    thickness=1,
    line_type=cv2.LINE_AA,
):
    """
    Write text on an image at the specified position.

    Args:
    - image_path: Path to the input image.
    - text: Text string to be written on the image.
    - text_position: Tuple (x, y) representing the position of the text.
    - font_face: Font type.
    - font_scale: Font scale (size).
    - font_color: Font color in BGR.
    - thickness: Thickness of the font.
    - line_type: Type of the line used.
    """

    # Load the image
    if img is None:
        print("Error: Image not found")
        return None

    # Put text on the image
    cv2.putText(
        img,
        text,
        text_position,
        font_face,
        font_scale,
        font_color,
        thickness,
        line_type,
    )

    return img


def add_unicode_text_to_image(
        image_cv, text, position, font_path, font_size, text_color=None, bg_color=(0, 0, 0)
):
    if text_color is None:
        text_color = random_color()
    """
    Adds Unicode text with a background to an image at the specified position.

    :param image_path: Path to the image file.
    :param text: Unicode text to be added.
    :param position: Tuple (x, y) specifying the position to add the text.
    :param font_path: Path to a .ttf font file that supports Unicode.
    :param font_size: Size of the font.
    :param text_color: Text color in RGB format (default is white).
    :param bg_color: Background color in RGB format (default is black).
    :return: Image with text as a numpy array in OpenCV format.
    """

    # Convert from OpenCV BGR to PIL RGB format
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    # Prepare to draw on the image
    draw = ImageDraw.Draw(image_pil)

    # Load the font and specify the size
    font = ImageFont.truetype(font_path, font_size)

    # Calculate text size and position
    background_area = (position[0], position[1], position[0], position[1])

    # Draw the background
    draw.rectangle(background_area, fill=bg_color)

    # Draw the text
    draw.text(position, text, font=font, fill=text_color)

    # Convert back to OpenCV format and return
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


def remove_background(image):

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an all white mask
    mask = np.ones_like(image) * 255

    # Fill the detected objects with black in the mask
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

    # Invert the mask
    mask = cv2.bitwise_not(mask)

    # Bitwise-OR mask and original image
    result = cv2.bitwise_or(image, mask)

    # Save or display the result
    cv2.imwrite(f'{contours}-image_no_background.jpg', result)
    return result


def combine_images(image1, image2, horizontal=True):
    """
    Combine two images side by side (horizontal) or one above the other (vertical).

    Args:
    - image_path1: Path to the first image.
    - image_path2: Path to the second image.
    - horizontal: Combine images horizontally if True, vertically otherwise.
    """

    # Read the images
    # image1 = cv2.imread(image_path1)
    # image2 = cv2.imread(image_path2)

    # Ensure images are read successfully
    if image1 is None or image2 is None:
        print("Error loading one or both images")
        return None

    if horizontal:
        # Combine images horizontally
        # The height of the new image is the max height of the two images
        new_height = max(image1.shape[0], image2.shape[0])

        # Resize images to the same height
        image1 = cv2.resize(
            image1, (int(image1.shape[1] *
                     new_height / image1.shape[0]), new_height)
        )
        image2 = cv2.resize(
            image2, (int(image2.shape[1] *
                     new_height / image2.shape[0]), new_height)
        )

        # Combine images
        combined_image = np.hstack((image1, image2))
    else:
        # Combine images vertically
        # The width of the new image is the max width of the two images
        new_width = max(image1.shape[1], image2.shape[1])

        # Resize images to the same width
        image1 = cv2.resize(
            image1, (new_width, int(
                image1.shape[0] * new_width / image1.shape[1]))
        )
        image2 = cv2.resize(
            image2, (new_width, int(
                image2.shape[0] * new_width / image2.shape[1]))
        )

        # Combine images
        combined_image = np.vstack((image1, image2))

    return combined_image


def insert_image_into_white_base(image1, position, width=400, height=400, bw=False):
    # Create a blank 300x300 black image

    base_image = np.zeros((width, height, 3), dtype="uint8")
    base_image[:] = (255, 255, 255)
    # Load another image (replace with the path to your image)
    # Ensure the image to insert is smaller than 300x300
    insert_image = image1  # Replace with the correct path
    insert_height, insert_width = insert_image.shape[:2]

    # Coordinates where to insert the image on the base image
    x_offset = int(width/2)-int(insert_width/2)  # Example x coordinate
    y_offset = 30  # Example y coordinate

    # Insert the image
    base_image[y_offset:y_offset + insert_height,
               x_offset:x_offset + insert_width] = insert_image
    if bw:
        bw_img = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('base_image_current.jpg', bw_img)
    final_img = cv2.imread('base_image_current.jpg')

    return final_img


def random_color():
    rgbl = [0, 0, 0]
    random.shuffle(rgbl)
    return tuple(rgbl)


def sub_thread(image_path, source_path, image, source_image, checking_areas):
    final_result_image = image.copy()
    final_source_image = source_image.copy()

    final_rs = []

    rs, data = calculate_async(
        image_path,
        source_path,
        image,
        source_image,
        final_result_image,
        final_source_image,
        checking_areas,
        final_rs,
    )

    url = "http://10.100.10.83:5000/api/VisualIspection/QD/InsertVisualData"

    payload = json.dumps(data)

    payload = json.dumps(
        {
            "barcode": "334",
            "status": 0,
            "orderNo": "1009574",
            "line": "string",
            "resultData": json.loads(payload),
        }
    )

    # print(payload)
    headers = {"Content-Type": "application/json"}

    # response = requests.post(url, data=payload, headers=headers)
    # cv2.imshow("Result", cv2.WINDOW_FULLSCREEN)
    # cv2.imshow("Result", cv2.imread("Results/result.jpg"))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(response)


def calculate_async(
        image_path,
        source_path,
        image,
        source_image,
        final_result_image,
        final_source_image,
        checking_areas,
        rs,
):
    final_result = []
    final_data = []
    rs = []
    final_source_image = final_result_image
    partial_area_image = final_result_image
    for checking_type, item in checking_areas:
        visual_inspection_result = ""
        visual_inspection_result_color = None
        bests = []
        final_rs = [ResultObject]
        result = False
        checking_content = ""
        source_text = ""
        top_left, bottom_right = item
        # source_top_left: int = int(top_left-(5,5))
        # source_top_left_y: int = int(item.strip().split(',')[1]-5)
        # source_bottom_right_x: int = int(item.strip().split(',')[2]+5)
        # source_bottom_right_y: int = int(item.strip().split(',')[3]+5)
        offset = 2
        source_top_left = (
            max(0, top_left[0] - offset), max(0, top_left[1] - offset))
        source_bottom_right = (
            min(image.shape[1], bottom_right[0] + offset),
            min(image.shape[0], bottom_right[1] + offset),
        )
        # hsv_partial_image = load_partial_image(hsv_image, top_left, bottom_right)
        # hsv_partial_path = 'Sources/hsv_partial_image.jpg'
        # cv2.imwrite(hsv_partial_path, hsv_partial_image)
        partial_image = load_partial_image(image, top_left, bottom_right)
        partial_path = "Sources/partial_image.jpg"
        cv2.imwrite(partial_path, partial_image)
        partial_source_image = load_partial_image(
            source_image, top_left, bottom_right)
        partial_source_path = "Sources/partial_source_image.jpg"
        cv2.imwrite(partial_source_path, partial_source_image)
        partial_area_image = load_partial_image(
            image, source_top_left, source_bottom_right
        )
        partial_area_path = "Sources/partial_area_image.jpg"
        cv2.imwrite(partial_area_path, partial_area_image)

        # partial_area_image = cv2.GaussianBlur(partial_area_image, (5, 5), 0)
        # partial_source_image = cv2.GaussianBlur(partial_source_image, (5, 5), 0)
        # partial_image = cv2.GaussianBlur(partial_image, (5, 5), 0)
        # image = cv2.GaussianBlur(image, (5, 5), 0)
        if checking_type == "c":
            wrong_color, roi = compare_color_and_save_mask(
                image, partial_source_image, item, 90
            )
            top_left, bottom_right = roi

            # wrong_color = is_similar(image, source_image)
            # wrong_color, color_mask = check_wrong_color(partial_image, red_color_ranges)
            print(f"Wrong Color: {wrong_color}")
            if not wrong_color:
                result = True
                final_color = (0, 255, 0)
            else:
                result = False
                final_color = (0, 0, 255)
            final_result.append(result)
            cv2.rectangle(partial_area_image, (0, 0), (5, 5), final_color, -1)
            cv2.rectangle(final_result_image, top_left,
                          bottom_right, final_color, 2)
        else:
            partial_area_image = remove_background(partial_area_image)
            partial_source_image = remove_background(partial_source_image)

            wrong_position = check_position(
                partial_area_image, partial_source_image, item, image
            )
            if not wrong_position:
                result = True
                final_color = (0, 255, 0)
            else:
                result = False
                final_color = (0, 0, 255)
            final_result.append(result)
            cv2.rectangle(partial_area_image, (0, 0), (5, 5), final_color, -1)
            cv2.rectangle(final_result_image, top_left,
                          bottom_right, final_color, 2)
            # if result:
            #     best_result = ''
            #     ocr_results = []

            #     rotated_image, partial_image_edges, gray = process_image(partial_path, partial_image, rotate=270)
            #     cv2.imwrite(f"Sources/area-{item}.jpg", rotated_image)
            #     text = pytesseract.image_to_string(
            #         Image.fromarray(rotated_image), lang='eng+fra+deu+vie+symbols+numbers'
            #     )
            #     ocr_results.append(text)
            #     best_result, score = find_best_ocr_result(ocr_results)
            #     bests.append((best_result, score))

            #     best_result = find_best_of_best(bests)
            #     # rotated_source_image, partial_source_edges = process_image(partial_source_path)

            #     # ocr_source_results = []

            #     # for angle in range(0, 360, 90):
            #     #     bests = []
            #     #     rotated_image, partial_image_edges, gray = process_image(partial_path)
            #     #
            #     #     alpha = 1.0
            #     #     beta = 0
            #     #     rotated_image = cv2.rotate(rotated_image, angle)
            #     #     #rotated_source_image = cv2.rotate(rotated_source_image, angle)
            #     #     text = pytesseract.image_to_string(Image.fromarray(rotated_image), lang='eng+fra+deu+vie+symbols+numbers')
            #     #
            #     #     adjust_time = 10
            #     #     while ocr_results is [] and adjust_time > 0:
            #     #         alpha = alpha + 0.1
            #     #         beta = beta - 10
            #     #         print(f'{angle}:{adjust_time}:{alpha}:{beta}')
            #     #         adjust_time -= 1
            #     #         rotated_image, partial_image_edges, gray = process_image(partial_path,alpha,beta)
            #     #         text = pytesseract.image_to_string(Image.fromarray(rotated_image), lang='eng+fra+deu+vie+symbols+numbers')
            #     #         ocr_results.append(text)
            #     #
            #     #         if not ocr_results:
            #     #             best_result, score = find_best_ocr_result(ocr_results)
            #     #             bests.append((best_result, score))
            #     #             print(f'{best_result}:{score}')
            #     #     if not bests:
            #     #         best_result = find_best_of_best(bests)

            #     checking_content = best_result
        final_result_image = add_unicode_text_to_image(
            final_result_image,
            str(checking_content),
            position=bottom_right,
            font_path="Fonts/TitilliumWeb-Italic.ttf",
            font_size=30,
            text_color=(0, 0, 255)
        )
        _, encoded_image = cv2.imencode(".jpg", partial_area_image)
        image_bytes = encoded_image.tobytes()
        tmp = {
            "topLeft": f"{item[0][0]},{item[0][1]}",
            "bottomRight": f"{item[1][0]},{item[1][1]}",
            "checkType": checking_type,
            "result": str(result),
            "finalResultImage": base64.b64encode(image_bytes).decode(),
            # 'finalResultImage': '',
            "checkingContent": checking_content,
        }
        final_data.append(tmp)
        cv2.imwrite(f"Results/{item}-result.jpg", partial_area_image)

    if False in final_result:
        print("Defected")
        visual_inspection_result = "FAIL"
        visual_inspection_result_color = (255, 0, 0)

    else:
        print("Similar")
        visual_inspection_result = "PASS"
        visual_inspection_result_color = (0, 255, 0)

    final_result_image_final = add_unicode_text_to_image(
        final_result_image,
        visual_inspection_result,
        (0, 0),
        font_path="Fonts/TitilliumWeb-Bold.ttf",
        font_size=90,
        text_color=visual_inspection_result_color,
    )
    cv2.imwrite(f"Results/result.jpg", final_result_image_final)
    return final_result, final_data
    # print(rs)
    # cv2.imshow(f'Result', )

def aoi():
    # pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
    # for i in range(0, 5):
    image_path = f"captured_image.jpg"
    source_path = f"Sources/source_image.jpg"
    coordinate_file_path = "coordinate.txt"

    final_result = []
    image, source_image = load_image(image_path)

    # Define acceptable color ranges in HSV (e.g., shades of green)
    red_color_ranges = [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([170, 100, 100]), np.array([180, 255, 255]))
        # Example HSV range
        # Add more ranges if needed
    ]
    if image is None:
        print("Could not read input image")
        final_result.append(False)
        return
    if source_image is None:
        print("Could not read source image")
        final_result.append(False)
        return
    if image.shape != source_image.shape:
        print("Input and source images must be of the same size")
        final_result.append(False)
        pass
    ocr_final = []
    ocrs = ''
    checking_areas = read_out_locations_need_to_be_checked(coordinate_file_path)
    ocr_result_final = []
    for area in filter(lambda x: x[0] == 's', checking_areas):
        Image.open(image_path).crop((area[1][0][0], area[1][0][1], area[1][1][0], area[1][1][1])).rotate(
            270, expand=True).save('rotated_image.jpg')
        img_ocr = cv2.imread(f'rotated_image.jpg')
        img_ocr = remove_background(img_ocr)
        img_resized = insert_image_into_white_base(
            img_ocr, (0, 0), 300, 300, True)

        thresh = 60
        text = ''
        ocr_result = []
        while thresh < 180:
            # print(area, thresh)
            thresh += 10
            _, img_adjust = cv2.threshold(
                img_resized, thresh, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow('', img_adjust)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            img = Image.fromarray(img_adjust)

            text = pytesseract.image_to_string(img)
            # print(text)
            if text == '' or text is None:
                continue
            print(text)
            ocr_result.append(text)
        ocr = find_best_ocr_result(ocr_result)
        ocr_result_final.append(ocr)
    with open('ocr_result.txt', 'w') as inputFile:
        for item in ocr_result_final:
            inputFile.write(item[0])
            inputFile.write('\n')
    sub_thread(image_path, source_path, image, source_image, checking_areas)


global spell
global content_tex
if __name__ == "__main__":
    contents = []
    spell = add_special_words_to_dictionary()
    with open('ocr.txt') as file:
        content_text = file.read()
        for i in content_text.splitlines:
            if i is not None and i != '':
                spell.word_frequency.add(i)
                contents.append(i)
                print(contents)
   # capture_frame(False)
    aoi()
# cv2.waitKey(0)
# cv2.destroyAllWindows()
