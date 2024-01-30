import cv2
import numpy as np
from capture_image import take_picture, capture_frame

import cv2


def write_text_on_image(img, text, text_position, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1,
                        font_color=(255, 255, 255), thickness=2, line_type=cv2.LINE_AA):
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
    cv2.putText(img, text, text_position, font_face, font_scale, font_color, thickness, line_type)

    return img


def load_image(image_path):
    """
    Load an image from a given path.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}")
    return image


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
        image1 = cv2.resize(image1, (int(image1.shape[1] * new_height / image1.shape[0]), new_height))
        image2 = cv2.resize(image2, (int(image2.shape[1] * new_height / image2.shape[0]), new_height))

        # Combine images
        combined_image = np.hstack((image1, image2))
    else:
        # Combine images vertically
        # The width of the new image is the max width of the two images
        new_width = max(image1.shape[1], image2.shape[1])

        # Resize images to the same width
        image1 = cv2.resize(image1, (new_width, int(image1.shape[0] * new_width / image1.shape[1])))
        image2 = cv2.resize(image2, (new_width, int(image2.shape[0] * new_width / image2.shape[1])))

        # Combine images
        combined_image = np.vstack((image1, image2))

    return combined_image


def calculate_color_difference(item, image1, image2):
    result_final = []
    final_image = None
    top_left, bottom_right = item

    area_source = image2[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    # Calculate the mean color in the area
    # mean_color_source = np.mean(area_source, axis=(0, 2))

    area = image1[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    if area.shape != area_source.shape:
        print("Images do not have the same dimensions")
        return None

    # Calculate the absolute difference
    difference = cv2.absdiff(area, area_source)

    # Convert to grayscale for easier visualization
    gray_difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    # Threshold the difference to highlight significant changes
    _, thresholded_diff = cv2.threshold(gray_difference, 40, 255, cv2.THRESH_BINARY)
    non_zero_count = np.count_nonzero(thresholded_diff)
    total_pixels = thresholded_diff.size
    diff_ratio = non_zero_count / total_pixels
    print(f"Number of non-zero pixels: {non_zero_count}")
    print(f"Total number of pixels: {total_pixels}")
    print(f"Ratio of different pixels: {diff_ratio}")
    print(f"Percentage of different pixels: {diff_ratio * 100}%")

    color_result = compare_color(item)
    general_result = color_result
    result, final_image = process_and_mark(area, area_source, item, image1, general_result)
    general_result = result & color_result
    cv2.imwrite(f"Results/result_{item}.jpg", image1)

    # Display the result
    display_images(area, area_source, thresholded_diff)
    cv2.imshow("Comparison Result", final_image)

    return general_result, image1


def compare_color(item):
    source_path = 'Sources/source_image.jpg'
    source = cv2.imread(source_path)

    result_final = []
    result_final_combined = False

    top_left, bottom_right = item
    for i in range(0, 1):
        step_result = []
        image_path = f'captured_image.jpg'
        image = cv2.imread(image_path)

        area_source = source[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Calculate the mean color in the area
        mean_color_source = np.mean(area_source, axis=(0, 2))

        area = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        # Calculate the mean color in the area
        mean_color = np.mean(area, axis=(0, 2))
        # Calculate the difference between mean color and reference color
        color_diff = np.abs(mean_color - mean_color_source)
        # Compare
        if np.average(color_diff < 30):  # The threshold of 30 is arbitrary and may need to be adjusted
            print(
                f"{image_path}: The mean color in the area {mean_color}, which is close to the reference "
                f"color {mean_color_source}.")
            print("PASS")
            result = image.copy()
            cv2.rectangle(result, top_left, bottom_right, (0, 255, 0), 1)
            result_final.append((i, "PASS"))
            result_final_combined = True
        else:
            print(f"{image_path}: The mean color in the area is {mean_color}, which is NOT close to the "
                  f"reference color {mean_color_source}.")
            print("FAIL")
            result = image.copy()
            cv2.rectangle(result, top_left, bottom_right, (0, 0, 255), 1)
            result_final.append((i, "FAIL"))
            result_final_combined = False
        cv2.imwrite(f"Results/result{item}.jpg", result)

    # result_final.sort(key=lambda x: x[0])
    # fail_list = filter(lambda x: x[1] == "FAIL", result_final)
    # if fail_list is None:
    #     result_final_combined = True
    # else:
    #     result_final_combined = False
    print(f'COLOR_RESULT: {result_final_combined}')
    return result_final_combined


# Global variables
roi = []  # List to store the coordinates of the ROI
cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0


def select_roi(event, x, y, flags, param):
    """
    Callback function to capture the mouse events and define the ROI.
    """
    global roi, cropping, x_start, y_start, x_end, y_end

    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x, y = x, y, 0, 0
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        x_end, y_end = x, y
        cropping = False
        roi = [(x_start, y_start), (x_end, y_end)]


def is_similar(region1, region2):
    mean_color = np.mean(region1, axis=(0, 2))
    mean_color_source = np.mean(region2, axis=(0, 2))
    color_diff = np.average(mean_color - mean_color_source)
    rs = color_diff <= 30
    return rs


def process_and_mark(image1, image2, area, captured, color_result):
    # Assuming both images are of the same size
    height, width, _ = image1.shape
    top_left, bottom_right = area  # Size of the region to compare

    region1 = image1[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    region2 = image2[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    if region1 is None or region2 is None:
        final_result = False
        return None
    shape_result = is_similar(region1, region2)
    if color_result:
        color = (0, 255, 0)
        final_result = True
        text = 'PASS'
        # Green for similar regions
    else:
        color = (0, 0, 255)
        final_result = False
        text = 'FAIL'
        # Red for different regions

    cv2.rectangle(captured, top_left, bottom_right, color, 1)
    write_text_on_image(captured, text, bottom_right, font_color=color)
    print(f'COLOR_RESULT:{color_result} SHAPE_RESULT:{shape_result} FINAL_RESULT: {final_result}')
    return final_result, captured


def display_images(image1, image2, difference):
    """
    Display two images and their difference.
    """
    img = combine_images(image1, image2)
    cv2.namedWindow("Result", cv2.WINDOW_KEEPRATIO & cv2.WINDOW_NORMAL & cv2.WINDOW_AUTOSIZE &
                    cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Result', img)
    cv2.waitKey(0)

    cv2.namedWindow("Diff", cv2.WINDOW_KEEPRATIO & cv2.WINDOW_NORMAL & cv2.WINDOW_AUTOSIZE &
                    cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Diff', difference)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(image_path1, image_path2):
    global roi
    global final_result
    global final_image
    image1 = load_image(image_path1)
    image2 = load_image(image_path2)
    final_result = []
    areas = []
    with open('coodinate.txt', 'r') as file:
        for line in file:
            top_left_x = int(line.strip().split(',')[0])
            top_left_y = int(line.strip().split(',')[1])
            bottom_right_x = int(line.strip().split(',')[2])
            bottom_right_y = int(line.strip().split(',')[3])
            top_left = (top_left_x, top_left_y)
            bottom_right = (bottom_right_x, bottom_right_y)
            print((top_left_x, top_left_y))
            print((bottom_right_x, bottom_right_y))
            areas.append([top_left, bottom_right])
            result, final_image = calculate_color_difference([top_left, bottom_right], image1, image2)
            final_result.append(result)
        if areas is None:
            print("No ROI defined")
            return
    position = (50, 50)
    cv2.imwrite('Results/final_image.jpg', final_image)
    cv2.imshow('Final result', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# take_picture('captured_image.jpg')
capture_frame(False)
# Paths to the images
path1 = 'captured_image.jpg'
# path1 = 'Sources/source_image.jpg'
path2 = 'Sources/source_image.jpg'

# Run the main function
main(path1, path2)
