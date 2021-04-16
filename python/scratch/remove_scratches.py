import cv2
import math
import os
import sys
import numpy as np 
from scipy.signal import convolve2d


# ============================ 
#       CE 4TN4 Project 2
#           ruigrokm
#           400157452
# ============================


# ================================================================
#                   Part 1 - Generating mask
# 
# 1. Apply canny algorithm to find edges
# 2. Use Hough transform on the edge image to find lines
# 3. Take only vertical or near-vertical lines and generate mask
#
#                   Part 2 - Removing Scratches
#
# 1. Morphologically open the mask
# 2. Use median filtering to fill in the scratches in the mask
# 3. Or use inpainting to fill scratches
#
# =================================================================

# ================================================
#         Canny Edge detection steps
#
# 1. Noise Reduction by Gaussian blurring
# 2. Intensity Gradients with 3x3 Sobel filter
# 3. Edge thinning with minimum supression
# 4. Double thresholding
# 5. Edge tracking with hysteresis
#
# ================================================

# Canny edge detector from scratch
def canny_edge_detect(image, kernel=3, weak_pixel=55, strong_pixel=220, low_thresh_ratio=0.06, high_thresh_ratio=0.19):
    blur = cv2.GaussianBlur(image, (kernel, kernel), 0)
    mag, theta = sobel_gradient(blur)
    thinned_image = min_suppression(mag, theta)
    thin = thinned_image.astype(np.uint8)
    edges = threshold_and_hysteresis(thinned_image, low_thresh_ratio=low_thresh_ratio, 
                                    high_thresh_ratio=high_thresh_ratio, weak_pixel=weak_pixel).astype(np.uint8)

    mag = mag.astype(np.uint8)
    theta = theta.astype(np.uint8)
    cv2.imwrite('theta.jpg', theta)
    cv2.imwrite('mag.jpg', mag)
    cv2.imwrite('mag_thinned.jpg', thin)
    return edges

# Sobel operator in x and y directions, returning the magnitude and direction of the gradient
def sobel_gradient(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    derivative_x = convolve2d(image, kernel_x, mode='same')
    derivative_y = convolve2d(image, kernel_y, mode='same')
    theta = np.arctan2(derivative_y, derivative_x)
    mag = np.sqrt(np.square(derivative_x) + np.square(derivative_y))
    mag *= 255.0 / mag.max()
    return (mag, theta)

# Want to suppress the images so there are thinner edges
def min_suppression(image, angle_rad):
    rows, cols = image.shape
    suppressed = np.zeros((rows, cols), dtype=np.int32)
    # Normalize the angle to between 0-180
    angle = angle_rad * 180. / np.pi
    angle[angle < 0] += 180
    # Check the pixels before and after at 0, 45, 90, 135, and 180 degrees
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            try:
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    p1, p2 = image[i, j+1], image[i, j-1]
                elif (22.5 <= angle[i,j] < 67.5):
                    p1, p2 = image[i+1, j-1], image[i-1, j+1]
                elif (67.5 <= angle[i,j] < 112.5):
                    p1, p2 = image[i+1, j], image[i-1, j]
                elif (112.5 <= angle[i,j] < 157.5):
                    p1, p2 = image[i-1, j-1], image[i+1, j+1]
                # Keep the original pixel if it is greater than both around it, else make it black
                if (image[i,j] >= p1) and (image[i,j] >= p2):
                    suppressed[i,j] = image[i,j]
                else:
                    suppressed[i,j] = 0
            except IndexError as e:
                pass

    return suppressed

# Double threshold to find the strong, weak, and zero pixels
def threshold_and_hysteresis(img, low_thresh_ratio=0.05, high_thresh_ratio=0.10,  weak_pixel=25, strong_pixel=255):
    highThreshold = img.max() * high_thresh_ratio
    lowThreshold = highThreshold * low_thresh_ratio
    rows, cols = img.shape
    threshold = np.zeros((rows, cols), dtype=np.int32)
    weak = np.int32(25)
    strong = np.int32(255)

    # Find the indices of the weak, strong, and zero pixels
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    threshold[strong_i, strong_j] = strong
    threshold[weak_i, weak_j] = weak
    
    threshold = threshold.astype(np.uint8)
    cv2.imwrite('thresh.jpg', threshold)

    # Perform hysteresis adn return edges image
    # Hysteresis function transforms a weak pixel into strong one if there is another strong pixel around
    img = threshold
    weak = weak_pixel
    stong = strong_pixel
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

# Return the mask of the vertical scratches taking the original and Canny edge detected image as input
def create_mask(image, edges, min_slope=40):
    image_copy = np.copy(image)
    mask_lines = []
    kernel = np.ones((5,1), np.uint8)
    edges_open = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    lines = cv2.HoughLinesP(image=edges_open, rho=1, theta=np.pi/180, threshold=40, 
                           lines=np.array([]), minLineLength = 30, maxLineGap=20)

    # Get the starting and end points of the lines and add to mask_lines
    # Create the mask
    rows, cols = image.shape[0], image.shape[1]
    image_mask = np.zeros([rows, cols], dtype=np.uint8)
    if lines is not None:
        for i in range(len(lines)):
            l = lines[i][0]
            if l[2] == l[0]:
                cv2.line(image_copy, (l[0], l[1]), (l[2], l[3]), 255, 1, cv2.LINE_AA)
                cv2.line(image_mask, (l[0], l[1]+10), (l[2], l[3]-10), 255, 2, cv2.LINE_AA)
            else:
                slope = abs((l[3] - l[1]) / (l[2] - l[0]))
                if slope > min_slope:
                    cv2.line(image_copy, (l[0], l[1]), (l[2], l[3]), 255, 1, cv2.LINE_AA)
                    cv2.line(image_mask, (l[0], l[1]+10), (l[2], l[3]-10), 255, 2, cv2.LINE_AA)
    else:
        print('Error: Did not detect any scratches')
    
    cv2.imwrite('edges.jpg', image_copy)
    cv2.imwrite('mask.jpg', image_mask)
    return image_copy, image_mask

# Fill the scratches in using median filtering, pass in image, mask, and the filter size
def fill_in_scratches(image, image_mask, filter_size=9):
    # Copy the image and get the image mask
    image_copy = np.copy(image)
    rows, cols = image.shape[0], image.shape[1]
    image = cv2.GaussianBlur(image, (3,3), 0)
    # Median filter the points in the mask - this is a 'poor man's' inpainting
    for i in range(rows):
        for j in range(cols):
            if image_mask[i][j] == 0:
                continue
            vals = []
            for k in range(i - filter_size // 2, i + filter_size // 2):
                for l in range(j - filter_size // 2, j + filter_size // 2):
                    if k < 0 or k >= rows or l < 0 or l >= cols:
                        continue
                    else:
                        vals.append(image[k][l])
            image_copy[i][j] = np.clip(np.median(vals), 0, 255)

    return image_copy

# For comparison - this is using openCV inpaint function
def inpaint(image, mask_lines, mask):
    image_copy = np.copy(image)
    '''
    rows, cols = image.shape[0], image.shape[1]
    image_mask = np.zeros([rows, cols, 3], dtype=np.uint8)
    for i in range(len(mask_lines)):
        x_start, y_start = mask_lines[i][1]
        x_end, y_end = mask_lines[i][0]
        cv2.line(image_mask, (x_end, y_end), (x_start, y_start), (255,255,255), 3, cv2.LINE_AA)
    '''

    kernel = np.ones((5, 1), np.uint8)
    image_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #image_copy = cv2.inpaint(image_copy, image_mask, 2, cv2.INPAINT_NS)
    image_copy = cv2.inpaint(image_copy, image_mask, 3, cv2.INPAINT_TELEA)
    return image_copy

# Test all of the given images
def run_all():
    images = [x for x in os.listdir(IMG_LOCATION) if x.endswith('input.png')]
    for i in images:
        image = cv2.imread(IMG_LOCATION + i, cv2.CV_8UC1)
        if image is None:
            print('couldn\'t load image ... skipping')
            continue
        
        #Perform canny edge detection, find scratch mask, and correct the image
        edges = canny_edge_detect(image)
        lines, mask = create_mask(image, edges,min_slope=50)
        image_final=  fill_in_scratches(image, mask, filter_size=11)
        cv2.imwrite(i.split('input')[0] + '_corrected.png', image_final)

# --------------------------------
# Parameters to adjust 
# --------------------------------

IMG_LOCATION = '..\\synthetic\\'
IMG_NAME = 'kodim09_input.png'
#IMG_LOCATION = '..\\real\\'
#IMG_NAME = 'mlts17_input.png'
FILTER_SIZE = 11
STRONG_PIXEL = 220
WEAK_PIXEL = 55
LOW_THRESH_RATIO = 0.06
HIGH_THRESH_RATIO = 0.19
GAUSS_KERNEL_SIZE = 3

# --------------------------------
#   Main program
# --------------------------------

if __name__ == "__main__":
    #run_all()
    #filename = IMG_LOCATION + IMG_NAME
    filename = 'test.jpg'
    image = cv2.imread(filename, cv2.CV_8UC1)
    if image is None:
        print('couldn\'t load image ... exiting')
        sys.exit(1)
    
    #Perform canny edge detection, find scratch mask, and correct the image
    edges = canny_edge_detect(image, kernel=GAUSS_KERNEL_SIZE, weak_pixel=WEAK_PIXEL, strong_pixel=STRONG_PIXEL,
                            low_thresh_ratio=LOW_THRESH_RATIO, high_thresh_ratio=HIGH_THRESH_RATIO)

    cv2.imwrite('canny.jpg', edges)
    print('Done Canny edge detection')
    lines, mask = create_mask(image, edges,min_slope=50)
    print('Mask created')
    image_final = fill_in_scratches(image, mask, filter_size=13)
    print('Scratches filled in')
    image_final_inpaint = inpaint(image, [], mask)

    cv2.imwrite('corrected_inpaint.jpg', image_final_inpaint)
    cv2.imwrite('lines.png', lines)
    cv2.imwrite('corrected.jpg', image_final)
    cv2.imwrite('orginal.jpg', image)
    cv2.waitKey(0)
    sys.exit(0)


    

