from geometry import *
from ImageProcessor import *
import os
import time
frame_counter = 0
search_in_detected_region = False
path = []

# region User variables

fish_regions_matrix = []
fish_data_matrix = []

default_region_width = 35
default_region_height = 35

offset = Vector(0, 0, 0)
space_between_regions = Vector(15.5, 15.5, 0)
start_point = Vector(30, 65, 0)

rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
for column in xrange(12):
    for row, row_name in enumerate(rows):
    #     if column == 6 and row_name == 'A':
        region_start = Vector.add(start_point, offset)
        region_start.x += column*default_region_width + column*space_between_regions.x
        region_start.y += row*default_region_height + row*space_between_regions.y
        fish_regions_matrix.append(Region(default_region_width, default_region_height, region_start, row_name+' - '+str(column+1)))


sharp_alpha = 2
sharp_beta = -0.5
sharp_gamma = 0

default_threshold = 50

min_contour_area = 10
kernel = np.ones((5, 5), np.uint8)

previous_frame = None

for img_name in os.listdir('media'):
    frame = os.path.join('media', img_name)
    img_full_frame = cv2.imread(frame)

    # img_full_frame = cv2.cvtColor(img_full_frame, cv2.COLOR_BGR2GRAY)

    # if previous_frame is None:
    #     previous_frame = img_full_frame.copy()
    #     continue
    #
    # fgmask = cv2.absdiff(img_full_frame,previous_frame)
    #
    # previous_frame = img_full_frame.copy()

    for region in fish_regions_matrix:
        img = img_full_frame[region.top_left.y:region.bottom_right.y, region.top_left.x:region.bottom_right.x]
        # img_contour = img_full_frame[region.top_left.y:region.bottom_right.y, region.top_left.x:region.bottom_right.x]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # img_blur = cv2.GaussianBlur(img_gray, (25, 25), 0)

        img_sharp = cv2.addWeighted(img_gray, sharp_alpha, img_gray, sharp_beta,sharp_gamma)

        img_inverted = invert(img_sharp)

        img_binary = to_binary_for_max(img_inverted)

        # img_eroded = erode(img_binary, kernel, 1)

        # img_dilated = dilate(img_binary, kernel, 1)
        # cv2.imshow(img_name, img_binary)
        # cv2.waitKey(10000)

        contour, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contour, -1, (0, 255, 0), 3)

        # if there is a detected fish (we know that if we have eyes),
        if contour.__len__() > 0:
            # Keep the biggest two contours (they represent the eyes)
            contour = get_biggest_n_contours(contour, 1)[0]

            # cv2.imshow(img_name+' '+region.name, img)
            cv2.imshow(img_name, img)
            cv2.waitKey(10000)
            # Get the center point of these contours (the center point between the eyes)
            contour_center = get_contour_centroid(contour)


