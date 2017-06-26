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
files = []
previous_positions = [None]*96

for column in xrange(12):
    for row, row_name in enumerate(rows):
        region_start = Vector.add(start_point, offset)
        region_start.x += column*default_region_width + column*space_between_regions.x
        region_start.y += row*default_region_height + row*space_between_regions.y

        current_region = Region(default_region_width, default_region_height, region_start, row_name+' - '+str(column+1))
        fish_regions_matrix.append(current_region)

        current_region_file = open(current_region.name+'.dist', 'w')
        files.append(current_region_file)


sharp_alpha = 2
sharp_beta = -0.5
sharp_gamma = 0

default_threshold = 50

min_contour_area = 10
kernel = np.ones((5, 5), np.uint8)

previous_frame = None
previous_position = None
for frame_index, img_name in enumerate(os.listdir('media')):

    frame = os.path.join('media', img_name)
    img_full_frame = cv2.imread(frame)

    img_full_frame = cv2.cvtColor(img_full_frame, cv2.COLOR_BGR2GRAY)

    if previous_frame is None:
        previous_frame = img_full_frame.copy()
        continue

    print img_name
    fgmask = cv2.absdiff(img_full_frame,previous_frame)

    previous_frame = img_full_frame.copy()

    for region_index, region in enumerate(fish_regions_matrix):
        img = img_full_frame[region.top_left.y:region.bottom_right.y, region.top_left.x:region.bottom_right.x]
        # img_contour = img_full_frame[region.top_left.y:region.bottom_right.y, region.top_left.x:region.bottom_right.x]
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # img_blur = cv2.GaussianBlur(img_gray, (25, 25), 0)

        img_sharp = cv2.addWeighted(img, sharp_alpha, img, sharp_beta,sharp_gamma)

        img_inverted = invert(img_sharp)

        img_binary = to_binary_for_max(img_inverted)

        # img_eroded = erode(img_binary, kernel, 1)

        img_dilated = dilate(img_binary, kernel, 1)
        # cv2.imshow(img_name, img_binary)
        # cv2.waitKey(10000)

        contour, hierarchy = cv2.findContours(img_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        distance = 0
        # if there is a detected fish (we know that if we have eyes),
        if contour.__len__() > 0:
            # Keep the biggest two contours (they represent the eyes)
            contour = get_biggest_n_contours(contour, 1)[0]
            cv2.drawContours(img, contour, -1, (0, 255, 0), 3)

            # print img_name, region.name
            # cv2.imshow(img_name, img)
            # cv2.waitKey(100000)
            # Get the center point of these contours (the center point between the eyes)
            contour_center = get_contour_centroid(contour)
            # print contour_center
            if previous_positions[region_index] and contour_center:
                distance = Vector.distance(contour_center, previous_positions[region_index])

            if not contour_center:
                distance = -1

            previous_positions[region_index] = contour_center

        else:
            previous_positions[region_index] = None

        file_line = str(frame_index) + ' , '
        file_line += str(distance) + '\n'
        files[region_index].write(file_line)


