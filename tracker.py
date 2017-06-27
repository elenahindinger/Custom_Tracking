from ImageProcessor import *
from configurations import *
import initializer
import os

ROIs = initializer.initialize_rois()

results_files = []
for roi in ROIs:
    current_region_file = open(os.path.join(RESULTS_FILES_DIR, roi.name) + '.dist', 'w')
    results_files.append(current_region_file)

previous_positions = [None] * len(ROIs)

previous_frame = None
previous_position = None

frame_index = frame_starting_index

while True:

    frame_name = frames_prefix + str(frame_index) + frames_extension
    frame_file = os.path.join(FRAMES_DIR, frame_name)

    if os.path.isfile(frame_file):
        img_full_frame = cv2.imread(frame_file)

        img_full_frame_gray = cv2.cvtColor(img_full_frame, cv2.COLOR_BGR2GRAY)

        if previous_frame is None:
            previous_frame = img_full_frame_gray.copy()
            continue

        img_foreground = cv2.absdiff(img_full_frame_gray, previous_frame)

        previous_frame = img_full_frame_gray.copy()

        for region_index, region in enumerate(ROIs):
            img_roi = img_full_frame_gray[region.top_left.y:region.bottom_right.y,
                      region.top_left.x:region.bottom_right.x]

            img_sharp = cv2.addWeighted(img_roi, sharp_alpha, img_roi, sharp_beta, sharp_gamma)

            img_inverted = invert(img_sharp)

            img_binary = to_binary_for_max(img_inverted)

            img_dilated = dilate(img_binary, kernel, 1)

            contour, hierarchy = cv2.findContours(img_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            distance = 0

            if contour.__len__() > 0:
                # Keep the biggest two contours (they represent the eyes)
                contour = get_biggest_n_contours(contour, 1)[0]
                # cv2.drawContours(img_roi, contour, -1, (0, 255, 0), 3)

                # Get the center point of these contours (the center point between the eyes)
                contour_center = get_contour_centroid(contour)

                if previous_positions[region_index] and contour_center:
                    distance = Vector.distance(contour_center, previous_positions[region_index])

                if not contour_center:
                    distance = -1

                previous_positions[region_index] = contour_center

            else:
                previous_positions[region_index] = None

            file_line = str(frame_index) + ' , '
            file_line += str(distance) + '\n'
            results_files[region_index].write(file_line)

        frame_index += 1
        # if frame_index == 1000:
        #     break

    else:
        break
