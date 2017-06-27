from configurations import *
import initializer
import cv2

ROIs = initializer.initialize_rois()
for img_name in os.listdir(FRAMES_DIR):
    if img_name.endswith(frames_extension):
        img_file = os.path.join(FRAMES_DIR, img_name)
        img = cv2.imread(img_file)

        for roi in ROIs:
            roi.draw(img, (0, 255, 0))

        cv2.imshow('ROIs check', img)
        cv2.waitKey(0)

        break
