from models import *
import os

FRAMES_DIR = os.path.join('media', 'frames')
RESULTS_FILES_DIR = os.path.join('media', 'results')
frames_extension = 'tiff'

# region ROIs

region_default_width = 35
region_default_height = 35

regions_offset_from_start = Vector(0, 0, 0)
space_between_regions = Vector(15.5, 15.5, 0)
regions_start_point = Vector(30, 65, 0)

rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
columns_count = 12

# endregion

# region Image processing

sharp_alpha = 2
sharp_beta = -0.5
sharp_gamma = 0

kernel = np.ones((5, 5), np.uint8)

# endregion
