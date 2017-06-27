from configurations import *

def initialize_rsults_folder():
    if not os.path.isdir(RESULTS_FILES_DIR):
        os.mkdir(RESULTS_FILES_DIR)

def initialize_rois():
    ROIs = []

    for column in xrange(columns_count):
        for row, row_name in enumerate(rows):
            region_start = Vector.add(regions_start_point, regions_offset_from_start)
            region_start.x += column * region_default_width + column * space_between_regions.x
            region_start.y += row * region_default_height + row * space_between_regions.y

            current_region = Region(region_default_width, region_default_height, region_start,
                                    row_name + ' - ' + str(column + 1))
            ROIs.append(current_region)

    return ROIs
