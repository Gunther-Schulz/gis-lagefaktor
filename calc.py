import re
import warnings
from shapely.geometry import Polygon
import glob
import os
import geopandas as gpd
import pandas as pd
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Process some integers.')

# Add the arguments
parser.add_argument('DirName', metavar='DirName',
                    type=str, help='the directory name')
parser.add_argument('--debug', action='store_true', help='enable debug mode')

# Parse the arguments
args = parser.parse_args()

# Now the directory name is available as args.DirName
DATA_DIR = './DATA'
CHANGING_DIR = f'{DATA_DIR}/{args.DirName}/changing'
UNCHANGING_DIR = f'{DATA_DIR}/{args.DirName}/unchanging'
COMPENSATORY_MEASURES_DIR = f'{DATA_DIR}/{args.DirName}/compensatory_measures'
PROTECTED_AREA_DIR = f'{DATA_DIR}/{args.DirName}/protected_area'
OUTPUT_DIR = f'{DATA_DIR}/{args.DirName}/output'
DEBUG_DIR = f'{DATA_DIR}/{args.DirName}/debug'
INTERFERENCE_SOURCES_DIR = f'{DATA_DIR}/{args.DirName}/interference_sources'

BUFFER_DISTANCES = (100, 625)
# BASE_VALUES = {'Acker': 1, 'Grünland': 1, 'Weg': 1, 'weg': 1}
BASE_VALUES = {'Acker': 1, 'Grünland': 1, 'Weg': 1, 'weg': 1, 'Baufeld': 1}
LAGEFAKTOR_VALUES = {'<100': 0.75, '>100<625': 1.0, '>625': 1.25}
PROTECTED_AREA_VALUES = {'high': 1.25,
                         'very_high': 1.5, 'VSG': 1.25, 'Test': 10, 'Test2': 20}
COMPENSATORY_MEASURES_AREA_VALUES = {'Grünfläche': 3}


# Create all directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHANGING_DIR, exist_ok=True)
os.makedirs(UNCHANGING_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(INTERFERENCE_SOURCES_DIR, exist_ok=True)
os.makedirs(COMPENSATORY_MEASURES_DIR, exist_ok=True)
os.makedirs(PROTECTED_AREA_DIR, exist_ok=True)

# Remove all files from output and debug dirs
for file in glob.glob(f'{OUTPUT_DIR}/*'):
    os.remove(file)
for file in glob.glob(f'{DEBUG_DIR}/*'):
    os.remove(file)


# Set the coordinate reference system (CRS) to EPSG 25833
CRS = 'epsg:25833'

# craete dirs if not exist
os.makedirs(CHANGING_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Global variable to keep track of the context
context = None


def custom_warning(message, category, filename, lineno, file=None, line=None):
    no_buffer_pattern = r"`?keep_geom_type=True`? in overlay resulted in .* dropped geometries of .* than .*\. Set `?keep_geom_type=False`? to retain all geometries"
    match = re.search(no_buffer_pattern, str(message))
    if match:
        if context == 'outside Buffer >625':
            print(
                'Custom Warning: The area of the feature is less than 0.1 and will be dropped')
        elif context == 'intersects with Buffer >100 <625 but not Buffer <100':
            print(
                'Custom Warning: The area of the feature is less than 0.1 and will be dropped')
        elif context == 'intersects with Buffer <100':
            print(
                'Custom Warning: The area of the feature is less than 0.1 and will be dropped')
        elif context == 'intersects with Buffer >100 <625':
            print(
                'Custom Warning: The area of the feature is less than 0.1 and will be dropped')
        else:
            print('Custom Warning: ' + str(message))
    else:
        print('Custom Warning: ' + str(message))


warnings.showwarning = custom_warning


def create_buffer(distance):
    # List to store buffers
    temp_buffers = []

    # Loop over the shapefiles in the interference sources directory
    for file in glob.glob(f'{INTERFERENCE_SOURCES_DIR}/*.shp'):
        # Load the shapefile
        interference_source = gpd.read_file(file)

        # Calculate the buffer for each feature
        # Change 1 to the desired buffer distance
        buffer = interference_source.buffer(distance)

        # Ensure the buffer has the same CRS as variable CRS
        buffer = buffer.to_crs(CRS)

        # Add the buffer to the list of buffers
        temp_buffers.append(buffer)

    # Combine the buffers into a single GeoDataFrame
    B = gpd.GeoDataFrame(
        pd.concat(temp_buffers, ignore_index=True), geometry=0)
    B = B.dissolve()  # TODO: should we do this?
    return B


def get_buffers(distances):
    # List to store buffers
    buffers = []

    # Loop over the buffer distances
    for distance in distances:
        # Create a buffer
        buffer = create_buffer(distance)

        # Add the buffer to the list of buffers
        buffers.append(buffer)

    return buffers


def get_features(dir):
    # List to store changing features
    features = []

    for file in glob.glob(f'{dir}/*.shp'):
        feature = gpd.read_file(file)
        feature = feature.to_crs(CRS)
        file_base_name = os.path.basename(file).split('.')[0]
        # madd column "file_name" to feature
        feature['file_name'] = file_base_name
        features.append(feature)
        area = feature.area.sum()
        # Print the area and name of the changing feature
        print(f"Area of {file_base_name}: {round(area)}")
        # print CRS
        print(f"CRS of {file_base_name}: {feature.crs}")

    return features


def cleanup_and_merge(feature, buffer_distance):
    # Save original string columns
    original_string_columns = feature.select_dtypes(include=['object'])

    # Perform operations on the geometry column
    geometry = feature.geometry.buffer(buffer_distance)
    geometry = gpd.GeoDataFrame(geometry=geometry)
    geometry = geometry.dissolve()
    geometry = geometry.buffer(-buffer_distance)
    geometry = gpd.GeoDataFrame(geometry=geometry)
    geometry = geometry.explode(index_parts=False)
    geometry = geometry[geometry.geometry.type == 'Polygon']

    # Filter original_string_columns based on the index of geometry
    original_string_columns = original_string_columns.loc[geometry.index]

    # Reset the index of the original string columns and the geometry GeoDataFrame
    original_string_columns.reset_index(drop=True, inplace=True)
    geometry.reset_index(drop=True, inplace=True)

    # Merge the new geometry with the original string columns
    return pd.concat([geometry, original_string_columns], axis=1)


def preprocess_compensatory_features(compensatory_features):
    cleaned_compensatory_features = []

    # Buffer distance
    buffer_distance = 10  # You can adjust this value as needed

    # Buffer all features in compensatory_features
    for compensatory_feature in compensatory_features:
        compensatory_feature = cleanup_and_merge(
            compensatory_feature, buffer_distance)
        file_name = compensatory_feature['file_name'][0]
        # add compensatory_measure_area_value to feature
        compensatory_feature['compensat'] = COMPENSATORY_MEASURES_AREA_VALUES[file_name]
        cleaned_compensatory_features.append(compensatory_feature)

    return cleaned_compensatory_features


def preprocess_protected_area_features(protected_area_features):
    cleaned_protected_area_features = []

    # Buffer distance
    buffer_distance = 10  # You can adjust this value as needed

    # Buffer all features in compensatory_features
    for protected_area_feature in protected_area_features:
        protected_area_feature = cleanup_and_merge(
            protected_area_feature, buffer_distance)
        file_name = protected_area_feature['file_name'][0]
        # add compensatory_measure_area_value to feature
        protected_area_feature['protected'] = PROTECTED_AREA_VALUES[file_name]
        cleaned_protected_area_features.append(protected_area_feature)
    print("cleaned_protected_area_features:", cleaned_protected_area_features)
    return cleaned_protected_area_features


def preprocess_changing_features(changing_features, unchanged_features):

    cleaned_changing_features = []

    # Buffer distance
    buffer_distance = 10  # You can adjust this value as needed

    # Buffer all features in changing_features
    for changing_feature in changing_features:
        changing_feature = cleanup_and_merge(changing_feature, buffer_distance)
        file_name = changing_feature['file_name'][0]

        # Buffer, dissolve and reduce buffer for unchanged_features
        for unchanged_feature in unchanged_features:
            unchanged_feature = cleanup_and_merge(
                unchanged_feature, buffer_distance)

            # punch holes
            changing_feature = gpd.overlay(
                changing_feature, unchanged_feature, how='difference')
            if args.debug:
                changing_feature.to_file(
                    f'{DEBUG_DIR}/2_{file_name}_punched_holes.shp')

        changing_feature['base_value'] = BASE_VALUES[file_name]
        cleaned_changing_features.append(changing_feature)

    return cleaned_changing_features


def calculate_intersection(changing_feature, buffer, context, file_base_name):
    context = context
    intersection = gpd.overlay(changing_feature, buffer, how='intersection')
    intersection = process_geometries(intersection)
    if args.debug:
        intersection.to_file(
            f'{DEBUG_DIR}/{context}_{file_base_name}_intersection.shp')
    return intersection


def calculate_difference(changing_feature1, changing_feature2, context, file_base_name):
    context = context
    difference = gpd.overlay(
        changing_feature1, changing_feature2, how='difference')
    difference = process_geometries(difference)
    if args.debug:
        difference.to_file(
            f'{DEBUG_DIR}/{context}_{file_base_name}_difference.shp')
    return difference


def calculate_area(changing_feature):
    return changing_feature.area.sum()


def print_results(file_base_name, buffer_distances, changing_feature_B1_area, changing_feature_B2_not_B1_area, changing_feature_outside_B2_area):
    print(
        f"Area of changing feature {file_base_name} intersecting with Buffer <{buffer_distances[0]}: {round(changing_feature_B1_area)}")
    print(
        f"Area of changing feature {file_base_name} intersecting with Buffer {buffer_distances[1]} but not {buffer_distances[0]}: {round(changing_feature_B2_not_B1_area)}")
    print(
        f"Area of changing feature {file_base_name} outside Buffer {buffer_distances[1]}: {round(changing_feature_outside_B2_area)}")


def create_lagefaktor_shapes(changing_feature, file_base_name, buffer_distance):
    if changing_feature.area.sum() > 0:
        changing_feature['area'] = changing_feature.geometry.area.round().astype(
            int)
        changing_feature.to_file(
            f'{OUTPUT_DIR}/{file_base_name}_buffer_{buffer_distance}_intersection.shp')


def add_protected_area_value(feature, protected_area_features):
    # Initialize 'protected' column with 1
    feature['protected'] = 1
    print(8, 2, 1)

    # Iterate over each protected_area_feature
    for protected_area_feature in protected_area_features:
        # Get the file_name from the first row of each protected_area_feature
        file_name = protected_area_feature['file_name'].iloc[0]
        print(8, 2, 2)
        # print crs of feature
        print(f"CRS of feature: {feature.crs}")
        does_intersect = False
        for _, row in feature.iterrows():
            for _, protected_row in protected_area_feature.iterrows():
                if row['geometry'].intersects(protected_row['geometry']):
                    does_intersect = True
                    break
            if does_intersect:
                break
        print(8, 2, 3)
        # If the feature intersects with the protected_area_feature
        if does_intersect:
            # Update 'protected' with the corresponding value from PROTECTED_AREA_VALUES
            feature['protected'] = PROTECTED_AREA_VALUES[file_name]
            return feature
    return feature


def add_lagefaktor_values(feature, protected_area_features, lagefaktor_value):
    # Convert feature to a GeoDataFrame if it's not already
    if not isinstance(feature, gpd.GeoDataFrame):
        feature = gpd.GeoDataFrame([feature])

    # Iterate over each row in the feature DataFrame
    for i, row in feature.iterrows():
        # Convert the row to a GeoDataFrame
        row_gdf = gpd.GeoDataFrame([row], geometry='geometry')
        # set CRS
        row_gdf.crs = CRS
        # Add protected_area_value to the row
        row = add_protected_area_value(row_gdf, protected_area_features)
        protected_area_value = row['protected'].item()

        # If protected_area_value is greater than 1 and lagefaktor_value equals '<100'
        if protected_area_value > 1 and lagefaktor_value == LAGEFAKTOR_VALUES['<100']:
            # Decrease lagefaktor_value by 0.25
            lagefaktor_value = lagefaktor_value - 0.25
        # Update 'lagefaktor' and 'protected' in the original DataFrame
        feature.loc[i, 'distance_f'] = lagefaktor_value
        feature.loc[i, 'protected'] = protected_area_value
        area = row['geometry'].area.item()
        print("area:", area)
        feature.loc[i, 'lagefaktor'] = lagefaktor_value * \
            protected_area_value * area
    return feature


def add_compensatory_measure_value(feature, compensatory_features):

    # If no intersection found or no compensatory_features
    feature['compensat'] = 1
    if compensatory_features:
        for compensatory_feature in compensatory_features:
            file_name = compensatory_feature['file_name'][0]
            if feature.intersects(compensatory_feature).any():
                feature['compensat'] = COMPENSATORY_MEASURES_AREA_VALUES[file_name]
                return feature
    return feature


def calculate_finals(feature):
    # If features is not a pandas DataFrame, make it a DataFrame
    if not isinstance(feature, gpd.GeoDataFrame):
        feature = gpd.GeoDataFrame([feature])

    # Iterate over the rows of the DataFrame
    for i, row in feature.iterrows():
        area = row['geometry'].area
        row['final'] = row['base_value'] * row['compensat'] * \
            row['lagefaktor'] * row['protected']
        row['raw_final'] = round(row['final'], 2)
        row['final'] = round(row['final'] * area, 2)
        # feature.loc[i, 'final'] = row['final']
        feature.loc[i, 'raw_final'] = row['raw_final']

    return feature


def process_geodataframes(changing_feature, protected_area_features):
    # Create separate polygons wherever changing_feature_B1_intersection overlaps with protected_area_features
    overlapping_areas = gpd.overlay(
        changing_feature, protected_area_features, how='intersection')

    # Get the parts of changing_feature_B1_intersection that don't overlap with protected_area_features
    non_overlapping_areas = gpd.overlay(
        changing_feature, protected_area_features, how='difference')

    # Combine overlapping_areas and non_overlapping_areas and keep separate polygons
    changing_feature = gpd.overlay(
        non_overlapping_areas, overlapping_areas, how='union')

    # Consolidate the 'file_name' columns
    changing_feature['file_name'] = changing_feature['file_name'].combine_first(
        changing_feature['file_name_1']).combine_first(changing_feature['file_name_2'])

    # Consolidate the 'base_value' columns
    changing_feature['base_value'] = changing_feature['base_value_1'].combine_first(
        changing_feature['base_value_2'])

    # Consolidate the 'protected' column
    changing_feature['protected'] = changing_feature['protected'].combine_first(
        changing_feature['id'])

    # Drop the redundant columns
    changing_feature = changing_feature.drop(
        columns=['file_name_1', 'file_name_2', 'base_value_1', 'base_value_2', 'id'])

    return changing_feature


def separate_features(changing_features, buffers, protected_area_features, compensatory_features):
    global context
    file_name = changing_features['file_name'][0]
    print("1 protected_area_features:", protected_area_features)
    # Convert protected_area_features to a single GeoDataFrame
    protected_area_features = pd.concat(protected_area_features)
    print("2 protected_area_features:", protected_area_features)

    # Loop over the changing features
    # Ensure the changing feature has the same CRS as the buffers
    # changing_features = changing_features.to_crs(CRS)
    # Calculate intersections
    changing_feature_B1_intersection = calculate_intersection(
        changing_features, buffers[0], 'intersects with Buffer <100', file_name)
    print(8)
    changing_feature_B1_intersection = process_geodataframes(
        changing_feature_B1_intersection, protected_area_features)

    # changing_feature_B1_intersection = add_lagefaktor_values(
    #     changing_feature_B1_intersection, protected_area_features, LAGEFAKTOR_VALUES['<100'])
    # print(9)
    # changing_feature_B1_intersection = add_compensatory_measure_value(
    #     changing_feature_B1_intersection, compensatory_features)
    # print(10)
    # changing_feature_B1_intersection = calculate_finals(
    #     changing_feature_B1_intersection)
    # print(11)

    changing_feature_B2_intersection = calculate_intersection(
        changing_features, buffers[1], 'intersects with Buffer >100 <625', file_name)
    # Subtract changing_feature_B1_intersection from changing_feature_B2_intersection
    changing_feature_B2_not_B1 = calculate_difference(
        changing_feature_B2_intersection, changing_feature_B1_intersection, 'intersects with Buffer >100 <625 but not Buffer <100', file_name)
    # changing_feature_B2_not_B1 = add_lagefaktor_values(
    #     changing_feature_B2_not_B1, protected_area_features, LAGEFAKTOR_VALUES['>100<625'])
    # changing_feature_B2_not_B1 = add_compensatory_measure_value(
    #     changing_feature_B2_not_B1, compensatory_features)
    # changing_feature_B2_not_B1 = calculate_finals(
    #     changing_feature_B2_not_B1)
    overlapping_areas = gpd.overlay(
        changing_feature_B2_not_B1, protected_area_features, how='intersection')

    non_overlapping_areas = gpd.overlay(
        changing_feature_B2_not_B1, protected_area_features, how='difference')
    changing_feature_B2_not_B1 = gpd.overlay(
        non_overlapping_areas, overlapping_areas, how='union')

    # Calculate area outside B2
    changing_feature_outside_B2 = calculate_difference(
        changing_features, buffers[1], 'outside Buffer >625', file_name)
    # changing_feature_outside_B2 = add_lagefaktor_values(
    #     changing_feature_outside_B2, protected_area_features, LAGEFAKTOR_VALUES['>625'])
    # changing_feature_outside_B2 = add_compensatory_measure_value(
    #     changing_feature_outside_B2, compensatory_features)
    # changing_feature_outside_B2 = calculate_finals(
    #     changing_feature_outside_B2)
    overlapping_areas = gpd.overlay(
        changing_feature_outside_B2, protected_area_features, how='intersection')
    non_overlapping_areas = gpd.overlay(
        changing_feature_outside_B2, protected_area_features, how='difference')
    changing_feature_outside_B2 = gpd.overlay(
        non_overlapping_areas, overlapping_areas, how='union')

    # Calculate area
    changing_feature_outside_B2_area = calculate_area(
        changing_feature_outside_B2)
    changing_feature_B2_not_B1_area = calculate_area(
        changing_feature_B2_not_B1)
    changing_feature_B1_intersection_area = calculate_area(
        changing_feature_B1_intersection)

    # Print the results
    print_results(file_name, BUFFER_DISTANCES, changing_feature_B1_intersection_area,
                  changing_feature_B2_not_B1_area, changing_feature_outside_B2_area)

    return [
        {'shape': changing_feature_B1_intersection,
            'file_base_name': file_name, 'buffer_distance': BUFFER_DISTANCES[0]},
        {'shape': changing_feature_B2_not_B1, 'file_base_name': file_name,
            'buffer_distance': BUFFER_DISTANCES[1]},
        {'shape': changing_feature_outside_B2, 'file_base_name': file_name,
            'buffer_distance': BUFFER_DISTANCES[1]}
    ]


def process_geometries(changing_feature_outside_B2):
    # Explode MultiPolygon geometries into individual Polygon geometries
    changing_feature_outside_B2 = changing_feature_outside_B2.explode(
        index_parts=False)

    # Filter to only include polygons
    changing_feature_outside_B2 = changing_feature_outside_B2[
        changing_feature_outside_B2.geometry.type == 'Polygon']

    # Remove polygons with area of < 0.1
    changing_feature_outside_B2 = changing_feature_outside_B2[
        changing_feature_outside_B2.geometry.area >= 1]

    # Merge overlapping polygons
    # Add a column with the same value for all rows
    changing_feature_outside_B2["group"] = 0
    changing_feature_outside_B2 = changing_feature_outside_B2.dissolve(
        by="group")

    # Explode MultiPolygon geometries into individual Polygon geometries again
    changing_feature_outside_B2 = changing_feature_outside_B2.explode(
        index_parts=False)

    # Reset index
    changing_feature_outside_B2 = changing_feature_outside_B2.reset_index(
        drop=True)

    return changing_feature_outside_B2


buffers = get_buffers(BUFFER_DISTANCES)
changing_features = get_features(CHANGING_DIR)
print("1 changing_features:", changing_features)
unchanging_features = get_features(UNCHANGING_DIR)
changing_features = preprocess_changing_features(
    changing_features, unchanging_features)
print("2 changing_features:", changing_features)
compensatory_features = get_features(COMPENSATORY_MEASURES_DIR)
print("3")
compensatory_features = preprocess_compensatory_features(compensatory_features)
print(4)
protected_area_features = get_features(PROTECTED_AREA_DIR)
print(5)
protected_area_features = preprocess_protected_area_features(
    protected_area_features)
print(6)
output_shapes = []
for changing_feature in changing_features:
    output_shapes.append(separate_features(
        changing_feature, buffers, protected_area_features, compensatory_features))

for shapes in output_shapes:
    for shape in shapes:
        print(shape['shape'])
# print(output_shapes[0][0]['shape'])

# create shapes
for shapes in output_shapes:
    for lagefaktor_shape in shapes:
        if lagefaktor_shape is shapes[-1]:
            lagefaktor_shape['file_base_name'] = lagefaktor_shape['file_base_name'] + \
                '_over'
        create_lagefaktor_shapes(
            lagefaktor_shape['shape'], lagefaktor_shape['file_base_name'], lagefaktor_shape['buffer_distance'])
