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
INPUT_DIR = f'{DATA_DIR}/{args.DirName}/input'
OUTPUT_DIR = f'{DATA_DIR}/{args.DirName}/output'
DEBUG_DIR = f'{DATA_DIR}/{args.DirName}/debug'
INTERFERENCE_SOURCES_DIR = f'{DATA_DIR}/{args.DirName}/interference_sources'

# Create debug directory if it doesn't exist and debug mode is enabled
if args.debug:
    os.makedirs(DEBUG_DIR, exist_ok=True)
# remove all files
for file in os.listdir(DEBUG_DIR):
    os.remove(f'{DEBUG_DIR}/{file}')

# Set the coordinate reference system (CRS) to EPSG 25833
CRS = 'epsg:25833'

# craete dirs if not exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# remove all files in output
for file in os.listdir(OUTPUT_DIR):
    os.remove(f'{OUTPUT_DIR}/{file}')


buffer_distances = [100, 625]


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


def get_input_features():
    # List to store input features
    input_features = []

    for file in glob.glob(f'{INPUT_DIR}/*.shp'):
        input_feature = gpd.read_file(file)
        input_feature = input_feature.to_crs(CRS)
        file_base_name = os.path.basename(file).split('.')[0]
        input_features.append((input_feature, file_base_name))
        area = input_feature.area.sum()
        # Print the area and name of the input feature
        print(f"Area of {file_base_name}: {area}")

    return input_features


def merged_features(input_features, buffers):

    # Merge and dissolve all input features
    merged_input_feature = pd.concat(
        [feature for feature, _ in input_features], ignore_index=True)
    if args.debug:
        merged_input_feature.to_file(f'{DEBUG_DIR}/1_merged_input_feature.shp')

    # Buffer distance
    buffer_distance = 10  # You can adjust this value as needed

    # Buffer all merged features
    buffered_features = merged_input_feature.buffer(buffer_distance)
    buffered_features = gpd.GeoDataFrame(geometry=buffered_features)
    if args.debug:
        buffered_features.to_file(f'{DEBUG_DIR}/2_buffered_features.shp')

    # Dissolve all features into a single feature
    dissolved_features = buffered_features.dissolve()
    if args.debug:
        dissolved_features.to_file(f'{DEBUG_DIR}/3_dissolved_features.shp')

    # Reduce the buffer
    reduced_buffer = dissolved_features.buffer(-buffer_distance)
    reduced_buffer = gpd.GeoDataFrame(geometry=reduced_buffer)
    if args.debug:
        reduced_buffer.to_file(f'{DEBUG_DIR}/4_reduced_buffer.shp')

    merged_input_feature = reduced_buffer

    merged_input_feature_B1_intersection = gpd.overlay(
        merged_input_feature, buffers[0], how='intersection')
    if args.debug:
        merged_input_feature_B1_intersection.to_file(
            f'{DEBUG_DIR}/5_merged_input_feature_B1_intersection.shp')

    merged_input_feature_B2_intersection = gpd.overlay(
        merged_input_feature, buffers[1], how='intersection')
    if args.debug:
        merged_input_feature_B2_intersection.to_file(
            f'{DEBUG_DIR}/6_merged_input_feature_B2_intersection.shp')

    merged_input_feature_B2_not_B1 = gpd.overlay(
        merged_input_feature_B2_intersection, merged_input_feature_B1_intersection, how='difference')
    if args.debug:
        merged_input_feature_B2_not_B1.to_file(
            f'{DEBUG_DIR}/7_merged_input_feature_B2_not_B1.shp')

    merged_input_feature_B1_area = merged_input_feature_B1_intersection.area.sum()
    merged_input_feature_B2_not_B1_area = merged_input_feature_B2_not_B1.area.sum()

    # Subtract merged_input_feature_B2_intersection from reduced_buffer
    merged_input_feature_outside_B2 = gpd.overlay(
        reduced_buffer, merged_input_feature_B2_intersection, how='difference')

    if args.debug:
        merged_input_feature_outside_B2.to_file(
            f'{DEBUG_DIR}/8_merged_input_feature_outside_B2.shp')

    merged_input_feature_outside_B2_area = merged_input_feature_outside_B2.area.sum()

    print(
        f"Area of merged input features intersecting with Buffer <{buffer_distances[0]}: {merged_input_feature_B1_area}")
    print(
        f"Area of merged input features intersecting with Buffer {buffer_distances[1]} but not {buffer_distances[0]}: {merged_input_feature_B2_not_B1_area}")
    print(
        f"Area of merged input features outside Buffer {buffer_distances[1]}: {merged_input_feature_outside_B2_area}")

    if merged_input_feature_B1_intersection.area.sum() > 0:
        merged_input_feature_B1_intersection['area'] = merged_input_feature_B1_intersection.geometry.area.round(
        ).astype(int)
        merged_input_feature_B1_intersection.to_file(
            f'{OUTPUT_DIR}/merged_input_features_buffer_{buffer_distances[0]}_intersection.shp')
    if merged_input_feature_B2_not_B1.area.sum() > 0:
        merged_input_feature_B2_not_B1['area'] = merged_input_feature_B2_not_B1.geometry.area.round(
        ).astype(int)
        merged_input_feature_B2_not_B1.to_file(
            f'{OUTPUT_DIR}/merged_input_features_intersects_buffer_{buffer_distances[1]}_not_buffer_{buffer_distances[0]}.shp')
    if merged_input_feature_outside_B2_area > 0:
        merged_input_feature_outside_B2['area'] = merged_input_feature_outside_B2.geometry.area.round(
        ).astype(int)
        merged_input_feature_outside_B2.to_file(
            f'{OUTPUT_DIR}/merged_input_features_outside_buffer_{buffer_distances[1]}.shp')


def separate_features(input_features, buffers):
    # Loop over the input features
    for input_feature, file_base_name in input_features:
        # Ensure the input feature has the same CRS as the buffers
        input_feature = input_feature.to_crs(CRS)
        # Calculate intersections
        input_feature_B1_intersection = gpd.overlay(
            input_feature, buffers[0], how='intersection')
        input_feature_B1_intersection = process_geometries(
            input_feature_B1_intersection)
        if args.debug:
            input_feature_B1_intersection.to_file(
                f'{DEBUG_DIR}/1_{file_base_name}_B1_intersection.shp')

        input_feature_B2_intersection = gpd.overlay(
            input_feature, buffers[1], how='intersection')
        input_feature_B2_intersection = process_geometries(
            input_feature_B2_intersection)
        if args.debug:
            input_feature_B2_intersection.to_file(
                f'{DEBUG_DIR}/2_{file_base_name}_B2_intersection.shp')

        # Subtract input_feature_B1_intersection from input_feature_B2_intersection
        input_feature_B2_not_B1 = gpd.overlay(
            input_feature_B2_intersection, input_feature_B1_intersection, how='difference')
        input_feature_B2_not_B1 = process_geometries(input_feature_B2_not_B1)
        if args.debug:
            input_feature_B2_not_B1.to_file(
                f'{DEBUG_DIR}/3_{file_base_name}_B2_not_B1.shp')

        # Calculate areas of intersections
        input_feature_B1_area = input_feature_B1_intersection.area.sum()
        input_feature_B2_not_B1_area = input_feature_B2_not_B1.area.sum()

        # Calculate area outside B2
        input_feature_outside_B2 = gpd.overlay(
            input_feature, buffers[1], how='difference')
        # if args.debug:
        #     input_feature_outside_B2.to_file(
        #         f'{DEBUG_DIR}/4_{file_base_name}_outside_B2.shp')

        input_feature_outside_B2 = process_geometries(input_feature_outside_B2)

        if args.debug:
            input_feature_outside_B2.to_file(
                f'{DEBUG_DIR}/5_{file_base_name}_outside_B2.shp')

        # Calculate area
        input_feature_outside_B2_area = input_feature_outside_B2.area.sum()

        # Print the results
        print(
            f"Area of input feature {file_base_name} intersecting with Buffer <{buffer_distances[0]}: {input_feature_B1_area}")
        print(
            f"Area of input feature {file_base_name} intersecting with Buffer {buffer_distances[1]} but not {buffer_distances[0]}: {input_feature_B2_not_B1_area}")
        print(
            f"Area of input feature {file_base_name} outside Buffer {buffer_distances[1]}: {input_feature_outside_B2_area}")

        # Create new shapes of all of the above
        if input_feature_B1_intersection.area.sum() > 0:
            input_feature_B1_intersection['area'] = input_feature_B1_intersection.geometry.area.round(
            ).astype(int)
            input_feature_B1_intersection.to_file(
                f'{OUTPUT_DIR}/{file_base_name}_buffer_{buffer_distances[0]}_intersection.shp')
        if input_feature_B2_not_B1.area.sum() > 0:
            input_feature_B2_not_B1['area'] = input_feature_B2_not_B1.geometry.area.round(
            ).astype(int)
            input_feature_B2_not_B1.to_file(
                f'{OUTPUT_DIR}/{file_base_name}_intersects_buffer_{buffer_distances[1]}_not_buffer_{buffer_distances[0]}.shp')
        if input_feature_outside_B2_area > 0:
            input_feature_outside_B2['area'] = input_feature_outside_B2.geometry.area.round(
            ).astype(int)
            input_feature_outside_B2.to_file(
                f'{OUTPUT_DIR}/{file_base_name}_outside_buffer_{buffer_distances[1]}.shp')


def process_geometries(input_feature_outside_B2):
    # Explode MultiPolygon geometries into individual Polygon geometries
    input_feature_outside_B2 = input_feature_outside_B2.explode(
        index_parts=False)

    # Filter to only include polygons
    input_feature_outside_B2 = input_feature_outside_B2[
        input_feature_outside_B2.geometry.type == 'Polygon']

    # Remove polygons with area of < 0.1
    input_feature_outside_B2 = input_feature_outside_B2[
        input_feature_outside_B2.geometry.area >= 0.1]

    # Merge overlapping polygons
    # Add a column with the same value for all rows
    input_feature_outside_B2["group"] = 0
    input_feature_outside_B2 = input_feature_outside_B2.dissolve(by="group")

    # Explode MultiPolygon geometries into individual Polygon geometries again
    input_feature_outside_B2 = input_feature_outside_B2.explode(
        index_parts=False)

    # Reset index
    input_feature_outside_B2 = input_feature_outside_B2.reset_index(drop=True)

    return input_feature_outside_B2


buffers = get_buffers(buffer_distances)
input_features = get_input_features()
# merged_features(input_features, buffers)
separate_features(input_features, buffers)
