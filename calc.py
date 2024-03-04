from shapely.geometry import Polygon
import glob
import os
import geopandas as gpd
import pandas as pd

INPUT_DIR = './input'
OUTPUT_DIR = './output'
# Directory containing interference sources
INTERFERENCE_SOURCES_DIR = './interference_sources'
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
        input_features.append(input_feature)

    return input_features


def merged_features(input_features, buffers):
    # Merge and dissolve all input features
    merged_input_feature = pd.concat(input_features, ignore_index=True)

    # Buffer distance
    buffer_distance = 10  # You can adjust this value as needed

    # Buffer all merged features
    buffered_features = merged_input_feature.buffer(buffer_distance)

    # Convert buffered_features to GeoDataFrame
    buffered_features = gpd.GeoDataFrame(geometry=buffered_features)

    # Dissolve all features into a single feature
    dissolved_features = buffered_features.dissolve()

    # Reduce the buffer
    reduced_buffer = dissolved_features.buffer(-buffer_distance)

    # Convert reduced_buffer to GeoDataFrame
    reduced_buffer = gpd.GeoDataFrame(geometry=reduced_buffer)

    merged_input_feature = reduced_buffer

    merged_input_feature_B1_intersection = gpd.overlay(
        merged_input_feature, buffers[0], how='intersection')
    merged_input_feature_B2_intersection = gpd.overlay(
        merged_input_feature, buffers[1], how='intersection')

    merged_input_feature_B2_not_B1 = gpd.overlay(
        merged_input_feature_B2_intersection, merged_input_feature_B1_intersection, how='difference')

    merged_input_feature_B1_area = merged_input_feature_B1_intersection.area.sum()
    merged_input_feature_B2_not_B1_area = merged_input_feature_B2_not_B1.area.sum()

    merged_input_feature_outside_B2 = gpd.overlay(
        merged_input_feature, buffers[1], how='difference')

    merged_input_feature_outside_B2 = merged_input_feature_outside_B2[
        merged_input_feature_outside_B2.geometry.type == 'Polygon']

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
        input_feature_B2_intersection = gpd.overlay(
            input_feature, buffers[1], how='intersection')

        # Subtract input_feature_B1_intersection from input_feature_B2_intersection
        input_feature_B2_not_B1 = gpd.overlay(
            input_feature_B2_intersection, input_feature_B1_intersection, how='difference')

        # Calculate areas of intersections
        input_feature_B1_area = input_feature_B1_intersection.area.sum()
        input_feature_B2_not_B1_area = input_feature_B2_not_B1.area.sum()

        # Calculate area outside B2
        input_feature_outside_B2 = gpd.overlay(
            input_feature, buffers[1], how='difference')

        # Filter to only include polygons
        input_feature_outside_B2 = input_feature_outside_B2[
            input_feature_outside_B2.geometry.type == 'Polygon']

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


buffers = get_buffers(buffer_distances)
input_features = get_input_features()
merged_features(input_features, buffers)
# separate_features(input_features, buffers)
