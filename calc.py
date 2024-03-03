import glob
import os
import geopandas as gpd
import pandas as pd

INPUT_DIR = './input'
OUTPUT_DIR = './output'
# Directory containing interference sources
INTERFERENCE_SOURCES_DIR = './interference_sources'

# craete dirs if not exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# remove all files in output
for file in os.listdir(OUTPUT_DIR):
    os.remove(f'{OUTPUT_DIR}/{file}')


buffer_distances = [100, 625]
buffers = []

for distance in buffer_distances:
    # List to store buffers
    temp_buffers = []

    # Loop over the shapefiles in the interference sources directory
    for file in glob.glob(f'{INTERFERENCE_SOURCES_DIR}/*.shp'):
        # Load the shapefile
        interference_source = gpd.read_file(file)

        # Calculate the buffer for each feature
        # Change 1 to the desired buffer distance
        buffer = interference_source.buffer(distance)

        # Ensure the buffer has the same CRS as the interference source
        buffer.crs = interference_source.crs

        # Add the buffer to the list of buffers
        temp_buffers.append(buffer)

    # Combine the buffers into a single GeoDataFrame
    B = gpd.GeoDataFrame(
        pd.concat(temp_buffers, ignore_index=True), geometry=0)
    buffers.append(B)

# List to store input features
input_features = []


# Loop over the shapefiles in the input directory
for file in glob.glob(f'{INPUT_DIR}/*.shp'):
    # Load the shapefile
    input_feature = gpd.read_file(file)

    # Get the base name of the file
    file_base_name = os.path.splitext(os.path.basename(file))[0]

    # Add the input feature and the base name of the file to the list of input features
    input_features.append([input_feature, file_base_name])

# Loop over the input features
for input_feature, file_base_name in input_features:
    # Ensure the input feature has the same CRS as the buffers
    input_feature = input_feature.to_crs(buffers[0].crs)
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
        input_feature_B1_intersection.to_file(
            f'{OUTPUT_DIR}/{file_base_name}_buffer_{buffer_distances[0]}_intersection.shp')
    if input_feature_B2_not_B1.area.sum() > 0:
        input_feature_B2_not_B1.to_file(
            f'{OUTPUT_DIR}/{file_base_name}_intersects_buffer_{buffer_distances[1]}_not_buffer_{buffer_distances[0]}.shp')
    if input_feature_outside_B2_area > 0:
        input_feature_outside_B2.to_file(
            f'{OUTPUT_DIR}/{file_base_name}_outside_buffer_{buffer_distances[1]}.shp')
