# -*- coding: utf-8 -*-
import shutil
import os
import argparse
import geopandas as gpd
import pandas as pd
import glob
import re
import warnings


# Constants
CRS = 'epsg:25833'
GRZ = '0.5'
DATA_DIR = './DATA'
GRZ_FACTORS = {
    '0.5': [0.5, 0.2, 0.6],
    '0.75': [0.75, 0.5, 0.8]
}

# Create the parser
parser = argparse.ArgumentParser(description='Process some integers.')

# Add the arguments
parser.add_argument('DirName', metavar='DirName',
                    type=str, help='the directory name')
parser.add_argument('--debug', action='store_true', help='enable debug mode')

# Parse the arguments
args = parser.parse_args()

# Define directories
dir_path = os.path.join(DATA_DIR, args.DirName)
SCOPE_DIR = os.path.join(dir_path, 'scope')
CHANGING_DIR = os.path.join(dir_path, 'changing')
CONSTRUCTION_DIR = os.path.join(dir_path, 'construction')
UNCHANGING_DIR = os.path.join(dir_path, 'unchanging')
COMPENSATORY_DIR = os.path.join(dir_path, 'compensatory')
PROTECTED_DIR = os.path.join(dir_path, 'protected')
OUTPUT_DIR = os.path.join(dir_path, 'output')
DEBUG_DIR = os.path.join(dir_path, 'debug')
INTERFERENCE_DIR = os.path.join(dir_path, 'interference')

BUFFER_DISTANCES = (100, 625)
CHANGING_CONSTRUCTION_BASE_VALUES = {
    'Acker': 1, 'Grünland': 1, 'Weg': 1, 'weg': 1}

CONSTRUCTION_LAGEFAKTOR_VALUES = {'<100': 0.75, '>100<625': 1, '>625': 1.25}
CONSTRUCTION_PROTECTED_VALUES = {
    'NSG': 1.5, 'VSG': 1.25, 'Test': 10, 'Test2': 20}

CHANGING_COMPENSATORY_BASE_VALUES = {'Acker': 0}
COMPENSATORY_MEASURE_VALUES = {
    'Grünfläche': 3, "comp_test1": 10, "comp_test2": 20}
COMPENSATORY_PROTECTED_VALUES = {
    'NSG': 1.1, 'VSG': 1.15, 'FFH-LTR': 1.25, 'Test': 2, 'Test2': 4}


# List of directories to create
dirs = [DATA_DIR, CONSTRUCTION_DIR, UNCHANGING_DIR, OUTPUT_DIR,
        DEBUG_DIR, INTERFERENCE_DIR, COMPENSATORY_DIR, PROTECTED_DIR]

# Create all directories
for dir in dirs:
    os.makedirs(dir, exist_ok=True)

# List of directories to clean
dirs = [OUTPUT_DIR, DEBUG_DIR]


# Remove all files and subdirectories in each directory
for dir in dirs:
    shutil.rmtree(dir, ignore_errors=True)
    os.makedirs(dir, exist_ok=True)


# Global variable to keep track of the context
context = None


def custom_warning(message, category, filename, lineno, file=None, line=None):
    no_buffer_pattern = r"`?keep_geom_type=True`? in overlay resulted in .* dropped geometries of .* than .*\. Set `?keep_geom_type=False`? to retain all geometries"
    keepdims_pattern = r"<class 'geopandas.array.GeometryArray'>._reduce will require a `keepdims` parameter in the future"
    match_no_buffer = re.search(no_buffer_pattern, str(message))
    match_keepdims = re.search(keepdims_pattern, str(message))

    if match_no_buffer:
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
            print('Custom Warning: ' + "During overlay operations, geometries such as lines or points that don't match the geometry type of the first DataFrame can be dropped.")
    elif not match_keepdims:
        # Only print the warning if it's not the 'keepdims' warning
        print('Custom Warning: ' + str(message))


warnings.showwarning = custom_warning


def create_buffer(distance):
    # List to store buffers
    temp_buffers = []

    # Loop over the shapefiles in the interference sources directory
    for file in glob.glob(f'{INTERFERENCE_DIR}/*.shp'):
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
    B.crs = CRS
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
        feature['file'] = file_base_name
        features.append(feature)
        print(f"CRS of {file_base_name}: {feature.crs}")

    # Check if the list is empty
    if not features:
        print(f"No shapefiles found in directory {dir}")
        # Return an empty GeoDataFrame with the same columns
        return gpd.GeoDataFrame(columns=['geometry', 'file'], crs=CRS)

    # Concatenate all GeoDataFrames in the list into a single GeoDataFrame
    features_gdf = pd.concat(features, ignore_index=True)

    # Ensure the resulting GeoDataFrame has the correct CRS
    features_gdf.set_crs(CRS, inplace=True)

    return features_gdf


def cleanup_and_merge(feature, buffer_distance):
    # Save original string columns
    original_string_columns = feature.select_dtypes(include=['object'])

    # Perform operations on the geometry column
    geometry = feature.geometry.buffer(buffer_distance)
    geometry = gpd.GeoDataFrame(geometry=geometry)

    # Dissolve based on 'file'
    geometry['file'] = original_string_columns['file']
    geometry = geometry.dissolve(by='file')

    # Reset the index of geometry DataFrame
    geometry.reset_index(inplace=True)

    geometry = geometry.buffer(-buffer_distance)
    geometry = gpd.GeoDataFrame(geometry=geometry)
    geometry = geometry.explode(index_parts=False)
    geometry = geometry[geometry.geometry.type == 'Polygon']

    # Filter original_string_columns based on the index of geometry
    original_string_columns = original_string_columns.loc[geometry.index]

    # Reset the index of the original string columns and the geometry GeoDataFrame
    original_string_columns.reset_index(drop=True, inplace=True)
    geometry.reset_index(drop=True, inplace=True)
    geometry.crs = CRS

    # Merge the new geometry with the original string columns
    merged = pd.concat([geometry, original_string_columns], axis=1)

    return merged


def preprocess_compensatory_features(compensatory_features):
    cleaned_compensatory_features = cleanup_and_merge(
        compensatory_features, buffer_distance=10)

    # Use map to assign 'compensat' for each item based on its 'file'
    cleaned_compensatory_features['compensat'] = cleaned_compensatory_features['file'].map(
        COMPENSATORY_MEASURE_VALUES)

    return cleaned_compensatory_features


def preprocess_protected_area_features(protected_area_features):
    # Iterate over each row in the GeoDataFrame
    for index, row in protected_area_features.iterrows():
        # Get the 'file' for the current row
        file_name = row['file']
        # Set the 'protected' value for the current row
        protected_area_features.at[index,
                                   'protected'] = CONSTRUCTION_PROTECTED_VALUES[file_name]
    return protected_area_features


def preprocess_base_features(base_features, changing_features, unchanged_features, values):
    # Buffer, dissolve and reduce buffer for unchanged_features
    unchanged_features = cleanup_and_merge(
        unchanged_features, buffer_distance=10)

    changing_features = cleanup_and_merge(
        changing_features, buffer_distance=10)

    base_features = cleanup_and_merge(
        base_features, buffer_distance=10)

    # Rename 'file' column in changing_features
    changing_features = changing_features.rename(
        columns={'file': 'changing_f'})

    # punch holes
    changing_features = gpd.overlay(
        changing_features, unchanged_features, how='difference')

    # Overlay construction_features with changing_features
    intersected_features = gpd.overlay(
        base_features, changing_features, how='intersection')

    # Select only the columns from construction_features
    intersected_features = intersected_features[base_features.columns]
    # Also add the 'changing_file_name' column
    intersected_features['changing_f'] = changing_features['changing_f']

    # Flatten the result into a single geometry and keep the first unique value for each group of data
    base_features = intersected_features.dissolve(
        by='file', aggfunc='first')

    # explode
    base_features = base_features.explode(index_parts=False)

    # Reset the index of base_features and changing_features
    base_features.reset_index(drop=False, inplace=True)
    changing_features.reset_index(drop=False, inplace=True)

    # Merge the base_features with the changing_features
    base_features['base_value'] = base_features['changing_f'].map(
        values)

    return base_features


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


def filter_features(scope, features):
    if not scope.empty:
        print('Filtering features')
        # Check if the geometry of each row in changing_feature is within or overlaps with the scope
        features = features[features.geometry.within(
            scope.geometry.unary_union) | features.geometry.overlaps(scope.geometry.unary_union)]
    return features


def create_lagefaktor_shapes(changing_features, file_base_name, buffer_distance):
    if changing_features.area.sum() > 0:
        # Set the file name
        file_name = f'Construction_{file_base_name}_buffer_{buffer_distance}_intersection'

        # Set the directory name to be the same as the file name
        new_dir = f'{OUTPUT_DIR}/{file_name}'
        os.makedirs(new_dir, exist_ok=True)

        # Save the shape in the new directory
        changing_features.to_file(f'{new_dir}/{file_name}.shp')


def resolve_overlaps(feature):
    resolved_geometries = gpd.GeoDataFrame(columns=feature.columns)
    for index, row in feature.iterrows():
        current_geom = row.geometry
        # Create a temporary GeoDataFrame for the current row to facilitate the use of concat
        temp_gdf = gpd.GeoDataFrame([row], columns=feature.columns)

        # Check for intersections with already resolved geometries (those with higher 'lagefaktor')
        for _, r_geom in resolved_geometries.iterrows():
            if current_geom.intersects(r_geom.geometry):
                # If there's an intersection, difference the current geometry with the higher 'lagefaktor' geometry
                current_geom = current_geom.difference(r_geom.geometry)

        # If there's any geometry left after resolving intersections, add it to the resolved geometries
        if not current_geom.is_empty:
            # Update the geometry in the temporary GeoDataFrame
            temp_gdf.geometry = [current_geom]

            # Drop columns that are entirely NA or empty in both DataFrames before concatenating
            resolved_geometries = resolved_geometries.dropna(how='all', axis=1)
            temp_gdf = temp_gdf.dropna(how='all', axis=1)

            resolved_geometries = pd.concat(
                [resolved_geometries, temp_gdf], ignore_index=True)

    # Explode any MultiPolygon geometries into individual Polygon geometries
    resolved_geometries = resolved_geometries.explode(index_parts=True)
    resolved_geometries.crs = CRS

    return resolved_geometries


def remove_slivers(gdf, buffer_distance=0.0001):
    """
    Apply a small buffer to fill in slivers and then reverse the buffer.
    Note: The buffer distance should be chosen based on the coordinate system of your GeoDataFrame.
    A very small value is usually sufficient and should be adjusted according to your specific needs.
    """
    # Apply a small buffer to fill in slivers
    buffered_gdf = gdf.buffer(buffer_distance)

    # Reverse the buffer to return to original size
    unbuffered_gdf = buffered_gdf.buffer(-buffer_distance)

    # Update the original GeoDataFrame geometries
    gdf.geometry = unbuffered_gdf
    gdf.crs = CRS

    return gdf


def add_lagefaktor_values(feature, lagefaktor_value):
    # save feature to debug file
    feature.to_file(f'{DEBUG_DIR}/feature{lagefaktor_value}.shp')
    # Add a new column 'lagefaktor'
    feature['lagefaktor'] = feature.apply(lambda row: row['protected'] if pd.notnull(
        row['protected']) else lagefaktor_value, axis=1)

    # If lagefaktor_value is 0.75, reduce the final 'lagefaktor' value by 0.25
    if lagefaktor_value == CONSTRUCTION_LAGEFAKTOR_VALUES['<100']:
        feature['lagefaktor'] = feature['lagefaktor'] - 0.25

    # Dissolve polygons based on 'lagefaktor', taking the first (or max, depending on your requirement) value for each group
    # Note: Adjust the aggregation function as needed. Here, we're using 'first' for simplicity.
    # We can use 'first' here , beacause there is only one value for each 'lagefaktor' group
    flattened_feature = feature.dissolve(by='lagefaktor', aggfunc='first')
    flattened_feature.reset_index(inplace=True)

    # Sort the GeoDataFrame by 'lagefaktor' in descending order, if needed
    flattened_feature.sort_values(
        by='lagefaktor', ascending=False, inplace=True)

    # Assuming 'gdf' is your GeoDataFrame with a 'lagefaktor' column and a 'geometry' column
    resolved_gdf = resolve_overlaps(flattened_feature)

    # Remove slivers
    resolved_gdf = remove_slivers(resolved_gdf, 0.001)

    # save resolved_gdf to debug file
    resolved_gdf.to_file(f'{DEBUG_DIR}/resolved_gdf{lagefaktor_value}.shp')

    return resolved_gdf


def add_compensatory_value(compensatory_features, protected_area_features):

    protected_area_features = protected_area_features.sort_values(
        by='protected', ascending=False)
    protected_area_features = resolve_overlaps(protected_area_features)

    # Add a new column 'compensat'
    compensatory_features['compensat'] = compensatory_features.apply(
        lambda row: COMPENSATORY_MEASURE_VALUES[row['file']], axis=1)

    # Use process_geodataframes function
    flattened_features = process_geodataframes(
        compensatory_features, protected_area_features, 'protected')

    # write debug file
    flattened_features.to_file(
        f'{DEBUG_DIR}/compensatory_features.shp')

    return flattened_features


def process_geodataframes(base_feature, cover_features, sort_by):
    cover_features = cover_features.sort_values(
        by=sort_by, ascending=False)
    # rolve_overlaps for cover_features
    cover_features = resolve_overlaps(cover_features)

    # Create separate polygons wherever changing_feature overlaps with protected_area_features
    overlapping_areas = gpd.overlay(
        base_feature, cover_features, how='intersection')

    overlapping_areas.crs = CRS

    # Get the parts of changing_feature that don't overlap with protected_area_features
    non_overlapping_areas = gpd.overlay(
        base_feature, cover_features, how='difference')

    non_overlapping_areas.crs = CRS

    # Combine overlapping_areas and non_overlapping_areas and keep separate polygons
    if not non_overlapping_areas.empty and not overlapping_areas.empty:
        base_feature = gpd.overlay(
            non_overlapping_areas, overlapping_areas, how='union')
    else:
        base_feature = overlapping_areas
    # Get the non-geometry columns
    non_geometry_columns = base_feature.columns.difference(['geometry'])

    # Iterate over the non-geometry columns
    for column in non_geometry_columns:
        # Get the columns that start with the same initial substring
        matching_columns = [
            col for col in base_feature.columns if col.split('_')[0] == column.split('_')[0]]

        # Skip if there are no matching columns
        if len(matching_columns) < 2:
            continue

        # Sort the matching columns by length
        matching_columns.sort(key=len)

        # Consolidate the matching columns
        for matching_column in matching_columns[1:]:
            base_feature[matching_columns[0]] = base_feature[matching_columns[0]].combine_first(
                base_feature[matching_column])

        # Drop the redundant columns
        base_feature = base_feature.drop(columns=matching_columns[1:])

        # Rename the consolidated column to the base name
        base_name = matching_columns[0].rsplit('_', 1)[0]
        base_feature = base_feature.rename(
            columns={matching_columns[0]: base_name})

    # remove slivers
    base_feature = remove_slivers(base_feature, 0.001)

    return base_feature


def separate_features(scope, construction_feature, buffers, protected_area_features):
    global context
    # TODO: we can hondly handle a singl construction feature at this time (ususally Baufled)
    file_name = construction_feature['file'].iloc[0]

    # Calculate intersections
    changing_feature_B1_intersection = calculate_intersection(
        construction_feature, buffers[0], 'intersects with Buffer <100', file_name)
    changing_feature_B1_intersection = process_geodataframes(
        changing_feature_B1_intersection, protected_area_features, 'protected')
    changing_feature_B1_intersection = add_lagefaktor_values(
        changing_feature_B1_intersection, CONSTRUCTION_LAGEFAKTOR_VALUES['<100'])
    changing_feature_B1_intersection = filter_features(scope,
                                                       changing_feature_B1_intersection)

    changing_feature_B2_intersection = calculate_intersection(
        construction_feature, buffers[1], 'intersects with Buffer >100 <625', file_name)
    # Subtract changing_feature_B1_intersection from changing_feature_B2_intersection
    changing_feature_B2_not_B1 = calculate_difference(
        changing_feature_B2_intersection, changing_feature_B1_intersection,
        'intersects with Buffer >100 <625 but not Buffer <100', file_name)
    changing_feature_B2_not_B1 = process_geodataframes(
        changing_feature_B2_not_B1, protected_area_features, 'protected')
    changing_feature_B2_not_B1 = add_lagefaktor_values(
        changing_feature_B2_not_B1, CONSTRUCTION_LAGEFAKTOR_VALUES['>100<625'])
    changing_feature_B2_not_B1 = filter_features(
        scope, changing_feature_B2_not_B1)

    # Calculate area outside B2
    changing_feature_outside_B2 = calculate_difference(
        construction_feature, buffers[1], 'outside Buffer >625', file_name)
    changing_feature_outside_B2 = process_geodataframes(
        changing_feature_outside_B2, protected_area_features, 'protected')
    changing_feature_outside_B2 = add_lagefaktor_values(
        changing_feature_outside_B2, CONSTRUCTION_LAGEFAKTOR_VALUES['>625'])
    changing_feature_outside_B2 = filter_features(
        scope, changing_feature_outside_B2)

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


def calculate_total_final_value(dicts_list, grz):
    # Calculate the final value for each feature and summarize them
    grz_f = GRZ_FACTORS[grz]
    total_final_value = 0
    for feature_dict in dicts_list:
        feature = feature_dict['shape']
        feature['final_val'] = feature['base_value'] * \
            feature['lagefaktor'] * feature.geometry.area
        total_final_value += feature['final_val'].sum()

    total_final_value = (
        (total_final_value * grz_f[0]) * grz_f[1]) + ((total_final_value * grz_f[0]) * grz_f[2])
    total_final_value = round(total_final_value, 2)

    return total_final_value


def process_geometries(changing_feature_outside_B2):
    # Explode MultiPolygon geometries into individual Polygon geometries
    changing_feature_outside_B2 = changing_feature_outside_B2.explode(
        index_parts=False)

    # Filter to only include polygons
    changing_feature_outside_B2 = changing_feature_outside_B2[
        changing_feature_outside_B2.geometry.type == 'Polygon']

    changing_feature_outside_B2 = remove_slivers(
        changing_feature_outside_B2, 0.001)

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

    changing_feature_outside_B2.crs = CRS

    return changing_feature_outside_B2


buffers = get_buffers(BUFFER_DISTANCES)
scope = get_features(SCOPE_DIR)
construction_features = get_features(CONSTRUCTION_DIR)
changing_features = get_features(CHANGING_DIR)
unchanging_features = get_features(UNCHANGING_DIR)
protected_area_features = get_features(PROTECTED_DIR)
protected_area_features = preprocess_protected_area_features(
    protected_area_features)
construction_features = preprocess_base_features(
    construction_features, changing_features, unchanging_features, CHANGING_CONSTRUCTION_BASE_VALUES)
compensatory_features = get_features(COMPENSATORY_DIR)

compensatory_features = preprocess_compensatory_features(compensatory_features)

compensatory_features = preprocess_base_features(
    compensatory_features, changing_features, unchanging_features, CHANGING_COMPENSATORY_BASE_VALUES)

compensatory_features = add_compensatory_value(
    compensatory_features, protected_area_features)

output_shapes = []
output_shapes = separate_features(
    scope, construction_features, buffers, protected_area_features)

# for shapes in output_shapes:
#     print(shape['shape'])
# print(output_shapes[0][0]['shape'])


total_final_value = calculate_total_final_value(output_shapes, GRZ)
print(f"Total final value: {total_final_value}")

# create shapes
for lagefaktor_shape in output_shapes:
    # Check if any column name is longer than 10 characters
    for column in lagefaktor_shape['shape'].columns:
        if len(column) > 10:
            print(
                f"Warning: Column name '{column}' is longer than 10 characters.")

    if lagefaktor_shape is output_shapes[-1]:
        lagefaktor_shape['file_base_name'] = lagefaktor_shape['file_base_name'] + '_over'

    create_lagefaktor_shapes(
        lagefaktor_shape['shape'], lagefaktor_shape['file_base_name'], lagefaktor_shape['buffer_distance'])

# create shapes for compensatory features, one shape file for each from in attribute 'file'
for file in compensatory_features['file'].unique():
    # Get the features for the current file
    current_features = compensatory_features[compensatory_features['file'] == file]
    current_features = filter_features(scope, current_features)

    # Calculate

    # Filter out the rows with area over 0
    current_features = current_features[current_features.geometry.area > 0]

    # Save the GeoDataFrame to a file in the output directory
    current_features.to_file(os.path.join(
        OUTPUT_DIR, 'Compensatory_' + file), driver='ESRI Shapefile')
