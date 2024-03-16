# -*- coding: utf-8 -*-
import sys
import shutil
import os
import argparse
import geopandas as gpd
import pandas as pd
import re
import warnings
import glob


# Constants
CRS = 'epsg:25833'
GRZ = '0.5'
DATA_DIR = './DATA'
GRZ_FACTORS = {
    '0.5': [0.5, 0.2, 0.6],
    '0.75': [0.75, 0.5, 0.8]
}


# Create the parser
parser = argparse.ArgumentParser(
    description='Calculate the final value of construction and compensatory features and create shapefiles for each feature.')

# Add the arguments
parser.add_argument('project', metavar='project', type=str, nargs='?', default=None,
                    help='the project name')
parser.add_argument('-n', '--new', metavar='NewProjectName',
                    type=str, help='the new project name')
parser.add_argument('-d', '--debug', action='store_true',
                    help='enable debug mode')

# Parse the arguments
args = parser.parse_args()

# Define directories
dir_path = os.path.join(DATA_DIR, args.new) if args.new else os.path.join(
    DATA_DIR, args.project)
SCOPE_DIR = os.path.join(dir_path, 'scope')
CHANGING_DIR = os.path.join(dir_path, 'changing')
CONSTRUCTION_DIR = os.path.join(dir_path, 'construction')
UNCHANGING_DIR = os.path.join(dir_path, 'unchanging')
COMPENSATORY_DIR = os.path.join(dir_path, 'compensatory')
PROTECTED_DIR = os.path.join(dir_path, 'protected')
OUTPUT_DIR = os.path.join(dir_path, 'output')
DEBUG_DIR = os.path.join(dir_path, 'debug')
INTERFERENCE_DIR = os.path.join(dir_path, 'interference')

# List of directories to create
dirs = [dir_path, SCOPE_DIR, CHANGING_DIR, CONSTRUCTION_DIR, UNCHANGING_DIR,
        COMPENSATORY_DIR, PROTECTED_DIR, OUTPUT_DIR, DEBUG_DIR, INTERFERENCE_DIR]

# If the --new argument is provided, create the project directory and all subdirectories
if args.new:
    # Create all directories
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

    print(
        f"New project '{args.new}' has been created with all necessary directories.")
    sys.exit()

# Check if the project directory exists and is empty
if os.path.exists(dir_path) and not os.listdir(dir_path):
    print(f"Project directory {dir_path} is empty.")
    sys.exit()
elif not os.path.exists(dir_path):
    print(f"Project directory {dir_path} does not exist.")
    sys.exit()

# Create all directories
for dir in dirs:
    os.makedirs(dir, exist_ok=True)

# List of directories to clean
dirs = [OUTPUT_DIR, DEBUG_DIR]

# Remove all files and subdirectories in each directory
for dir in dirs:
    shutil.rmtree(dir, ignore_errors=True)
    os.makedirs(dir, exist_ok=True)

BUFFER_GEN_DISTANCES = (100, 625)
BUFFER_DISTANCES = {
    '<100': '<100',
    '>100<625': '>100<625',
    '>625': '>625'
}
CHANGING_CONSTRUCTION_BASE_VALUES = {
    'Acker': 1, 'Grünland': 1, 'Weg': 1, 'weg': 1}

CONSTRUCTION_LAGEFAKTOR_VALUES = {'<100': 0.75, '>100<625': 1, '>625': 1.25}
CONSTRUCTION_PROTECTED_VALUES = {
    'NSG': 1.5, 'VSG': 1.25, 'GGB': 1.5, 'Test': 10, 'Test2': 20}

# TODO:
CHANGING_COMPENSATORY_BASE_VALUES = {'Acker': 0, 'Grünland': 0, 'weg': 0}
COMPENSATORY_MEASURE_VALUES = {
    'Grünfläche': 3, "comp_test": 10}
COMPENSATORY_PROTECTED_VALUES = {
    'NSG': 1.1, 'VSG': 1.15, 'GGB': 1.25, 'Test': 2, 'Test2': 4}

output_data = {
    'construction': [

    ],
    'compensatory': [

    ]
}

# Global variable to keep track of the context
context = None

# ----> Utility Functions <----


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


def get_value_with_warning(values, key):
    if key not in values:
        print(f"Warning: Value for {key} does not exist.")
        return None  # or return a default value
    return values[key]

# ----> Feature Retrieval and Initialization <----


def read_shapefile(file_path):
    print(f"Reading shapefile {os.path.basename(file_path)}")
    feature = gpd.read_file(file_path)
    feature = feature.to_crs(CRS)
    feature = feature[['geometry']]
    feature['s_name'] = os.path.basename(os.path.dirname(file_path))
    return feature


def get_features(dir):
    print(f"Reading shapefiles from directory {dir}")
    shapefiles = glob.glob(f"{dir}/*/*.shp")

    if not shapefiles:
        print(f"No shapefiles found in directory {dir}")
        return gpd.GeoDataFrame(columns=['geometry', 's_name'], crs=CRS)

    features = [read_shapefile(shapefile) for shapefile in shapefiles]
    features_gdf = pd.concat(features, ignore_index=True)
    features_gdf.set_crs(CRS, inplace=True)

    return features_gdf

# ----> Buffer Operations <----


def create_buffer(linestrings, distance):
    # Create a buffer around each linestring and dissolve all geometries into a single one
    buffers = linestrings.buffer(distance).to_frame().rename(
        columns={0: 'geometry'}).set_geometry('geometry').dissolve()
    return buffers


def get_buffers(features, distances):
    # Create a buffer for each distance and return the list of buffers
    return [create_buffer(features, distance) for distance in distances]


# ----> Feature Cleanup and Geometry Manipulation <----

def cleanup_and_merge_features(feature, buffer_distance):
    """
    Cleans up and merges features based on their geometry and 's_name'.

    Parameters:
    - feature: GeoDataFrame to be processed.
    - buffer_distance: Distance for buffering operations.

    Returns:
    - GeoDataFrame with merged features.
    """
    # Preserve original string columns
    original_strings = feature.select_dtypes(include=['object'])

    # Geometry manipulation
    feature = (feature
               .assign(geometry=lambda x: x.geometry.buffer(buffer_distance))
               .explode(index_parts=False)
               .assign(geometry=lambda x: x.geometry.buffer(-buffer_distance))
               .loc[lambda x: x.geometry.geom_type == 'Polygon']
               .dissolve(by='s_name')
               .reset_index())

    # Set the CRS
    feature.crs = CRS

    # Merge with original string columns
    original_strings = original_strings.loc[feature.index].reset_index(
        drop=True)
    original_strings.drop(columns='s_name', inplace=True)
    merged_features = pd.concat([feature, original_strings], axis=1)

    return merged_features


def resolve_overlaps(feature):
    """
    Resolves overlaps in geometries based on 'lagefaktor'.

    Parameters:
    - feature: GeoDataFrame with potential overlaps.

    Returns:
    - GeoDataFrame with resolved geometries.
    """
    resolved = gpd.GeoDataFrame(columns=feature.columns)

    for _, row in feature.iterrows():
        current_geom = row.geometry
        temp_gdf = gpd.GeoDataFrame([row], columns=feature.columns)

        for _, r_row in resolved.iterrows():
            if current_geom.intersects(r_row.geometry):
                current_geom = current_geom.difference(r_row.geometry)

        if not current_geom.is_empty:
            temp_gdf.geometry = [current_geom]
            # Exclude empty or all-NA columns before concatenation
            temp_gdf.dropna(how='all', axis=1, inplace=True)
            resolved.dropna(how='all', axis=1, inplace=True)
            resolved = pd.concat([resolved, temp_gdf], ignore_index=True)

    resolved = resolved.explode(index_parts=True)
    resolved.crs = CRS

    return resolved


def remove_slivers(gdf, buffer_distance=0.0001):
    """
    Removes slivers from geometries by applying a small buffer.

    Parameters:
    - gdf: GeoDataFrame to be processed.
    - buffer_distance: Distance for buffering operations.

    Returns:
    - GeoDataFrame with slivers removed.
    """
    gdf.geometry = gdf.geometry.buffer(
        buffer_distance).buffer(-buffer_distance)
    gdf.crs = CRS
    return gdf

# ----> Feature Processing and Transformation <----


def preprocess_features(features, feature_type, buffer_distance=10):
    """
    Generalized function to preprocess different types of features.

    Parameters:
    - features: GeoDataFrame of features to be processed.
    - feature_type: Type of features being processed ('compensatory', 'protected_area', or 'base').
    - buffer_distance: Buffer distance for cleanup and merge operation, default is 10.

    Returns:
    - Processed GeoDataFrame.
    """
    # Cleanup and merge features
    processed_features = cleanup_and_merge_features(
        features, buffer_distance=buffer_distance)

    if feature_type == 'compensatory':
        # Assign 'compensat' based on 's_name'
        processed_features['compensat'] = processed_features['s_name'].map(
            lambda x: get_value_with_warning(COMPENSATORY_MEASURE_VALUES, x))
    elif feature_type == 'protected_area':
        # Set 'protected' value based on 's_name'
        processed_features['protected'] = processed_features['s_name'].apply(
            lambda x: CONSTRUCTION_PROTECTED_VALUES.get(x, None))
    return processed_features


def process_and_overlay_features(base_features, unchanged_features, changing_features, values):
    """
    Processing and overlaying features based on 'base_features', 'unchanged_features', 'changing_features', and 'values'.
    The purpose is to assign values to the base features based on the changing features.

    Parameters:
    - base_features: GeoDataFrame of base features (e.g., construction or compensatory features).
    - unchanged_features: GeoDataFrame of unchanged features.
    - changing_features: GeoDataFrame of changing features.
    - values: Dictionary of values to be assigned to the base features.

    Returns:
    - Processed GeoDataFrame.
    """
    # # Assuming unchanged_features and changing_features are defined globally or passed as parameters
    # global unchanged_features, changing_features, values

    # Rename 's_name' column in changing_features
    changing_features = changing_features.rename(
        columns={'s_name': 'changing_f'})

    # Punch holes
    changing_features = gpd.overlay(
        changing_features, unchanged_features, how='difference')

    # Overlay base_features with changing_features
    intersected_features = gpd.overlay(
        base_features, changing_features, how='intersection')

    # Select only the columns from base_features and add 'changing_f'
    intersected_features = intersected_features[base_features.columns]
    intersected_features['changing_f'] = changing_features['changing_f']

    # Flatten the result into a single geometry and keep the first unique value for each group
    base_features = intersected_features.dissolve(
        by='s_name', aggfunc='first').explode(index_parts=False)

    # Reset the index
    base_features.reset_index(drop=False, inplace=True)

    # Merge the base_features with the changing_features
    base_features['base_value'] = base_features['changing_f'].map(
        lambda x: get_value_with_warning(values, x))

    return base_features


# def calculate_overlay(changing_feature, buffer, context, file_base_name):
#     context = context
#     intersection = gpd.overlay(changing_feature, buffer, how='intersection')

#     return intersection


# def calculate_difference(feature1, feature2, context, file_base_name):

#     context = context
#     difference = gpd.overlay(
#         feature1, feature2, how='difference')

#     return difference

def calculate_overlay(feature1, feature2, operation, context, file_base_name):
    """
    Calculate the geometric overlay between two features.

    Parameters:
    - feature1: The first GeoDataFrame.
    - feature2: The second GeoDataFrame.
    - operation: A string specifying the overlay operation ('intersection' or 'difference').
    - context: The context in which this function is called. (Unused in this refactored version)
    - file_base_name: The base name for any output files. (Unused in this refactored version)

    Returns:
    - A GeoDataFrame resulting from the specified overlay operation between feature1 and feature2.
    """
    return gpd.overlay(feature1, feature2, how=operation)


def calculate_area(changing_feature):
    """
    Calculate the total area of all geometries in a GeoDataFrame.

    Parameters:
    - changing_feature: A GeoDataFrame whose total area is to be calculated.

    Returns:
    - The sum of the areas of all geometries in the GeoDataFrame.
    """
    return changing_feature.area.sum()


def filter_features(scope, features):
    """
    Filter features based on their spatial relationship to a given scope and their area.

    Parameters:
    - scope: A GeoDataFrame representing the area of interest.
    - features: A GeoDataFrame containing features to be filtered.

    Returns:
    - A GeoDataFrame containing features that are within or overlap the scope and have an area greater than 0.
    """
    if not scope.empty:
        print('Filtering features')
        # Filter features based on spatial relationship and area
        features = features[
            (features.geometry.within(scope.geometry.unary_union) |
             features.geometry.overlaps(scope.geometry.unary_union)) &
            (features.geometry.area > 0)
        ]
    return features

# ----> Value Assignment and Aggregation <----


def add_lagefaktor_values(feature, lagefaktor_value):
    """
    Adds or updates the 'lagefaktor' column in the feature GeoDataFrame.
    """
    if 'protected' in feature.columns:
        feature['lagefaktor'] = feature['protected'].fillna(lagefaktor_value)
        if lagefaktor_value == CONSTRUCTION_LAGEFAKTOR_VALUES.get('<100'):
            feature['lagefaktor'] -= 0.25

        feature = dissolve_sort_and_resolve(feature, 'lagefaktor')
    else:
        feature['lagefaktor'] = lagefaktor_value

    return feature


def add_compensatory_value(compensatory_features, protected_area_features):
    """
    Adds a 'compensat' column to compensatory_features based on 's_name' values.
    """
    compensatory_features['compensat'] = compensatory_features['s_name'].map(
        COMPENSATORY_MEASURE_VALUES)

    if not protected_area_features.empty:
        protected_area_features = protected_area_features.sort_values(
            by='protected', ascending=False)
        protected_area_features = resolve_overlaps(protected_area_features)
        compensatory_features = process_geodataframe_overlaps(
            compensatory_features, protected_area_features, 'protected')

    return compensatory_features


def process_geodataframe_overlaps(base_feature, cover_features, sort_by=None):
    """
    Processes base_feature and cover_features GeoDataFrames to handle overlaps and differences.
    """
    if not cover_features.empty:
        print('Processing cover features')
        cover_features = cover_features.sort_values(
            by=sort_by, ascending=False)
        cover_features = resolve_overlaps(cover_features)
        cover_features = cover_features.rename(
            columns={'s_name': f'{sort_by[:8]}_f'})

        overlapping_areas = gpd.overlay(
            base_feature, cover_features, how='intersection')
        non_overlapping_areas = gpd.overlay(
            base_feature, cover_features, how='difference')

        base_feature = gpd.overlay(non_overlapping_areas, overlapping_areas,
                                   how='union') if not non_overlapping_areas.empty and not overlapping_areas.empty else overlapping_areas
        base_feature = consolidate_columns(base_feature)

        base_feature = remove_slivers(base_feature, 0.001)

    return base_feature


def dissolve_sort_and_resolve(feature, by_column):
    """
    Dissolves, sorts, and resolves overlaps in a GeoDataFrame.
    """
    feature = feature.dissolve(by=by_column, aggfunc='first').reset_index()
    feature.sort_values(by=by_column, ascending=False, inplace=True)
    feature = resolve_overlaps(feature)
    feature = remove_slivers(feature, 0.001)
    return feature


def consolidate_columns(feature):
    """
    Consolidates columns with matching initial substrings in a GeoDataFrame.
    """
    non_geometry_columns = feature.columns.difference(['geometry'])
    for column in non_geometry_columns:
        matching_columns = [col for col in feature.columns if col.split('_')[
            0] == column.split('_')[0]]
        if len(matching_columns) < 2:
            continue
        matching_columns.sort(key=len)
        for matching_column in matching_columns[1:]:
            feature[matching_columns[0]] = feature[matching_columns[0]
                                                   ].combine_first(feature[matching_column])
        feature = feature.drop(columns=matching_columns[1:])
        base_name = matching_columns[0].rsplit('_', 1)[0]
        feature = feature.rename(columns={matching_columns[0]: base_name})
    return feature

# ----> Feature Separation and Finalization <----


def separate_features(scope, construction_feature, buffers, protected_area_features):
    global context
    # TODO: we can hondly handle a singl construction feature at this time (ususally Baufled)
    file_name = construction_feature['s_name'].iloc[0]

    # Calculate intersections
    changing_feature_B1_intersection = calculate_overlay(
        construction_feature, buffers[0], 'intersection', 'intersects with Buffer <100', file_name)
    changing_feature_B1_intersection = process_geodataframe_overlaps(
        changing_feature_B1_intersection, protected_area_features, 'protected')
    changing_feature_B1_intersection = add_lagefaktor_values(
        changing_feature_B1_intersection, CONSTRUCTION_LAGEFAKTOR_VALUES['<100'])
    changing_feature_B1_intersection = filter_features(scope,
                                                       changing_feature_B1_intersection)
    # add attribute buffer_distance
    changing_feature_B1_intersection['buffer_dis'] = '<100'

    changing_feature_B2_intersection = calculate_overlay(
        construction_feature, buffers[1], 'intersection', 'intersects with Buffer >100 <625', file_name)

    # Subtract changing_feature_B1_intersection from changing_feature_B2_intersection
    changing_feature_B2_not_B1 = calculate_overlay(
        changing_feature_B2_intersection, changing_feature_B1_intersection, 'difference',
        'intersects with Buffer >100 <625 but not Buffer <100', file_name)
    changing_feature_B2_not_B1 = process_geodataframe_overlaps(
        changing_feature_B2_not_B1, protected_area_features, 'protected')
    changing_feature_B2_not_B1 = add_lagefaktor_values(
        changing_feature_B2_not_B1, CONSTRUCTION_LAGEFAKTOR_VALUES['>100<625'])
    changing_feature_B2_not_B1 = filter_features(
        scope, changing_feature_B2_not_B1)
    changing_feature_B2_not_B1['buffer_dis'] = '>100<625'

    # Calculate area outside B2
    changing_feature_outside_B2 = calculate_overlay(
        construction_feature, buffers[1], 'difference', 'outside Buffer >625', file_name)
    changing_feature_outside_B2 = process_geodataframe_overlaps(
        changing_feature_outside_B2, protected_area_features, 'protected')
    changing_feature_outside_B2 = add_lagefaktor_values(
        changing_feature_outside_B2, CONSTRUCTION_LAGEFAKTOR_VALUES['>625'])
    changing_feature_outside_B2 = filter_features(
        scope, changing_feature_outside_B2)
    changing_feature_outside_B2['buffer_dis'] = '>625'

    # Calculate area
    changing_feature_outside_B2_area = calculate_area(
        changing_feature_outside_B2)
    changing_feature_B2_not_B1_area = calculate_area(
        changing_feature_B2_not_B1)
    changing_feature_B1_intersection_area = calculate_area(
        changing_feature_B1_intersection)

    # Print the results
    print_results(file_name, BUFFER_GEN_DISTANCES, changing_feature_B1_intersection_area,
                  changing_feature_B2_not_B1_area, changing_feature_outside_B2_area)

    return [
        {'shape': changing_feature_B1_intersection,
            'file_base_name': file_name, 'buffer_distance': BUFFER_GEN_DISTANCES[0]},
        {'shape': changing_feature_B2_not_B1, 'file_base_name': file_name,
            'buffer_distance': BUFFER_GEN_DISTANCES[1]},
        {'shape': changing_feature_outside_B2, 'file_base_name': file_name,
            'buffer_distance': BUFFER_GEN_DISTANCES[1]}
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


def process_scope(scope, construction_features, compensatory_features):
    # merge and flatten construction_features and compensatory_features
    if scope.empty:
        scope = gpd.overlay(construction_features,
                            compensatory_features, how='union')
    # Explode MultiPolygon geometries into individual Polygon geometries
    scope = scope.explode(index_parts=False)

    # Filter to only include polygons
    scope = scope[scope.geometry.type == 'Polygon']

    scope = remove_slivers(scope, 0.001)

    # Merge overlapping polygons
    # Add a column with the same value for all rows
    scope["group"] = 0
    scope = scope.dissolve(by="group")

    # Explode MultiPolygon geometries into individual Polygon geometries again
    scope = scope.explode(index_parts=False)

    # Reset index
    scope = scope.reset_index(drop=True)

    scope.crs = CRS

    return scope

# ----> Output Shapefile Creation <----


def print_results(file_base_name, buffer_distances, changing_feature_B1_area, changing_feature_B2_not_B1_area, changing_feature_outside_B2_area):
    print(
        f"Area of changing feature {file_base_name} intersecting with Buffer <{buffer_distances[0]}: {round(changing_feature_B1_area)}")
    print(
        f"Area of changing feature {file_base_name} intersecting with Buffer {buffer_distances[1]} but not {buffer_distances[0]}: {round(changing_feature_B2_not_B1_area)}")
    print(
        f"Area of changing feature {file_base_name} outside Buffer {buffer_distances[1]}: {round(changing_feature_outside_B2_area)}")


def create_lagefaktor_shapes(changing_features, file_base_name, buffer_distance):

    # Set the file name
    file_name = f'Construction_{file_base_name}_buffer_{buffer_distance}_intersection'

    # Set the directory name to be the same as the file name
    new_dir = f'{OUTPUT_DIR}/{file_name}'
    os.makedirs(new_dir, exist_ok=True)

    # Save the shape in the new directory
    changing_features.to_file(f'{new_dir}/{file_name}.shp')

# ----> Main Logic Flow <----


interference = get_features(INTERFERENCE_DIR)
buffers = get_buffers(interference, BUFFER_GEN_DISTANCES)
scope = get_features(SCOPE_DIR)
compensatory_features = get_features(COMPENSATORY_DIR)
construction_features = get_features(CONSTRUCTION_DIR)
changing_features = get_features(CHANGING_DIR)
unchanging_features = get_features(UNCHANGING_DIR)
protected_area_features = get_features(PROTECTED_DIR)

scope = process_scope(scope, construction_features, compensatory_features)

protected_area_features = preprocess_features(
    protected_area_features, 'protected_area')

construction_features = process_and_overlay_features(
    construction_features, unchanging_features, changing_features, CHANGING_CONSTRUCTION_BASE_VALUES)

compensatory_features = preprocess_features(
    compensatory_features, 'compensatory')
# def process_base_features(base_features, unchanged_features, changing_features, values):
compensatory_features = process_and_overlay_features(
    compensatory_features, unchanging_features, changing_features, CHANGING_COMPENSATORY_BASE_VALUES)
compensatory_features = add_compensatory_value(
    compensatory_features, protected_area_features)

output_shapes = []
output_shapes = separate_features(
    scope, construction_features, buffers, protected_area_features)


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

    # write attribute data to output_data
    column_names = [
        col for col in lagefaktor_shape['shape'].columns if col != 'geometry']

    for _, row in lagefaktor_shape['shape'].iterrows():
        attribute_data = [(column, row[column]) for column in column_names]
        output_data['construction'].append(attribute_data)

        create_lagefaktor_shapes(
            lagefaktor_shape['shape'], lagefaktor_shape['file_base_name'], lagefaktor_shape['buffer_distance'])

# create shapes for compensatory features, one shape file for each from in attribute 's_name'
total = 0
for file in compensatory_features['s_name'].unique():
    # Get the features for the current file
    current_features = compensatory_features[compensatory_features['s_name'] == file]
    current_features = filter_features(scope, current_features)

    # TODO: Alle Komp.Flächen müssen > 2000 m² sein ?
    def calculate_value(row):
        if row.geometry.area >= 2000:
            print('Calculating value for area >= 2000')
            final_v = (row['compensat'] - row['base_value']) * \
                row.geometry.area
            if 'protected' in current_features.columns and pd.notnull(row['protecte_f']):
                final_v = final_v * \
                    COMPENSATORY_PROTECTED_VALUES[row['protecte_f']]
            return final_v
        else:
            return 0

    current_features['final_v'] = current_features.apply(
        calculate_value, axis=1)

    # Summarize and add to total
    total += current_features['final_v'].sum()

    # write attribute data to output_data
    column_names = [
        col for col in current_features.columns if col != 'geometry']

    for _, row in current_features.iterrows():
        attribute_data = [(column, row[column]) for column in column_names]
        output_data['compensatory'].append(attribute_data)

    # Save the GeoDataFrame to a file in the output directory
    current_features.to_file(os.path.join(
        OUTPUT_DIR, 'Compensatory_' + file), driver='ESRI Shapefile')
print(f"Total final value for compensatory features: {round(total, 2)}")

# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(output_data)
