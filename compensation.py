# -*- coding: utf-8 -*-
from termcolor import colored
import unicodedata
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


def pt(df, table_name=None):

    # get the name of the calling function
    fn_name = sys._getframe(1).f_code.co_name

    max_length = df.drop(columns='geometry').apply(
        lambda x: x.map(lambda y: len(str(y)))).max().max()

    fixed_width_df = df.drop(columns='geometry').apply(
        lambda x: x.astype(str).apply(lambda y: unicodedata.normalize('NFC', y)[:max_length].ljust(max_length, ' ')))

    # Define the colors to use
    colors = ['\033[38;5;95m', '\033[38;5;130m', '\033[38;5;140m', '\033[38;5;105m', '\033[38;5;124m', '\033[38;5;160m', '\033[38;5;196m', '\033[38;5;202m', '\033[38;5;208m', '\033[38;5;214m', '\033[38;5;220m', '\033[38;5;226m', '\033[38;5;190m',
              '\033[38;5;154m', '\033[38;5;118m', '\033[38;5;82m', '\033[38;5;46m', '\033[38;5;47m', '\033[38;5;48m', '\033[38;5;49m', '\033[38;5;50m', '\033[38;5;51m', '\033[38;5;45m', '\033[38;5;39m', '\033[38;5;33m', '\033[38;5;27m', '\033[38;5;21m', '\033[0m']

    print()
    if fn_name:
        print(colored(fn_name, 'green'))
    print(colored(table_name, 'red'))  # prints table name in red

    # Print the column names with fixed width
    for i, name in enumerate(fixed_width_df.columns):
        print(colors[i % len(colors)] +
              name.ljust(max_length + 1, ' '), end='')

    print('\033[0m')  # Reset color

    # Print a line of dashes
    print('-' * (max_length + 1) * len(fixed_width_df.columns))

    # Print each row with fixed width columns
    for index, row in fixed_width_df.iterrows():
        for i, value in enumerate(row):
            print(colors[i % len(colors)] + value, end=' ')
        print('\033[0m')  # Reset color after each row

    print()


def normalize_string(input_string):
    normalized_string = unicodedata.normalize('NFC', input_string)
    encoded_string = normalized_string.encode(
        'ISO-8859-1', 'replace').decode('ISO-8859-1')
    return encoded_string


def get_value_with_warning(values, key):
    normalized_key = normalize_string(key)
    normalized_values = {normalize_string(k): v for k, v in values.items()}

    if normalized_key not in normalized_values:
        print(f"Warning: Value for {normalized_key} does not exist.")
        return None  # or return a default value

    value = normalized_values[normalized_key]
    return value

# ----> Feature Retrieval and Initialization <----


def read_shapefile(file_path):
    print(f"Reading shapefile {os.path.basename(file_path)}")
    feature = gpd.read_file(file_path)
    feature = feature.to_crs(CRS)
    feature = feature[['geometry']]
    s_name = os.path.basename(os.path.dirname(file_path))
    encoded_s_name = normalize_string(s_name)
    feature['s_name'] = encoded_s_name
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
    Resolves overlaps in geometries. Geometries are resolved by subtracting overlapping geometries from the original geometry.
    The sorting of the GeoDataFrame is important for the resolution of overlaps. 
    Precedence is given to the first geometry in the GeoDataFrame.

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
            lambda x: get_value_with_warning(CONSTRUCTION_PROTECTED_VALUES, x))
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

    pt(base_features, 'base_features')

    return base_features


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
    compensatory_features['compensat'] = compensatory_features['s_name'].apply(
        lambda x: get_value_with_warning(COMPENSATORY_MEASURE_VALUES, x))

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


def calculate_intersection_area(construction_feature, buffer, buffer_distance, protected_area_features, file_name, scope):
    """
    Calculate the intersection area of construction features with a buffer and process overlaps.
    """
    intersection = calculate_overlay(
        construction_feature, buffer, 'intersection', f'intersects with Buffer {buffer_distance}', file_name)
    intersection = process_geodataframe_overlaps(
        intersection, protected_area_features, 'protected')
    intersection = add_lagefaktor_values(
        intersection, CONSTRUCTION_LAGEFAKTOR_VALUES[buffer_distance])
    intersection = filter_features(scope, intersection)
    intersection['buffer_dis'] = buffer_distance
    return intersection


def process_and_separate_buffer_zones(scope, construction_feature, buffers, protected_area_features):
    """
    Separate features based on their intersection with different buffer zones and process them.
    """
    file_name = construction_feature['s_name'].iloc[0]

    # Calculate intersections for each buffer zone
    changing_feature_B1_intersection = calculate_intersection_area(
        construction_feature, buffers[0], BUFFER_DISTANCES['<100'], protected_area_features, file_name, scope)

    changing_feature_B2_intersection = calculate_intersection_area(
        construction_feature, buffers[1], BUFFER_DISTANCES['>100<625'], protected_area_features, file_name, scope)

    # Subtract changing_feature_B1_intersection from changing_feature_B2_intersection
    changing_feature_B2_not_B1 = calculate_overlay(
        changing_feature_B2_intersection, changing_feature_B1_intersection, 'difference',
        'intersects with Buffer >100 <625 but not Buffer <100', file_name)

    # Calculate area outside B2 by taking the difference between the construction feature and buffer B2
    changing_feature_outside_B2 = calculate_overlay(
        construction_feature, buffers[1], 'difference', 'outside Buffer >625', file_name)
    changing_feature_outside_B2 = process_geodataframe_overlaps(
        changing_feature_outside_B2, protected_area_features, 'protected')
    changing_feature_outside_B2 = add_lagefaktor_values(
        changing_feature_outside_B2, CONSTRUCTION_LAGEFAKTOR_VALUES[BUFFER_DISTANCES['>625']])
    changing_feature_outside_B2 = filter_features(
        scope, changing_feature_outside_B2)
    changing_feature_outside_B2['buffer_dis'] = BUFFER_DISTANCES['>625']

    # Calculate area for each feature
    changing_feature_outside_B2_area = calculate_area(
        changing_feature_outside_B2)
    changing_feature_B2_not_B1_area = calculate_area(
        changing_feature_B2_not_B1)
    changing_feature_B1_intersection_area = calculate_area(
        changing_feature_B1_intersection)

    # Print the results
    print_results(file_name, BUFFER_DISTANCES, changing_feature_B1_intersection_area,
                  changing_feature_B2_not_B1_area, changing_feature_outside_B2_area)

    features = pd.concat([changing_feature_B1_intersection, changing_feature_B2_not_B1,
                          changing_feature_outside_B2], ignore_index=True)
    return features
    # return [
    #     {'shape': changing_feature_B1_intersection, 'file_base_name': file_name,
    #         'buffer_distance': BUFFER_DISTANCES['<100']},
    #     {'shape': changing_feature_B2_not_B1, 'file_base_name': file_name,
    #         'buffer_distance': BUFFER_DISTANCES['>100<625']},
    #     {'shape': changing_feature_outside_B2, 'file_base_name': file_name,
    #         'buffer_distance': BUFFER_DISTANCES['>625']}
    # ]


def calculate_total_value(features, grz):
    """
    Calculate the total final value based on features and GRZ factors.

    Args:
        features (list of dict): List of feature dictionaries.
        grz (str): The GRZ factor.

    Returns:
        DataFrame: The features DataFrame with an additional 'score' column.
    """

    scores = []
    for _, feature in features.iterrows():
        area = feature.geometry.area
        total_value = feature['base_value'] * feature['lagefaktor'] * area
        factor_a, factor_b, factor_c = GRZ_FACTORS[grz]
        total_value_adjusted = total_value * factor_a * (factor_b + factor_c)
        score = round(total_value_adjusted, 2)
        scores.append(score)

    features['score'] = scores
    return features


def process_geometric_scope(scope, construction_features, compensatory_features, sliver_threshold=0.001):
    """
    Process and merge geometric features for a given scope.

    Args:
        scope (GeoDataFrame): The initial scope GeoDataFrame.
        construction_features (GeoDataFrame): GeoDataFrame of construction features.
        compensatory_features (GeoDataFrame): GeoDataFrame of compensatory features.
        sliver_threshold (float): Threshold for removing slivers.

    Returns:
        GeoDataFrame: The processed scope GeoDataFrame.
    """
    if scope.empty:
        scope = gpd.overlay(construction_features,
                            compensatory_features, how='union')

    scope = scope.explode(index_parts=False)
    scope = scope[scope.geometry.type == 'Polygon']
    scope = remove_slivers(scope, sliver_threshold)

    # Simplify merging overlapping polygons by assigning a constant group value
    scope['group'] = 0
    scope = scope.dissolve(by='group').explode(
        index_parts=False).reset_index(drop=True)
    scope.crs = CRS

    return scope

# ----> Output Shapefile Creation <----


def print_results(file_base_name, buffer_distances, changing_feature_B1_area, changing_feature_B2_not_B1_area, changing_feature_outside_B2_area):
    buffer_keys = list(buffer_distances.keys())
    print(
        f"Area of changing feature {file_base_name} intersecting with Buffer {buffer_keys[0]}: {round(changing_feature_B1_area)}")
    print(
        f"Area of changing feature {file_base_name} intersecting with Buffer {buffer_keys[1]} but not {buffer_keys[0]}: {round(changing_feature_B2_not_B1_area)}")
    print(
        f"Area of changing feature {file_base_name} outside Buffer {buffer_keys[1]}: {round(changing_feature_outside_B2_area)}")


def create_lagefaktor_shapes(changing_features, file_base_name, buffer_distance):

    # Set the file name
    file_name = f'Construction_{file_base_name}_buffer_{buffer_distance}_intersection'

    # Set the directory name to be the same as the file name
    new_dir = f'{OUTPUT_DIR}/{file_name}'
    os.makedirs(new_dir, exist_ok=True)

    # Save the shape in the new directory
    changing_features.to_file(f'{new_dir}/{file_name}.shp')


def process_features(directory, feature_type, unchanged_features, changing_features, changing_values):
    features = get_features(directory)
    features = preprocess_features(features, feature_type)
    features = process_and_overlay_features(
        features, unchanged_features, changing_features, changing_values)
    return features


def calculate_compensatory_score(features, scope):
    total = 0
    for file in features['s_name'].unique():
        current_features = filter_features(
            scope, features[features['s_name'] == file])
        current_features['score'] = current_features.apply(
            lambda row: calculate_value(row, current_features), axis=1)
        total += current_features['score'].sum()
        write_attribute_data(current_features, 'compensatory')
        save_features_to_file(current_features, 'Compensatory_' + file)
    return total


def write_attribute_data(features, feature_type):
    column_names = [col for col in features.columns if col != 'geometry']
    for _, row in features.iterrows():
        attribute_data = [(column, row[column]) for column in column_names]
        output_data[feature_type].append(attribute_data)


def save_features_to_file(features, filename):
    features.to_file(os.path.join(OUTPUT_DIR, filename),
                     driver='ESRI Shapefile')


def calculate_value(row, current_features):
    pt(current_features, 'current_features')
    if row.geometry.area >= 2000:
        print('Calculating value for area >= 2000')
        final_v = (row['compensat'] - row['base_value']) * row.geometry.area
        if 'protected' in current_features.columns and pd.notnull(row['protecte_f']):
            final_v = final_v * \
                get_value_with_warning(
                    COMPENSATORY_PROTECTED_VALUES, row['protecte_f'])
        return final_v
    else:
        return 0


def check_and_warn_column_length(df, limit):
    """
    Check the length of all columns in a DataFrame and issue a warning if any exceeds a limit.

    Args:
        df (pandas.DataFrame): The DataFrame to check.
        limit (int): The maximum allowed length.
    """
    for column_name in df.columns:
        if len(df[column_name]) > limit:
            print(
                f"Warning: The length of column '{column_name}' exceeds the limit of {limit}.")


# ----> Main Logic Flow <----


interference = get_features(INTERFERENCE_DIR)
buffers = get_buffers(interference, BUFFER_GEN_DISTANCES)
scope = get_features(SCOPE_DIR)

unchanging_features = get_features(UNCHANGING_DIR)
changing_features = get_features(CHANGING_DIR)

construction_features = process_features(
    CONSTRUCTION_DIR, 'construction', unchanging_features, changing_features, CHANGING_CONSTRUCTION_BASE_VALUES)

compensatory_features = process_features(
    COMPENSATORY_DIR, 'compensatory', unchanging_features, changing_features, CHANGING_COMPENSATORY_BASE_VALUES)

protected_area_features = get_features(PROTECTED_DIR)
protected_area_features = preprocess_features(
    protected_area_features, 'protected_area')

compensatory_features = add_compensatory_value(
    compensatory_features, protected_area_features)

construction_feature_buffer_zones = process_and_separate_buffer_zones(
    scope, construction_features, buffers, protected_area_features)


construction_feature_buffer_zones = calculate_total_value(
    construction_feature_buffer_zones, GRZ)
pt(construction_feature_buffer_zones, 'combined_features')
# sum up teh scores
total_final_value = construction_feature_buffer_zones['score'].sum()
print(f"Total final value: {total_final_value}")

# save construction features grouped by s_name
for file in construction_feature_buffer_zones['s_name'].unique():
    current_features = filter_features(
        scope, construction_feature_buffer_zones[construction_feature_buffer_zones['s_name'] == file])
    write_attribute_data(current_features, 'construction')
    save_features_to_file(current_features, 'Construction_' + file)
total = calculate_compensatory_score(compensatory_features, scope)
print(f"Total final value for compensatory features: {round(total, 2)}")
# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(output_data)
