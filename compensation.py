# -*- coding: utf-8 -*-
from matplotlib.legend import Legend
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import simplejson as sjson
import inspect
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
# DEFAULT_SLIVER = 0.0001
DEFAULT_SLIVER = 0.001

AREA_LIMIT = 1

# Create the parser
parser = argparse.ArgumentParser(
    description='Calculate the final value of construction and compensatory features and create shapefiles for each feature and JSON output.')

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
    'NSG': 1.5, 'VSG': 1.25, 'GGB': 1.25, 'Test': 10, 'Test2': 20}

# TODO:
CHANGING_COMPENSATORY_BASE_VALUES = {'Acker': 0, 'Grünland': 0, 'weg': 0}
COMPENSATORY_MEASURE_VALUES = {
    'Grünfläche': 3, "comp_test": 10}
COMPENSATORY_MEASURE_MINIMUM_AREAS = {
    'Grünfläche': 2000, "comp_test": 100}
COMPENSATORY_PROTECTED_VALUES = {
    'VSG': 1.1, 'GGB': 1.1, 'Test': 2, 'Test2': 4}

PROJECT_NAME = args.project

# Global debug counter dictionary
debug_counter_dict = {}


def get_calling_function_name():
    """
    Walk up the call stack and return the name of the first function found that is defined in the main script.
    """
    frame = inspect.currentframe()

    # Skip the first two frames
    frame = frame.f_back.f_back

    while frame:
        fn_module = inspect.getmodule(frame)
        if fn_module is not None and fn_module.__name__ == "__main__":
            return frame.f_code.co_name
        frame = frame.f_back

    return None


def get_calling_line_number():
    """
    Walk up the call stack and return the line number of the first call found that is defined in the main script.
    """
    frame = inspect.currentframe()

    # Skip the first two frames
    frame = frame.f_back.f_back

    while frame:
        fn_module = inspect.getmodule(frame)
        if fn_module is not None and fn_module.__name__ == "__main__":
            return frame.f_lineno
        frame = frame.f_back

    return None


def custom_warning(message, category, filename, lineno, file=None, line=None):
    """
    This function reads a shapefile from a given file path, transforms its CRS, and adds an encoded name column.

    Parameters:
    file_path (str): The file path from which to read the shapefile.

    Returns:
    GeoDataFrame: A GeoDataFrame containing the features from the shapefile, with an additional 'name' column 
    representing the encoded name of the shapefile's parent directory.
    """
    no_buffer_pattern = r"`?keep_geom_type=True`? in overlay resulted in .* dropped geometries of .* than .*\. Set `?keep_geom_type=False`? to retain all geometries"
    keepdims_pattern = r"<class 'geopandas.array.GeometryArray'>._reduce will require a `keepdims` parameter in the future"
    match_no_buffer = re.search(no_buffer_pattern, str(message))
    match_keepdims = re.search(keepdims_pattern, str(message))

    # Get the name of the calling function
    calling_fn_name = get_calling_function_name()
    calling_fn_line = get_calling_line_number()

    if match_no_buffer:
        # print(colored('Warning:', 'red') + f' {calling_fn_name}, line {str(calling_fn_line)}: ' +
        #       "During overlay operations, geometries such as lines or points that don't match the geometry type of the first DataFrame can be dropped.")
        pass
    elif not match_keepdims:
        print(colored('Warning:', 'red') + f' {calling_fn_name}, line {str(calling_fn_line)}: ' +
              str(message))


warnings.showwarning = custom_warning


def debug(gdf, prefix='', include_line_numbers=False):
    """
    This function writes a GeoDataFrame to a shapefile for debugging purposes.

    Parameters:
    gdf (GeoDataFrame): The GeoDataFrame to write.
    prefix (str, optional): An optional prefix to add to the filename.
    include_line_numbers (bool, optional): Whether to include line numbers in the filename.

    Returns:
    None
    """
    if args.debug:
        # Get the name of the calling function and line number
        frame = inspect.stack()[1]
        calling_function = frame.function
        if calling_function == '<module>':
            calling_function = 'main'
        line_number = frame.lineno

        # Get the line numbers of the entire call stack
        stack_line_numbers = '-'.join(
            str(frame.lineno) for frame in reversed(inspect.stack()[2:]))

        # Increment the counter for the calling function
        debug_counter_dict[calling_function] = debug_counter_dict.get(
            calling_function, 0) + 1

        # Increment the absolute counter for the debug function
        debug_counter_dict['debug'] = debug_counter_dict.get('debug', 0) + 1

        if prefix:
            prefix = '--' + prefix
        # Create the filename
        filename = os.path.join(
            DEBUG_DIR, f"{debug_counter_dict['debug']}_{calling_function}")
        if include_line_numbers:
            filename += f"-{stack_line_numbers}-{line_number}"
        filename += f"{prefix}_#{debug_counter_dict[calling_function]}.shp"

        # Write the GeoDataFrame to a shapefile
        gdf.to_file(filename)


def pt(df, table_name=None):
    """
    This function prints a DataFrame in a fixed-width format.

    Parameters:
    df (DataFrame): The DataFrame to print.
    table_name (str, optional): The name of the table. Defaults to None.

    Returns:
    None
    """
    # get the name of the calling function
    fn_name = sys._getframe(1).f_code.co_name

    max_length_values = df.drop(columns='geometry').apply(
        lambda x: x.map(lambda y: len(str(y)))).max().max()

    max_length_columns = max([len(col) for col in df.columns])

    max_length = max(max_length_values, max_length_columns)

    fixed_width_df = df.drop(columns='geometry').apply(
        lambda x: x.astype(str).apply(lambda y: unicodedata.normalize('NFC', y)[:max_length].ljust(max_length, ' ')))

    # Define the colors to use
    colors = ['\033[38;5;95m', '\033[38;5;160m', '\033[38;5;140m', '\033[38;5;202m', '\033[38;5;124m', '\033[38;5;214m', '\033[38;5;196m', '\033[38;5;105m', '\033[38;5;130m', '\033[38;5;220m', '\033[38;5;208m', '\033[38;5;154m', '\033[38;5;190m',
              '\033[38;5;82m', '\033[38;5;226m', '\033[38;5;48m', '\033[38;5;46m', '\033[38;5;51m', '\033[38;5;47m', '\033[38;5;50m', '\033[38;5;45m', '\033[38;5;49m', '\033[38;5;39m', '\033[38;5;33m', '\033[38;5;27m', '\033[38;5;21m', '\033[0m']
    print()
    if fn_name:
        print(colored(f'Calling Fn: {fn_name}', 'green'))
    # prints table name in red
    print(colored(f'Table Name: {table_name}', 'red'))

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
    """
    This function normalizes a string using NFC normalization and encodes it in ISO-8859-1.

    Parameters:
    input_string (str): The string to normalize.

    Returns:
    str: The normalized string.
    """
    normalized_string = unicodedata.normalize('NFC', input_string)
    encoded_string = normalized_string.encode(
        'ISO-8859-1', 'replace').decode('ISO-8859-1')
    return encoded_string


def get_value_with_warning(values, key):
    """
    This function retrieves a value from a dictionary, printing a warning if the key is not found.

    Parameters:
    values (dict): The dictionary from which to retrieve the value.
    key (str): The key of the value to retrieve.

    Returns:
    The value associated with the key, or None if the key is not found.
    """
    normalized_key = normalize_string(key)
    normalized_values = {normalize_string(k): v for k, v in values.items()}

    if normalized_key not in normalized_values:
        print(f"Warning: Value for {normalized_key} does not exist.")
        return None  # or return a default value

    value = normalized_values[normalized_key]
    return value


def check_and_warn_column_length(df, column_name_limit=10, value_length_limit=255):
    """
    Check the length of all column names and string values in a DataFrame and issue a warning if any exceeds their respective limits.

    Args:
        df (pandas.DataFrame): The DataFrame to check.
        column_name_limit (int): The maximum allowed length for column names.
        value_length_limit (int): The maximum allowed length for string values.
    """
    for column_name in df.columns:
        # Check length of column name
        if len(column_name) > column_name_limit:
            warnings.warn(
                f"Warning: The length of column name '{column_name}' exceeds the limit of {column_name_limit}.")

        # Check length of string values in the column
        if df[column_name].dtype == 'object':
            too_long = df[column_name].astype(
                str).apply(len) > value_length_limit
            if too_long.any():
                warnings.warn(
                    f"Warning: Some values in column '{column_name}' exceed the limit of {value_length_limit}.")
# ----> Feature Retrieval and Initialization <----


def read_shapefile(file_path):
    """
    This function reads a shapefile from a given file path, transforms its CRS, and adds an encoded name column.

    Parameters:
    file_path (str): The file path from which to read the shapefile.

    Returns:
    GeoDataFrame: A GeoDataFrame containing the features from the shapefile, with an additional 'name' column 
    representing the encoded name of the shapefile's parent directory.
    """
    s_name = os.path.basename(os.path.dirname(file_path))
    encoded_s_name = normalize_string(s_name)
    print(colored(
        f"  {encoded_s_name}/{os.path.basename(file_path)}", 'yellow', attrs=['dark']))
    feature = gpd.read_file(file_path)
    feature = feature.to_crs(CRS)
    feature = feature[['geometry']]
    feature['name'] = encoded_s_name
    return feature


def get_features(dir):
    """
    This function reads shapefiles from a given directory and returns a GeoDataFrame of the features.

    Parameters:
    dir (str): The directory from which to read the shapefiles.

    Returns:
    GeoDataFrame: A GeoDataFrame containing the features from the shapefiles.
    """
    print(colored(
        f'Reading shapefiles from directory "{os.path.basename(dir)}":', 'yellow', attrs=['dark']))
    shapefiles = glob.glob(f"{dir}/*/*.shp")

    if not shapefiles:
        print(
            colored(f"No shapefiles found in directory {dir}", 'yellow', attrs=['dark']))
        return gpd.GeoDataFrame(columns=['geometry', 'name'], crs=CRS)

    features = [read_shapefile(shapefile) for shapefile in shapefiles]
    features_gdf = pd.concat(features, ignore_index=True)
    features_gdf = features_gdf.to_crs(CRS)

    return features_gdf

# ----> Buffer Operations <----


def create_buffer(linestrings, distance):
    """
    This function creates a buffer around each linestring and dissolves all geometries into a single one.

    Parameters:
    linestrings (GeoSeries): The linestrings around which to create buffers.
    distance (float): The distance for the buffer.

    Returns:
    GeoDataFrame: A GeoDataFrame containing the buffers.
    """
    # Create a buffer around each linestring and dissolve all geometries into a single one
    buffers = linestrings.buffer(distance).to_frame().rename(
        columns={0: 'geometry'}).set_geometry('geometry').dissolve()
    return buffers


def get_buffers(features, distances):
    """
    This function creates buffers around the given features for each specified distance.

    Parameters:
    features (GeoDataFrame): The geospatial features for which to create buffers.
    distances (list): A list of distances for which to create buffers.

    Returns:
    list: A list of GeoDataFrames, each representing the buffers around the features at a specific distance.
    """
    # Create a buffer for each distance and return the list of buffers
    return [create_buffer(features, distance) for distance in distances]

# ----> Feature Cleanup and Geometry Manipulation <----


def cleanup_and_merge_features(feature, buffer_distance):
    """
    Cleans up and merges features based on their geometry and 'name'.

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
               .dissolve(by='name')
               .reset_index())

    # Set the CRS
    feature.crs = CRS

    # Merge with original string columns
    original_strings = original_strings.loc[feature.index].reset_index(
        drop=True)
    original_strings.drop(columns='name', inplace=True)
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


def remove_slivers(gdf, buffer_distance=DEFAULT_SLIVER):
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
    - feature_type: Type of features being processed ('compensatory' or 'protected_area').
    - buffer_distance: Buffer distance for cleanup and merge operation, default is 10.

    Returns:
    - Processed GeoDataFrame.
    """
    # Cleanup and merge features
    processed_features = cleanup_and_merge_features(
        features, buffer_distance=buffer_distance)

    if feature_type == 'compensatory':
        # Assign 'compensat' based on 'name'
        processed_features['compensat'] = processed_features['name'].map(
            lambda x: get_value_with_warning(COMPENSATORY_MEASURE_VALUES, x))
    elif feature_type == 'protected_area':
        # Set 'prot_cons' and 'prot_comp' based on 'name'
        processed_features['prot_cons'] = processed_features['name'].apply(
            lambda x: get_value_with_warning(CONSTRUCTION_PROTECTED_VALUES, x))
        processed_features['prot_comp'] = processed_features['name'].apply(
            lambda x: get_value_with_warning(COMPENSATORY_PROTECTED_VALUES, x))
        processed_features = processed_features.rename(
            columns={'name': 'prot_name'})
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

    # Rename 'name' column in changing_features
    changing_features = changing_features.rename(
        columns={'name': 'base_name'})

    # Punch holes
    changing_features = gpd.overlay(
        changing_features, unchanged_features, how='difference')

    # Overlay base_features with changing_features
    intersected_features = gpd.overlay(
        base_features, changing_features, how='intersection')

    # Select only the columns from base_features and add 'base_name'
    intersected_features = intersected_features[base_features.columns]
    intersected_features['base_name'] = changing_features['base_name']

    # # Flatten the result into a single geometry and keep the first unique value for each group
    # base_features = intersected_features.dissolve(
    #     by='name', aggfunc='first').explode(index_parts=False)

    # # Reset the index
    # base_features.reset_index(drop=False, inplace=True)

    # # Merge the base_features with the changing_features
    # base_features['base_value'] = base_features['base_name'].map(
    #     lambda x: get_value_with_warning(values, x))

    intersected_features['base_value'] = intersected_features['base_name'].map(
        lambda x: get_value_with_warning(values, x))

    # return base_features
    return intersected_features


def calculate_overlay(feature1, feature2, operation):
    """
    Calculate the geometric overlay between two features.

    Parameters:
    - feature1: The first GeoDataFrame.
    - feature2: The second GeoDataFrame.
    - operation: A string specifying the overlay operation ('intersection' or 'difference').

    Returns:
    - A GeoDataFrame resulting from the specified overlay operation between feature1 and feature2.
    """
    gdf = gpd.overlay(feature1, feature2, how=operation)
    gdf = remove_slivers(gdf)
    return gdf


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
    This function adds 'lagefaktor' values to the given feature GeoDataFrame.

    Parameters:
    feature (GeoDataFrame): The GeoDataFrame to which to add 'lagefaktor' values.
    lagefaktor_value (float): The 'lagefaktor' value to add.

    Returns:
    GeoDataFrame: The updated GeoDataFrame with 'lagefaktor' values.
    """

    if 'prot_cons' in feature.columns:
        # Check if 'prot_cons' is not null
        is_protected_not_null = feature['prot_cons'].notnull()

        feature['lagefaktor'] = feature['prot_cons'].fillna(lagefaktor_value)
        if lagefaktor_value == CONSTRUCTION_LAGEFAKTOR_VALUES.get('<100'):
            # Only subtract 0.25 from 'lagefaktor' if 'prot_cons' is not null
            feature.loc[is_protected_not_null, 'lagefaktor'] -= 0.25
    else:
        feature['lagefaktor'] = lagefaktor_value

    # remove column prot_comp if it exists
    if 'prot_comp' in feature.columns:
        feature = feature.drop(columns='prot_comp')

    return feature


def add_compensatory_value(compensatory_features, protected_area_features):
    """
    This function adds compensatory values to the given compensatory features GeoDataFrame.

    Parameters:
    compensatory_features (GeoDataFrame): The GeoDataFrame to which to add compensatory values.
    protected_area_features (GeoDataFrame): The GeoDataFrame of protected area features.

    Returns:
    GeoDataFrame: The updated GeoDataFrame with compensatory values.
    """
    compensatory_features['compensat'] = compensatory_features['name'].apply(
        lambda x: get_value_with_warning(COMPENSATORY_MEASURE_VALUES, x))

    # Add 'eligible' column
    compensatory_features['eligible'] = compensatory_features.apply(
        lambda row: row['geometry'].area > get_value_with_warning(
            COMPENSATORY_MEASURE_MINIMUM_AREAS, row['name']), axis=1)

    if not protected_area_features.empty:
        protected_area_features = protected_area_features.sort_values(
            by='prot_comp', ascending=False)
        protected_area_features = resolve_overlaps(protected_area_features)
        compensatory_features = process_geodataframe_overlaps(
            compensatory_features, protected_area_features)
        compensatory_features = compensatory_features.drop(columns='prot_cons')

    return compensatory_features


def process_geodataframe_overlaps(base_feature, cover_features):
    """
    This function processes overlaps in a GeoDataFrame.

    Parameters:
    base_feature (GeoDataFrame): The base GeoDataFrame.
    cover_features (GeoDataFrame): The GeoDataFrame of features that may overlap with the base features.
    sort_by (str, optional): The column by which to sort the GeoDataFrames. Defaults to None.

    Returns:
    GeoDataFrame: The processed GeoDataFrame with overlaps resolved.
    """
    if not cover_features.empty:
        # cover_features = cover_features.sort_values(
        #     by=sort_by, ascending=False)
        cover_features = resolve_overlaps(cover_features)

        # new_column_name = f'{sort_by[:8]}_t'
        # cover_features = cover_features.rename(
        #     columns={'name': new_column_name})

        # Print old and new column name in one line
        # print(
        #     f"During overlap operation, renaming: {sort_by} -> {new_column_name}")

        overlapping_areas = gpd.overlay(
            base_feature, cover_features, how='intersection')

        non_overlapping_areas = gpd.overlay(
            base_feature, cover_features, how='difference')

        base_feature = gpd.overlay(non_overlapping_areas, overlapping_areas,
                                   how='union') if not non_overlapping_areas.empty and not overlapping_areas.empty else overlapping_areas

        base_feature = consolidate_columns(base_feature)
        base_feature = remove_slivers(base_feature)

    return base_feature


def dissolve_sort_and_resolve(feature, by_column):
    """
    This function dissolves, sorts, and resolves overlaps in a GeoDataFrame.

    Parameters:
    feature (GeoDataFrame): The GeoDataFrame to process.
    by_column (str): The column by which to dissolve and sort the GeoDataFrame.

    Returns:
    GeoDataFrame: The processed GeoDataFrame.
    """
    feature = feature.dissolve(by=by_column, aggfunc='first').reset_index()
    feature.sort_values(by=by_column, ascending=False, inplace=True)
    feature = resolve_overlaps(feature)
    feature = remove_slivers(feature)
    return feature


def consolidate_columns(feature):
    """
    This function consolidates columns in a GeoDataFrame.

    Parameters:
    feature (GeoDataFrame): The GeoDataFrame to process.

    Returns:
    GeoDataFrame: The processed GeoDataFrame with consolidated columns.
    """
    non_geometry_columns = feature.columns.difference(['geometry'])
    for column in non_geometry_columns:
        base_name = re.split('_\d+', column)[0]
        matching_columns = [col for col in feature.columns if re.split(
            '_\d+', col)[0] == base_name]
        if len(matching_columns) < 2:
            continue
        matching_columns.sort(key=len)
        for matching_column in matching_columns[1:]:
            feature[matching_columns[0]] = feature[matching_columns[0]
                                                   ].combine_first(feature[matching_column])
        feature = feature.drop(columns=matching_columns[1:])
        old_column_name = matching_columns[0]
        feature = feature.rename(columns={old_column_name: base_name})

        # Print old and new column name
        print(f"While consolidating columns: {old_column_name} -> {base_name}")

    return feature

# ----> Feature Separation and Finalization <----


def calculate_intersection_area(construction_feature, buffer, buffer_distance, protected_area_features, scope):
    """
    This function calculates the intersection area of a construction feature and a buffer.

    Parameters:
    construction_feature (GeoDataFrame): The construction feature.
    buffer (GeoDataFrame): The buffer.
    buffer_distance (float): The buffer distance.
    protected_area_features (GeoDataFrame): The protected area features.
    scope (str): The scope of the calculation.

    Returns:
    GeoDataFrame: The intersection area.
    """
    intersection = calculate_overlay(
        construction_feature, buffer, 'intersection')
    intersection = process_geodataframe_overlaps(
        intersection, protected_area_features)
    intersection = add_lagefaktor_values(
        intersection, CONSTRUCTION_LAGEFAKTOR_VALUES[buffer_distance])
    intersection = filter_features(scope, intersection)
    intersection['buffer_dis'] = buffer_distance
    return intersection


def process_and_separate_buffer_zones(scope, construction_feature, buffers, protected_area_features):
    """
    This function processes and separates buffer zones.

    Parameters:
    scope (str): The scope of the processing.
    construction_feature (GeoDataFrame): The construction feature.
    buffers (list): The list of buffers.
    protected_area_features (GeoDataFrame): The protected area features.

    Returns:
    GeoDataFrame: The processed and separated buffer zones.
    """

    # Sort protected_area_features by 'prot_cons' in descending order
    protected_area_features = protected_area_features.sort_values(
        by='prot_cons', ascending=False)

    # Calculate intersections for each buffer zone
    changing_feature_B1_intersection = calculate_intersection_area(
        construction_feature, buffers[0], BUFFER_DISTANCES['<100'], protected_area_features, scope)

    # Calculate intersection for buffer B2
    changing_feature_B2_intersection = calculate_intersection_area(
        construction_feature, buffers[1], BUFFER_DISTANCES['>100<625'], protected_area_features, scope)

    # Subtract changing_feature_B1_intersection from changing_feature_B2_intersection
    changing_feature_B2_not_B1 = calculate_overlay(
        changing_feature_B2_intersection, changing_feature_B1_intersection, 'difference')

    # Calculate area outside B2 by taking the difference between the construction feature and buffer B2
    changing_feature_outside_B2 = calculate_overlay(
        construction_feature, buffers[1], 'difference')
    changing_feature_outside_B2 = process_geodataframe_overlaps(
        changing_feature_outside_B2, protected_area_features)
    changing_feature_outside_B2 = add_lagefaktor_values(
        changing_feature_outside_B2, CONSTRUCTION_LAGEFAKTOR_VALUES[BUFFER_DISTANCES['>625']])
    changing_feature_outside_B2 = filter_features(
        scope, changing_feature_outside_B2)
    changing_feature_outside_B2['buffer_dis'] = BUFFER_DISTANCES['>625']

    features = pd.concat([changing_feature_B1_intersection, changing_feature_B2_not_B1,
                          changing_feature_outside_B2], ignore_index=True)

    return features


def add_construction_score(features, grz):
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


def process_geometric_scope(scope, construction_features, compensatory_features, sliver_threshold):
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

    # TODO does this even work?
    # Simplify merging overlapping polygons by assigning a constant group value
    scope['group'] = 0
    scope = scope.dissolve(by='group').explode(
        index_parts=False).reset_index(drop=True)
    scope.crs = CRS

    return scope

# ----> Output Shapefile Creation <----


def process_features(directory, feature_type, unchanged_features, changing_features, changing_values):
    """
    This function processes geospatial features from a given directory.

    Parameters:
    directory (str): The directory from which to read the features.
    feature_type (str): The type of the features.
    unchanged_features (GeoDataFrame): The features that remain unchanged.
    changing_features (GeoDataFrame): The features that are changing.
    changing_values (list): The values that are changing.

    Returns:
    GeoDataFrame: The processed features.
    """
    features = get_features(directory)
    features = preprocess_features(features, feature_type)
    features = process_and_overlay_features(
        features, unchanged_features, changing_features, changing_values)
    return features


def calculate_compensatory_score(row, current_features):
    """
    This function calculates the compensatory score for a row in a GeoDataFrame.

    Parameters:
    row (GeoSeries): The row for which to calculate the compensatory score.
    current_features (GeoDataFrame): The current features.

    Returns:
    float: The compensatory score.
    """
    if row['eligible'] == True:
        final_v = (row['compensat'] - row['base_value']) * row.geometry.area
        if 'prot_comp' in current_features.columns and pd.notnull(row['prot_comp']):
            final_v = final_v * \
                get_value_with_warning(
                    COMPENSATORY_PROTECTED_VALUES, row['prot_name'])
        return final_v
    else:
        return 0


def add_compensatory_score(features, scope):
    """
    This function adds compensatory scores to a GeoDataFrame of features.

    Parameters:
    features (GeoDataFrame): The features to which to add compensatory scores.
    scope (str): The scope of the operation.

    Returns:
    GeoDataFrame: The features with added compensatory scores.
    """
    all_features = pd.DataFrame()
    for file in features['name'].unique():
        current_features = filter_features(
            scope, features[features['name'] == file])
        current_features['score'] = round(current_features.apply(
            lambda row: calculate_compensatory_score(row, current_features), axis=1), 2)
        all_features = pd.concat([all_features, current_features])
    return all_features


def save_features_to_file(features, filename, interference, scope):
    """
    This function saves a GeoDataFrame of features to a file.

    Parameters:
    features (GeoDataFrame): The features to save.
    filename (str): The name of the file to which to save the features.

    Returns:
    None
    """
    features.to_file(os.path.join(OUTPUT_DIR, filename),
                     driver='ESRI Shapefile')

    pt(features)
    # Assuming 'features' is a GeoDataFrame
    fig, ax = plt.subplots(1, 1)

    column_name = ''

    if 'buffer_dis' in features.columns:
        column_name = 'buffer_dis'

        # Define a colormap with enough colors for each unique value in 'buffer_dis'
        cmap = ListedColormap(plt.cm.viridis(np.linspace(
            0, 1, len(features[column_name].unique()))))

        # Plot 'buffer_dis' with the colormap
        features_map = features.plot(column=column_name, ax=ax, cmap=cmap)

        # Plot 'interference' on the same axes
        interference.plot(ax=ax, color='red')

        # Plot 'scope' on the same axes with dashed lines and no fill
        scope.boundary.plot(ax=ax, color='black', linestyle='dashed')

        # Create a legend entry for each unique value in 'buffer_dis'
        buffer_dis_patches = [mpatches.Patch(color=cmap(
            i), label=label) for i, label in enumerate(features[column_name].unique())]

        # Convert 'buffer_dis_patches' and 'features[column_name].unique()' to lists
        buffer_dis_patches = list(buffer_dis_patches)
        buffer_dis_labels = list(features[column_name].unique())

        # Create a legend for 'buffer_dis'
        legend1 = ax.legend(handles=buffer_dis_patches,
                            labels=buffer_dis_labels,
                            title='Buffer Distance',  # Add title here
                            loc='upper left', bbox_to_anchor=(1.05, 1))
        # Add the first legend manually to the current Axes.
        ax.add_artist(legend1)

        # Create a legend entry for 'interference'
        interference_patch = mpatches.Patch(color='red', label='Interference')

        # Create a second legend for 'interference', use the same location parameters
        ax.legend(handles=[interference_patch],
                  labels=['Interference'],
                  title='Interference',
                  loc='upper left', bbox_to_anchor=(1.05, 0.5))
    elif 'compensat' in features.columns:
        column_name = 'compensat'
        # Define the color map and norm
        colors = ['red', 'green', 'blue']  # replace with the colors you want
        bounds = [1, 2, 3, 4]  # replace with the boundaries you want
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Plot the GeoDataFrame without the legend
        features.plot(column=column_name, ax=ax, cmap=cmap, norm=norm)

        # Plot 'scope' on the same axes with dashed lines and no fill
        scope.boundary.plot(ax=ax, color='black', linestyle='dashed')

        # Get the unique values in the column
        unique_values = features[column_name].unique()

        # Create a legend entry for each unique value
        legend_patches = [mpatches.Patch(color=cmap(
            norm(value)), label=value) for value in unique_values]

        # Add the legend to the plot
        legend = ax.legend(handles=legend_patches)

        # Set the title of the legend
        legend.set_title("Kompensationswert")

    plt.title(PROJECT_NAME)
    plt.show()


def write_output_json_and_excel(total_score, data, filename='output'):
    """
    This function writes output data to a JSON file and an Excel file.

    Parameters:
    total_score (float): The total score to write.
    data (GeoDataFrame): The data to write.
    filename (str, optional): The name of the files to which to write the data. Defaults to 'output'.

    Returns:
    None
    """
    data = data.copy()
    data['area'] = data.geometry.area.round(2)
    data = data.drop(columns='geometry')

    output_dict = {}
    for name, group in data.groupby('name'):
        output_dict[name] = group.to_dict('records')

    # Create a new dictionary and add total_score to it first
    final_output_dict = {'total_score': total_score}
    # Update the new dictionary with output_dict
    final_output_dict.update(output_dict)

    with open(os.path.join(OUTPUT_DIR, PROJECT_NAME + '_' + filename + '.json'), 'w') as file:
        sjson.dump(final_output_dict, file, ignore_nan=True,
                   ensure_ascii=False, indent=4)

    # Convert the data to a DataFrame and write it to an Excel file
    df = pd.DataFrame(data)

    if filename == 'Construction':
        # Rename the columns to a more readable format
        df = df.rename(columns={
            "name": "Name",
            "base_name": "Bestandsfläche (base_name)",
            "base_value": "Bestandsflächenwert (base_value)",
            "prot_name": "Schutzgebiet (prot_name)",
            "prot_cons": "Schutzstgebietsfaktor (prot_cons)",
            "lagefaktor": "Lagefaktor (lagefaktor)",
            "buffer_dis": "Pufferzone (buffer_dis)",
            "score": "Punktzahl (score)",
            "area": "Fläche (area)"
        })

    if filename == 'Compensatory':
        # Rename the columns to a more readable format
        df = df.rename(columns={
            "name": "Name",
            "base_name": "Bestandsfläche (base_name)",
            "base_value": "Bestandsflächenwert (base_value)",
            "compensat": "Kompensationswert (compensat)",
            "eligible": "Berechtigt (eligible)",
            "prot_name": "Schutzgebiet (prot_name)",
            "prot_comp": "Schutzstgebietsfaktor (prot_comp)",
            "score": "Punktzahl (score)",
            "area": "Fläche (area)"
        })

    # Append total_score to the bottom of the DataFrame using pd.concat
    df = pd.concat(
        [df, pd.DataFrame({'Punktzahl': [total_score]})], ignore_index=True)

    df.to_excel(os.path.join(OUTPUT_DIR, PROJECT_NAME + '_' +
                filename + '.xlsx'), index=False)


def filter_area_limit(gdf, limit):
    """
    Filter out areas that are less than the given limit.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame to filter.
        limit (float): The minimum area limit.

    Returns:
        GeoDataFrame: The filtered GeoDataFrame.
    """
    return gdf[gdf.geometry.area > limit]


def merge_by_combination(gdf, type):
    """

    """
    # Check if type is 'construction'
    if type == 'construction':
        by_cols = ['name', 'base_name', 'lagefaktor', 'buffer_dis']
        if 'prot_cons' in gdf.columns:
            by_cols.append('prot_cons')
        gdf = gdf.dissolve(by=by_cols, aggfunc='first').reset_index()
    # Check if type is 'compensatory'
    elif type == 'compensatory':
        by_cols = ['base_name', 'name', 'compensat']
        if 'prot_comp' in gdf.columns:
            by_cols.append('prot_comp')
        gdf = gdf.dissolve(by=by_cols, aggfunc='first').reset_index()
    return gdf


# ----> Main Logic Flow <----


interference = get_features(INTERFERENCE_DIR)
buffers = get_buffers(interference, BUFFER_GEN_DISTANCES)
scope = get_features(SCOPE_DIR)

unchanging_features = get_features(UNCHANGING_DIR)
changing_features = get_features(CHANGING_DIR)

construction_features = process_features(
    CONSTRUCTION_DIR, 'construction', unchanging_features, changing_features, CHANGING_CONSTRUCTION_BASE_VALUES)
debug(construction_features, '1 construction_features')

compensatory_features = process_features(
    COMPENSATORY_DIR, 'compensatory', unchanging_features, changing_features, CHANGING_COMPENSATORY_BASE_VALUES)

protected_area_features = get_features(PROTECTED_DIR)
protected_area_features = preprocess_features(
    protected_area_features, 'protected_area')

compensatory_features = add_compensatory_value(
    compensatory_features, protected_area_features)

construction_feature_buffer_zones = process_and_separate_buffer_zones(
    scope, construction_features, buffers, protected_area_features)
debug(construction_feature_buffer_zones, '2 construction_feature_buffer_zones')

# compensatory_features = merge_by_combination(
#     compensatory_features, 'compensatory')
# compensatory_features = filter_area_limit(compensatory_features, 10)

# construction_feature_buffer_zones = merge_by_combination(
#     construction_feature_buffer_zones, 'construction')
# construction_feature_buffer_zones = filter_area_limit(
#     construction_feature_buffer_zones, 10)

# ---> Construction Output Shapefile Creation <---

print()

construction_feature_buffer_zones = add_construction_score(
    construction_feature_buffer_zones, GRZ)

total_construction_score = round(
    construction_feature_buffer_zones['score'].sum(), 2)
print(colored(
    f"Total Construction score: {total_construction_score}", 'yellow'))

for file in construction_feature_buffer_zones['name'].unique():
    current_features = filter_features(
        scope, construction_feature_buffer_zones[construction_feature_buffer_zones['name'] == file])
    check_and_warn_column_length(current_features)
    save_features_to_file(
        current_features, 'Construction_' + file, interference, scope)

write_output_json_and_excel(total_construction_score, construction_feature_buffer_zones,
                            'Construction')

# ---> Compensatory Output Shapefile Creation <---

compensatory_features = add_compensatory_score(compensatory_features, scope)

total_compensatory_score = round(compensatory_features['score'].sum(), 2)
print(colored(
    f"Total Compensatory score: {total_compensatory_score}", 'yellow'))

for file in compensatory_features['name'].unique():
    current_features = filter_features(
        scope, compensatory_features[compensatory_features['name'] == file])
    check_and_warn_column_length(current_features)
    save_features_to_file(
        current_features, 'Compensatory_' + file, interference, scope)

write_output_json_and_excel(total_compensatory_score,
                            compensatory_features, 'Compensatory')
