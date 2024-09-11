# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import argparse

import openpyxl
import yaml
from gis_lagefaktor.data_handling import get_features, save_to_shapefile, write_output_json_and_excel, check_and_warn_column_length, get_parcel_features
from gis_lagefaktor.feature_processing import process_features, add_compensatory_score, process_and_separate_buffer_zones
from gis_lagefaktor.geospatial_ops import get_buffers, filter_features
from gis_lagefaktor.visualization import create_plot
import os
import shutil

import sys
from gis_lagefaktor.debugging import do_debug, pt
from gis_lagefaktor.custom_warning import custom_warning
from gis_lagefaktor.feature_processing import preprocess_features
from gis_lagefaktor.geospatial_ops import remove_geometries_with_small_areas
from gis_lagefaktor.feature_processing import add_compensatory_value, add_construction_score
from termcolor import colored
import warnings
import gis_lagefaktor.config as config
import geopandas as gpd
import pandas as pd

warnings.showwarning = custom_warning

if len(sys.argv) == 1:
    print("No arguments provided. Please run the script with the required arguments.")
    sys.exit()

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

# Update settings with command line arguments if provided
if args.project:
    config.settings.project_name = args.project

# Load the base configuration
config.load_config()

# Load project-specific configuration
config.load_project_config()

GRZ = config.settings.projects[config.settings.project_name].grz

# Define directories
dir_path = config.settings.projects[config.settings.project_name].path
SCOPE_PATH = os.path.join(dir_path, 'scope')
CHANGING_PATH = os.path.join(dir_path, 'changing')
CONSTRUCTION_PATH = os.path.join(dir_path, 'construction')
UNCHANGING_PATH = os.path.join(dir_path, 'unchanging')
COMPENSATORY_PATH = os.path.join(dir_path, 'compensatory')
PROTECTED_PATH = os.path.join(dir_path, 'protected')
OUTPUT_PATH = os.path.join(dir_path, 'output')
DEBUG_PATH = os.path.join(dir_path, 'debug')
INTERFERENCE_PATH = os.path.join(dir_path, 'interference')
PARCEL_PATH = os.path.join(dir_path, 'parcel')

# List of directories to create
dirs = [dir_path, SCOPE_PATH, CHANGING_PATH, CONSTRUCTION_PATH, UNCHANGING_PATH,
        COMPENSATORY_PATH, PROTECTED_PATH, OUTPUT_PATH, DEBUG_PATH, INTERFERENCE_PATH, PARCEL_PATH]

# If the --new argument is provided, create the project directory and all subdirectories
if args.new:
    if args.new not in config.settings.projects:
        print(
            f"Error: Project '{args.new}' is not defined in the configuration settings.")
        sys.exit()

    # Create all directories
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

    # Create a sample config.yaml in the project directory
    sample_config_path = os.path.join(dir_path, 'config.yaml')
    config_dict = {
        "crs": "epsg:25833",
        "buffer_gen_distances": [100, 625],
        "buffer_distances": {
            "<100": "<100",
            ">100<625": ">100<625",
            ">625": ">625"
        },
        "grz_factors": {
            "0.5": [0.5, 0.2, 0.6],
            "0.75": [0.75, 0.5, 0.8]
        },
        "default_sliver": 0.001,
        "filter_small_areas": True,
        "filter_small_areas_limit": 1,
        "count_small_compensatory_if_adjacent": False,
        "projects": {
            args.new: {
                "path": dir_path
            }
        }
    }
    with open(sample_config_path, 'w') as config_file:
        yaml.dump(config_dict, config_file)

    print(f"New project '{args.new}' has been created with all necessary directories and a sample config.yaml. Please edit the project config, add shape files (some are optional) to the project directory in the 'scope', 'changing', 'construction', 'unchanging', 'compensatory', and 'protected' directories, and run the program again.")
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
dirs = [OUTPUT_PATH, DEBUG_PATH]

# Remove all files and subdirectories in each directory
for dir in dirs:
    shutil.rmtree(dir, ignore_errors=True)
    os.makedirs(dir, exist_ok=True)

# Global debug counter dictionary
debug_counter_dict = {}


# TODO: Debug does not seem to do anything
def debug(gdf, prefix='', show_plot_option=True, include_line_numbers=False):
    do_debug(args.debug, debug_counter_dict, DEBUG_PATH, gdf,
             prefix, show_plot_option, include_line_numbers)


interference = get_features(INTERFERENCE_PATH)
buffers = get_buffers(interference, config.settings.buffer_gen_distances)
scope = get_features(SCOPE_PATH)

print("Processing unchanged features...")
unchanging_features = get_features(UNCHANGING_PATH)
unchanging_features = filter_features(scope, unchanging_features)
print("Processing changing features...")
changing_features = get_features(CHANGING_PATH)
changing_features = filter_features(scope, changing_features)

print("Processing construction features...")
construction_features = process_features(
    CONSTRUCTION_PATH, 'construction', unchanging_features, changing_features, config.settings.projects[config.settings.project_name].changing_construction_base_values, scope)

print("Processing compensatory features...")
compensatory_features = process_features(
    COMPENSATORY_PATH, 'compensatory', unchanging_features, changing_features, config.settings.projects[config.settings.project_name].changing_compensatory_base_values, scope)

# debug(compensatory_features, 'compensatory')

print("Processing protected area features...")
protected_area_features = get_features(PROTECTED_PATH)
protected_area_features = filter_features(scope, protected_area_features)
protected_area_features = preprocess_features(
    protected_area_features, 'protected_area')

print("Processing geometric scope: Creating buffer zones for construction features...")
construction_features = process_and_separate_buffer_zones(
    scope, construction_features, buffers, protected_area_features)

print("Processing geometric scope: Creating buffer zones for compensatory features...")
compensatory_features = process_and_separate_buffer_zones(
    scope, compensatory_features, buffers, protected_area_features)

print("Processing geometric scope: Removing small areas from construction feature buffer zones...")
construction_features = remove_geometries_with_small_areas(
    construction_features, scope=scope)

print("Processing geometric scope: Removing small areas from compensatory features...")
compensatory_features = remove_geometries_with_small_areas(
    compensatory_features, scope=scope)

print("Adding compensatory values...")
if not compensatory_features.empty:
    compensatory_features = add_compensatory_value(
        compensatory_features, protected_area_features)
else:
    print("No compensatory features found.")

# ---> Construction Output Shapefile Creation <---

print()
print(config.settings.project_name)

print("Calculating construction score...")
construction_features = add_construction_score(
    construction_features, config.settings.projects[config.settings.project_name].grz)

total_construction_score = round(
    construction_features['score'].sum(), 2)
print(colored(
    f"Total Construction score: {total_construction_score}", 'yellow'))

print("Creating output shapefiles...")
for file in construction_features['name'].unique():
    current_features = construction_features[construction_features['name'] == file].copy(
    )
    check_and_warn_column_length(current_features)
    save_to_shapefile(
        current_features, 'Construction_' + file, OUTPUT_PATH, True)

print("Writing output JSON and Excel files...")
write_output_json_and_excel(total_construction_score, construction_features,
                            filename='Construction', output_dir=OUTPUT_PATH)

# ---> Compensatory Output Shapefile Creation <---
# Check if the compensatory_features DataFrame is not empty
if not compensatory_features.empty:
    print("Calculating compensatory score...")
    compensatory_features = add_compensatory_score(
        compensatory_features.copy(), scope)

    print("Creating output shapefiles...")
    total_compensatory_score = round(compensatory_features['score'].sum(), 2)
    print(colored(
        f"Total Compensatory score: {total_compensatory_score}", 'yellow'))

    print("Writing output JSON and Excel files...")
    for file in compensatory_features['name'].unique():
        current_features = compensatory_features[compensatory_features['name'] == file].copy(
        )
        check_and_warn_column_length(current_features)
        save_to_shapefile(
            current_features, 'Compensatory_' + file, OUTPUT_PATH, True)

    print("Writing output JSON and Excel files...")
    write_output_json_and_excel(total_compensatory_score, compensatory_features,
                                filename='Compensatory', output_dir=OUTPUT_PATH)
else:
    print(colored(
        "No compensatory features found. Skipping the rest of the Compensatory Output operations.", 'yellow'))

print("Creating plot...")
create_plot(construction_features, compensatory_features,
            interference, scope, OUTPUT_PATH, True)


def assign_sub_type(features, config):
    # Example of assigning sub-type based on configuration
    for feature in features.itertuples():
        land_use_type = getattr(feature, 'land_use_type', None)
        if land_use_type in config['changing_construction_base_values']:
            features.at[feature.Index, 'sub_type'] = 'changing_construction'
        elif land_use_type in config['changing_compensatory_base_values']:
            features.at[feature.Index, 'sub_type'] = 'changing_compensatory'
        elif land_use_type in config['compensatory_measure_values']:
            features.at[feature.Index, 'sub_type'] = 'compensatory_measure'
    return features


def process_features(file_path, feature_type, config):
    features = get_features(file_path)
    features['type'] = feature_type
    features = assign_sub_type(features, config)
    return features


def write_output_json_and_excel(total_score, features, filename, output_dir):
    import pandas as pd
    import json

    # Calculate area if not already present
    if 'area' not in features.columns:
        features['area'] = features.geometry.area

    # Prepare data for output
    output_data = features[['name', 'type', 'sub_type', 'area', 'score']]

    # Write to Excel
    excel_path = os.path.join(output_dir, f"{filename}.xlsx")
    output_data.to_excel(excel_path, index=False)

    # Write to JSON
    json_path = os.path.join(output_dir, f"{filename}.json")
    with open(json_path, 'w') as f:
        json.dump(output_data.to_dict(orient='records'), f)

    print(f"Output written to {excel_path} and {json_path}")


print("Processing parcel features...")
parcel_features = get_parcel_features(PARCEL_PATH)
parcel_features = filter_features(scope, parcel_features)

# At the beginning of your script, after loading parcel_features
if parcel_features.crs is None:
    parcel_features = parcel_features.set_crs(config.settings.crs)

# Ensure all feature sets have the same CRS
construction_features = construction_features.to_crs(parcel_features.crs)
compensatory_features = compensatory_features.to_crs(parcel_features.crs)


def calculate_overlap_area(feature1, feature2):
    overlap = gpd.overlay(feature1, feature2, how='intersection')
    overlap['overlap_area'] = overlap.geometry.area
    return overlap


def generate_parcel_report(parcel_features, construction_features, compensatory_features):
    construction_overlap = calculate_overlap_area(
        parcel_features, construction_features)
    compensatory_overlap = calculate_overlap_area(
        parcel_features, compensatory_features)

    report = []
    for label in parcel_features['label'].unique():
        parcel = parcel_features[parcel_features['label'] == label]
        construction_area = construction_overlap[construction_overlap['label'] == label]['overlap_area'].sum(
        )
        compensatory_area = compensatory_overlap[compensatory_overlap['label'] == label]['overlap_area'].sum(
        )

        report.append({
            'label': label,
            'total_area': parcel.geometry.area.sum(),
            'construction_area': construction_area,
            'compensatory_area': compensatory_area
        })

    return pd.DataFrame(report)


# Add this after processing construction and compensatory features
print("Generating parcel report...")
parcel_report = generate_parcel_report(
    parcel_features, construction_features, compensatory_features)


def add_parcel_report_to_excel(parcel_report, output_dir):
    for filename in ['Construction', 'Compensatory']:
        excel_path = os.path.join(
            output_dir, f"{config.settings.project_name}_{filename}.xlsx")
        if os.path.exists(excel_path):
            # Read the existing Excel file
            book = openpyxl.load_workbook(excel_path)

            # Filter and sort the parcel report
            area_column = f"{filename.lower()}_area"
            filtered_report = parcel_report[['label', area_column]]
            filtered_report = filtered_report[filtered_report[area_column] > 0]
            filtered_report = filtered_report.sort_values('label')

            # Create a new ExcelWriter object with the existing workbook
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
                # Write filtered parcel report to a new sheet
                filtered_report.to_excel(
                    writer, sheet_name='Parcel Report', index=False)

            print(f"Parcel report added to {excel_path}")
        else:
            print(
                f"Warning: {excel_path} not found. Skipping parcel report addition.")


# Add this at the end of main.py, after the parcel report generation
add_parcel_report_to_excel(parcel_report, OUTPUT_PATH)


def plot_parcels_with_features(parcel_features, construction_features, compensatory_features):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Plot parcels with construction features
    parcel_features.plot(ax=ax1, color='lightgrey', edgecolor='black')
    construction_features.plot(ax=ax1, color='red', alpha=0.5)
    ax1.set_title('Parcels with Construction Features')

    # Plot parcels with compensatory features
    parcel_features.plot(ax=ax2, color='lightgrey', edgecolor='black')
    compensatory_features.plot(ax=ax2, color='green', alpha=0.5)
    ax2.set_title('Parcels with Compensatory Features')

    for ax in (ax1, ax2):
        ax.set_axis_off()
        # Add labels to parcels
        for idx, row in parcel_features.iterrows():
            centroid = row.geometry.centroid
            ax.annotate(text=row['label'], xy=(centroid.x, centroid.y),
                        xytext=(3, 3), textcoords="offset points",
                        fontsize=8, color='black', ha='center', va='center')

    plt.tight_layout()
    plt.show()

# Debugging function to check for small overlaps


def check_overlaps(parcel_features, feature_gdf, feature_type):
    for idx, parcel in parcel_features.iterrows():
        overlap = gpd.overlay(gpd.GeoDataFrame([parcel], crs=parcel_features.crs),
                              feature_gdf, how='intersection')
        if not overlap.empty:
            overlap['area'] = overlap.geometry.area
            print(
                f"Overlap detected for parcel {parcel['label']} with {feature_type}:")
            print(overlap[['area']])
            print(f"Total overlap area: {overlap['area'].sum():.2f}")
            print("---")


# Run the plot function
plot_parcels_with_features(
    parcel_features, construction_features, compensatory_features)

# Check for small overlaps
print("Checking overlaps with construction features:")
check_overlaps(parcel_features, construction_features, "construction")

print("\nChecking overlaps with compensatory features:")
check_overlaps(parcel_features, compensatory_features, "compensatory")

# Print the total area for each parcel in the parcel report
print("\nParcel report areas:")
for _, row in parcel_report.iterrows():
    print(f"Parcel {row['label']}: Construction area = {row['construction_area']:.2f}, Compensatory area = {row['compensatory_area']:.2f}")
