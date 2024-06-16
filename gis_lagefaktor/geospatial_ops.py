import geopandas as gpd
import pandas as pd
import numpy as np
import re
from termcolor import colored
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from gis_lagefaktor.lagefaktor import add_lagefaktor_values
from gis_lagefaktor.debugging import pt
from gis_lagefaktor.config import settings


def create_buffer(linestrings, distance, resolution=64):
    """
    This function creates a buffer around each linestring and dissolves all geometries into a single one.

    Parameters:
    linestrings (GeoSeries): The linestrings around which to create buffers.
    distance (float): The distance for the buffer.
    resolution (int): The number of segments used to approximate a quarter circle around each vertex in the buffer operation.

    Returns:
    GeoDataFrame: A GeoDataFrame containing the buffers.
    """
    # Create a buffer around each linestring with specified resolution and dissolve all geometries into a single one
    buffers = linestrings.buffer(distance, resolution=resolution).to_frame().rename(
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
    resolved.crs = settings.crs

    return resolved


def remove_slivers(gdf, buffer_distance=None, resolution=64):
    """
    Removes slivers from geometries by applying a small buffer.

    Parameters:
    - gdf: GeoDataFrame to be processed.
    - buffer_distance: Distance for buffering operations.
    - resolution: The number of segments used to approximate a quarter circle around each vertex.

    Returns:
    - GeoDataFrame with slivers removed.
    """

    if buffer_distance is None:
        buffer_distance = settings.default_sliver

    gdf.geometry = gdf.geometry.buffer(
        buffer_distance, resolution=resolution).buffer(-buffer_distance, resolution=resolution)
    gdf.crs = settings.crs
    return gdf


def clean_geometries(gdf, resolution=16):
    """
    This function cleans invalid geometries in a GeoDataFrame and plots the invalid and cleaned geometries.

    Parameters:
    gdf (GeoDataFrame): The GeoDataFrame to clean.
    resolution (int): The number of segments used to approximate a quarter circle around each vertex in the buffer operation.

    Returns:
    GeoDataFrame: The cleaned GeoDataFrame.
    """
    invalid_geometries = gdf[~gdf.geometry.is_valid]
    if not invalid_geometries.empty:
        print(colored('Warning: Invalid geometries found. Cleaning...', 'red'))

        # Create a figure with 2 subplots
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Plot invalid geometries
        invalid_geometries.plot(ax=axs[0], color='red')
        axs[0].set_title('Invalid Geometries')

        # Clean geometries by applying a zero-width buffer with increased resolution
        gdf['geometry'] = gdf.geometry.buffer(0, resolution=resolution)

        # Plot valid geometries
        gdf.plot(ax=axs[1], color='green')

        # Highlight previously invalid geometries
        previously_invalid = gdf.loc[invalid_geometries.index.intersection(
            gdf.index)]
        previously_invalid.plot(ax=axs[1], color='red')

        axs[1].set_title('Cleaned Geometries (Previously Invalid in Red)')

        # Show the plots
        plt.tight_layout()
        plt.show()

    return gdf


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


def merge_and_flatten_overlapping_geometries(gdf):
    """
    This function merges a GeoDataFrame by all columns except 'geometry' and
    flattens overlapping geometries into a single geometry.

    Parameters:
    gdf (GeoDataFrame): The GeoDataFrame to merge.

    Returns:
    GeoDataFrame: The merged GeoDataFrame with flattened geometries.
    """

    # Fill NaN values with a common value
    gdf = gdf.fillna("missing")

    # Exclude 'geometry' column for the dissolve operation
    columns_to_dissolve_by = [col for col in gdf.columns if col != 'geometry']

    # Dissolve the GeoDataFrame by all columns except 'geometry'
    gdf = gdf.dissolve(by=columns_to_dissolve_by)

    # Convert MultiPolygons to individual Polygons
    # TODO: Should be keep index_parts=True ?
    gdf = gdf.geometry.explode(index_parts=True)

    # Create a new GeoDataFrame, keeping the original column values
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry')
    gdf[columns_to_dissolve_by] = gdf.index.to_frame()[columns_to_dissolve_by]

    # Reset the index
    gdf = gdf.reset_index(drop=True)

    # Replace 'missing' values with NaN
    gdf = gdf.replace("missing", np.nan)

    return gdf


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
        intersection, settings.projects[settings.project_name].construction_lagefaktor_values[buffer_distance])
    # intersection = filter_features(scope, intersection)
    intersection['buffer_dis'] = buffer_distance
    return intersection


def remove_geometries_with_small_areas(gdf, area_limit=None):
    """
    This function checks for geometries with zero area in a GeoDataFrame.

    Parameters:
    gdf (GeoDataFrame): The GeoDataFrame to check.

    Returns:
    GeoDataFrame: The GeoDataFrame with geometries with zero area removed.
    """
    if area_limit is None:
        area_limit = settings.filter_small_areas_limit

    zero_area = gdf[gdf.geometry.area <= area_limit]
    if not zero_area.empty:
        areas = zero_area.geometry.area.tolist()
        print(colored(
            f'Warning: Geometries with small area found: {areas}. Removing...', 'red'))
        pt(zero_area, 'Zero Area Geometries')
        # plot zero_area
        zero_area.plot()
        plt.show()
        gdf = gdf[gdf.geometry.area > area_limit]
    return gdf
