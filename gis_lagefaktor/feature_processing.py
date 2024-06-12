import geopandas as gpd
import pandas as pd
import geopandas as gpd
import pandas as pd
from gis_lagefaktor.debugging import pt
from gis_lagefaktor.geospatial_ops import calculate_overlay, clean_geometries, filter_features, process_geodataframe_overlaps, merge_and_flatten_overlapping_geometries, remove_geometries_with_small_areas, calculate_intersection_area, remove_slivers, resolve_overlaps
from gis_lagefaktor.data_handling import get_value_with_warning, get_features
from gis_lagefaktor.lagefaktor import add_lagefaktor_values
from gis_lagefaktor.config import settings

CRS = settings.crs
GRZ_FACTORS = settings.grz_factors
BUFFER_DISTANCES = settings.buffer_distances
CONSTRUCTION_LAGEFAKTOR_VALUES = settings.projects[
    settings.project_name].construction_lagefaktor_values
CONSTRUCTION_PROTECTED_VALUES = settings.projects[
    settings.project_name].construction_protected_values
COMPENSATORY_MEASURE_VALUES = settings.projects[
    settings.project_name].compensatory_measure_values
COMPENSATORY_MEASURE_MINIMUM_AREAS = settings.projects[
    settings.project_name].compensatory_measure_minimum_area
COMPENSATORY_PROTECTED_VALUES = settings.projects[
    settings.project_name].compensatory_protected_values
COUNT_SAMLL_COMPENSATORY_IF_ADJECENT = settings.count_small_compensatory_if_adjacent


# def add_lagefaktor_values(feature, lagefaktor_value):
#     """
#     This function adds 'lagefaktor' values to the given feature GeoDataFrame.

#     Parameters:
#     feature (GeoDataFrame): The GeoDataFrame to which to add 'lagefaktor' values.
#     lagefaktor_value (float): The 'lagefaktor' value to add.

#     Returns:
#     GeoDataFrame: The updated GeoDataFrame with 'lagefaktor' values.
#     """

#     if 'prot_cons' in feature.columns:
#         # Check if 'prot_cons' is not null
#         is_protected_not_null = feature['prot_cons'].notnull()

#         feature['lagefaktor'] = feature['prot_cons'].fillna(lagefaktor_value)
#         if lagefaktor_value == CONSTRUCTION_LAGEFAKTOR_VALUES.get('<100'):
#             # Only subtract 0.25 from 'lagefaktor' if 'prot_cons' is not null
#             feature.loc[is_protected_not_null, 'lagefaktor'] -= 0.25
#     else:
#         feature['lagefaktor'] = lagefaktor_value

#     # remove column prot_comp if it exists
#     if 'prot_comp' in feature.columns:
#         feature = feature.drop(columns='prot_comp')

#     return feature


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
    if COUNT_SAMLL_COMPENSATORY_IF_ADJECENT == True:
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
        compensatory_features = merge_and_flatten_overlapping_geometries(
            compensatory_features)

    if not compensatory_features.empty:
        if COUNT_SAMLL_COMPENSATORY_IF_ADJECENT == False:
            compensatory_features['eligible'] = compensatory_features.apply(
                lambda row: row['geometry'].area > get_value_with_warning(
                    COMPENSATORY_MEASURE_MINIMUM_AREAS, row['name']), axis=1)

    return compensatory_features


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
    # intersected_features = intersected_features[base_features.columns]

    # intersected_features['base_name'] = changing_features['base_name']

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
    # # Cleanup and merge features
    # processed_features = cleanup_and_merge_features(
    #     features, buffer_distance=buffer_distance)

    features = clean_geometries(features)
    processed_features = merge_and_flatten_overlapping_geometries(
        features)

    # processed_features = features

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


def process_features(directory, feature_type, unchanged_features, changing_features, changing_values, scope):
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
    features = filter_features(scope, features)

    features = preprocess_features(features, feature_type)
    features = process_and_overlay_features(
        features, unchanged_features, changing_features, changing_values)

    # TODO: Can I merge function below somehow with function clean_and_merge_features?
    features = merge_and_flatten_overlapping_geometries(features)

    return features


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

    # Initialize an empty DataFrame to store the features
    features = pd.DataFrame()

    # Check if there is a '<100' buffer
    if not buffers[0].empty:
        changing_feature_B1_intersection = calculate_intersection_area(
            construction_feature, buffers[0], BUFFER_DISTANCES['<100'], protected_area_features, scope)
        features = pd.concat(
            [features, changing_feature_B1_intersection], ignore_index=True)

    # Check if there is a '>100<625' buffer
    if len(buffers) > 1 and not buffers[1].empty:
        changing_feature_B2_intersection = calculate_intersection_area(
            construction_feature, buffers[1], BUFFER_DISTANCES['>100<625'], protected_area_features, scope)
        # Subtract changing_feature_B1_intersection from changing_feature_B2_intersection
        if not features.empty:
            changing_feature_B2_not_B1 = calculate_overlay(
                changing_feature_B2_intersection, features, 'difference')
            features = pd.concat(
                [features, changing_feature_B2_not_B1], ignore_index=True)
        # debug(construction_feature, 'construction_feature', show_plot_option=True)

    # Calculate area outside B2 by taking the difference between the construction feature and buffer B2
    if len(buffers) > 1 and not buffers[1].empty:
        changing_feature_outside_B2 = calculate_overlay(
            construction_feature, buffers[1], 'difference')
    else:
        changing_feature_outside_B2 = construction_feature.copy()
    changing_feature_outside_B2 = process_geodataframe_overlaps(
        changing_feature_outside_B2, protected_area_features)
    changing_feature_outside_B2 = add_lagefaktor_values(
        changing_feature_outside_B2, CONSTRUCTION_LAGEFAKTOR_VALUES[BUFFER_DISTANCES['>625']])
    # changing_feature_outside_B2 = filter_features(
    #     scope, changing_feature_outside_B2)
    changing_feature_outside_B2['buffer_dis'] = BUFFER_DISTANCES['>625']

    features = pd.concat(
        [features, changing_feature_outside_B2], ignore_index=True)

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
        final_v = (row['compensat'] - row['base_value']) * \
            row.geometry.area * row['lagefaktor']
        if 'prot_comp' in current_features.columns and pd.notnull(row['prot_comp']):
            prot_value = get_value_with_warning(
                COMPENSATORY_PROTECTED_VALUES, row['prot_name'])
        else:
            prot_value = 1

        final_v = final_v * prot_value
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

    pt(features)
    all_features = pd.DataFrame()
    for file in features['name'].unique():
        current_features = features[features['name'] == file]
        current_features['score'] = round(current_features.apply(
            lambda row: calculate_compensatory_score(row, current_features), axis=1), 2)
        all_features = pd.concat([all_features, current_features])
    return all_features
