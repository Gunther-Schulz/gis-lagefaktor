import geopandas as gpd
import pandas as pd
import geopandas as gpd
import pandas as pd
from gis_lagefaktor.debugging import pt
from gis_lagefaktor.geospatial_ops import calculate_overlay, clean_geometries, filter_features, process_geodataframe_overlaps, merge_and_flatten_overlapping_geometries, remove_geometries_with_small_areas, calculate_intersection_area, remove_slivers, resolve_overlaps
from gis_lagefaktor.data_handling import get_value_with_warning, get_features, write_protocol
from gis_lagefaktor.lagefaktor import add_lagefaktor_values
from gis_lagefaktor.config import settings
import warnings
from shapely.validation import explain_validity


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
        lambda x: get_value_with_warning(settings.projects[
            settings.project_name].compensatory_measure_values, x))

    # Add 'eligible' column
    if settings.count_small_compensatory_if_adjacent == True:
        compensatory_features['eligible'] = compensatory_features.apply(
            lambda row: row['geometry'].area > get_value_with_warning(
                settings.projects[
                    settings.project_name].compensatory_measure_minimum_area, row['name']), axis=1)

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
        if settings.count_small_compensatory_if_adjacent == False:
            compensatory_features['eligible'] = compensatory_features.apply(
                lambda row: row['geometry'].area > get_value_with_warning(
                    settings.projects[
                        settings.project_name].compensatory_measure_minimum_area, row['name']), axis=1)

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

    # Check for empty or invalid geometries in input GeoDataFrames
    def check_geometries(gdf, name):
        if gdf.empty:
            print(f"Warning: {name} GeoDataFrame is empty.")
        elif gdf.geometry.is_empty.any() or gdf.geometry.is_valid.all() == False:
            print(f"Warning: {name} GeoDataFrame contains empty or invalid geometries.")
            print(f"Invalid geometries in {name}:", gdf[~gdf.geometry.is_valid])

    check_geometries(base_features, "base_features")
    check_geometries(unchanged_features, "unchanged_features")
    check_geometries(changing_features, "changing_features")

    # Rename 'name' column in changing_features
    changing_features = changing_features.rename(columns={'name': 'base_name'})

    # Punch holes
    print("Performing overlay operation: changing_features with unchanged_features")
    changing_features = gpd.overlay(changing_features, unchanged_features, how='difference')
    check_geometries(changing_features, "changing_features after overlay with unchanged_features")

    print(f"Base features count: {len(base_features)}")
    print(f"Base features types: {base_features['name'].unique()}")
    print(f"Changing features count: {len(changing_features)}")
    print(f"Changing features types: {changing_features['base_name'].unique()}")

    # Overlay base_features with changing_features
    print("Performing overlay operation: base_features with changing_features")
    try:
        # Check for valid geometries
        base_features = base_features[base_features.geometry.is_valid]
        changing_features = changing_features[changing_features.geometry.is_valid]

        # Attempt to fix any remaining invalid geometries
        base_features['geometry'] = base_features.geometry.buffer(0)
        changing_features['geometry'] = changing_features.geometry.buffer(0)

        # Remove None geometries
        base_features = base_features[base_features.geometry.notna()]
        changing_features = changing_features[changing_features.geometry.notna()]

        print("Base features geometry types:")
        print(base_features.geometry.type.value_counts())
        print("Changing features geometry types:")
        print(changing_features.geometry.type.value_counts())

        print("Sample of base features:")
        print(base_features[['name', 'geometry']].head())
        print("Sample of changing features:")
        print(changing_features[['base_name', 'geometry']].head())

        print("Base features total area:", base_features.geometry.area.sum())
        print("Changing features total area:", changing_features.geometry.area.sum())

        print("Checking geometry validity before overlay:")
        print("Base features invalid geometries:")
        for idx, geom in base_features.geometry.items():
            if not geom.is_valid:
                print(f"Index {idx}: {explain_validity(geom)}")
        
        print("Changing features invalid geometries:")
        for idx, geom in changing_features.geometry.items():
            if not geom.is_valid:
                print(f"Index {idx}: {explain_validity(geom)}")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            intersected_features = gpd.overlay(base_features, changing_features, how='intersection')
            for warning in w:
                print(f"Warning during overlay: {warning.message}")

        print("Checking geometry types after overlay:")
        print(intersected_features.geometry.type.value_counts())

        print("Intersected features total area:", intersected_features.geometry.area.sum())
        print("Intersected features count:", len(intersected_features))

        if intersected_features.empty:
            print("Warning: The intersection result is empty.")
            return gpd.GeoDataFrame(columns=base_features.columns, crs=base_features.crs)
        else:
            print("Intersected features types:")
            print(intersected_features.geometry.type.value_counts())

        check_geometries(intersected_features, "intersected_features")

        intersected_features['base_value'] = intersected_features['base_name'].map(
            lambda x: get_value_with_warning(values, x))

        return intersected_features

    except Exception as e:
        print(f"Error during overlay operation: {str(e)}")
        print("Base features:")
        print(base_features[['name', 'geometry']].head())
        print("\nChanging features:")
        print(changing_features[['base_name', 'geometry']].head())
        
        # Additional debugging information
        print("\nBase features with invalid geometries:")
        print(base_features[~base_features.geometry.is_valid])
        print("\nChanging features with invalid geometries:")
        print(changing_features[~changing_features.geometry.is_valid])
        
        return gpd.GeoDataFrame(columns=base_features.columns, crs=base_features.crs)


def create_feature_identifier(feature):
    """
    Creates a meaningful identifier for a feature based on its characteristics.
    """
    # Get centroid coordinates for spatial reference
    centroid = feature.geometry.centroid
    x, y = round(centroid.x), round(centroid.y)
    
    # Create identifier using feature type, buffer distance, and location
    area = feature.geometry.area
    feature_type = feature['name']
    buffer_dist = feature.get('buffer_dis', 'no_buffer')
    
    # Match the legend categories
    if feature_type == "Acker" and buffer_dist == "<100":
        category = "Acker - <100"
    elif feature_type == "Acker" and buffer_dist == ">100<625":
        category = "Acker - >100<625"
    elif feature_type == "Grünland" and buffer_dist == "<100":
        category = "Grünland - <100"
    elif feature_type == "Grünland" and buffer_dist == ">100<625":
        category = "Grünland - >100<625"
    else:
        category = f"{feature_type} - {buffer_dist}"
    
    identifier = f"{category}_{area:.0f}m²_at_{x}_{y}"
    return identifier


def calculate_compensatory_score(row, current_features, output_dir, project_name):
    """
    Calculate compensatory score with protocol logging.
    """
    if row['eligible'] == True:
        area = row.geometry.area
        compensat = row['compensat']
        base_value = row['base_value']
        lagefaktor = row['lagefaktor']
        
        # Create spatial identifier
        feature_id = create_feature_identifier(row)
        
        # Step-by-step calculation
        initial_value = compensat - base_value
        adjusted_value = initial_value * area
        final_v = adjusted_value * lagefaktor
        
        if 'prot_comp' in current_features.columns and pd.notnull(row['prot_comp']):
            prot_value = get_value_with_warning(
                settings.projects[settings.project_name].compensatory_protected_values, 
                row['prot_name']
            )
        else:
            prot_value = 1
            
        final_v *= prot_value
        
        # Detailed protocol message
        protocol_message = (
            f"Compensatory Score Calculation:\n"
            f"  Feature ID: {feature_id}\n"
            f"  Feature Type: {row['name']}\n"
            f"  Area: {area:.2f}\n"
            f"  Initial Value (Compensatory - Base): {initial_value}\n"
            f"  Adjusted Value (Initial * Area): {adjusted_value:.2f}\n"
            f"  Final Value (Adjusted * Lagefaktor): {final_v:.2f}\n"
            f"  Protection Value: {prot_value}\n"
            f"  Final Score: {final_v:.2f}\n"
            f"-------------------"
        )
        write_protocol(protocol_message, output_dir, project_name)
        
        return round(final_v, 2)
    else:
        feature_id = create_feature_identifier(row)
        write_protocol(
            f"Feature marked as not eligible - Feature ID: {feature_id} - Feature Type: {row['name']} - Score: 0\n-------------------", 
            output_dir, 
            project_name
        )
        return 0


def add_compensatory_score(features, scope, output_dir, project_name):
    """
    This function adds compensatory scores to a GeoDataFrame of features.
    """
    pt(features)
    all_features = []
    for file in features['name'].unique():
        current_features = features[features['name'] == file].copy()
        current_features['score'] = current_features.apply(
            lambda row: round(calculate_compensatory_score(row, current_features, output_dir, project_name), 2), axis=1)
        all_features.append(current_features)

    return pd.concat(all_features, ignore_index=True)


def add_construction_score(features, grz, output_dir, project_name):
    """
    Calculate the total final value based on features and GRZ factors.
    Now includes enhanced protocol writing for intermediate calculations.
    """
    scores = []
    for _, feature in features.iterrows():
        area = feature.geometry.area
        base_value = feature['base_value']
        lagefaktor = feature['lagefaktor']
        
        # Create spatial identifier
        feature_id = create_feature_identifier(feature)
        
        # Step-by-step calculation
        initial_value = base_value * lagefaktor * area
        factor_a, factor_b, factor_c = settings.grz_factors[grz]
        adjusted_value = initial_value * factor_a
        final_value = adjusted_value * (factor_b + factor_c)
        score = round(final_value, 2)
        scores.append(score)
        
        # Detailed protocol message
        protocol_message = (
            f"Construction Score Calculation:\n"
            f"  Feature ID: {feature_id}\n"
            f"  Feature Type: {feature['name']}\n"
            f"  Area: {area:.2f}\n"
            f"  Initial Value (Base * Lagefaktor * Area): {initial_value:.2f}\n"
            f"  Adjusted Value (Initial * Factor A): {adjusted_value:.2f}\n"
            f"  Final Value (Adjusted * (Factor B + Factor C)): {final_value:.2f}\n"
            f"  GRZ Factors (a,b,c): {factor_a}, {factor_b}, {factor_c}\n"
            f"  Final Score: {score}\n"
            f"-------------------"
        )
        write_protocol(protocol_message, output_dir, project_name)

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
    scope.crs = settings.crs

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
            lambda x: get_value_with_warning(settings.projects[
                settings.project_name].compensatory_measure_values, x))
    elif feature_type == 'protected_area':
        # Set 'prot_cons' and 'prot_comp' based on 'name'
        processed_features['prot_cons'] = processed_features['name'].apply(
            lambda x: get_value_with_warning(settings.projects[
                settings.project_name].construction_protected_values, x))
        processed_features['prot_comp'] = processed_features['name'].apply(
            lambda x: get_value_with_warning(settings.projects[
                settings.project_name].compensatory_protected_values, x))
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
    scope (GeoDataFrame): The geometric scope to clip features to.

    Returns:
    GeoDataFrame: The processed features.
    """
    features = get_features(directory)
    features = gpd.clip(features, scope)

    features = preprocess_features(features, feature_type)

    # Ensure features contain only Polygon geometries
    features = features[features.geometry.type.isin(
        ['Polygon', 'MultiPolygon'])]
    features = features.explode(index_parts=False)
    features = features[features.geometry.type == 'Polygon']

    features = process_and_overlay_features(
        features, unchanged_features, changing_features, changing_values)

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
            construction_feature, buffers[0], settings.buffer_distances['<100'], protected_area_features, scope)
        features = pd.concat(
            [features, changing_feature_B1_intersection], ignore_index=True)

    # Check if there is a '>100<625' buffer
    if len(buffers) > 1 and not buffers[1].empty:
        changing_feature_B2_intersection = calculate_intersection_area(
            construction_feature, buffers[1], settings.buffer_distances['>100<625'], protected_area_features, scope)
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
        changing_feature_outside_B2, settings.projects[
            settings.project_name].construction_lagefaktor_values[settings.buffer_distances['>625']])
    # changing_feature_outside_B2 = filter_features(
    #     scope, changing_feature_outside_B2)
    changing_feature_outside_B2['buffer_dis'] = settings.buffer_distances['>625']

    features = pd.concat(
        [features, changing_feature_outside_B2], ignore_index=True)

    return features
