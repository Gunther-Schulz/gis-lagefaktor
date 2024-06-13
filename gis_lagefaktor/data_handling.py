import os
import glob
import geopandas as gpd
import pandas as pd
from termcolor import colored
from colorama import Fore
import unicodedata
import os
import geopandas as gpd
import pandas as pd
import simplejson as sjson
import warnings

from gis_lagefaktor.config import settings


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
    encoded_name = normalize_string(s_name)
    print(colored(
        f"  {encoded_name}/{os.path.basename(file_path)}", 'yellow', attrs=['dark']))
    feature = gpd.read_file(file_path)
    feature = feature.to_crs(settings.crs)
    feature = feature[['geometry']]
    feature['name'] = encoded_name
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
            colored(f"No shapefiles found in directory {dir}. Make sure shapefiles are under a subdirectory with the name of the type. For example 'ProjectName/construction/" + Fore.RED + "Baufeld" + Fore.RESET + "/*.shp", 'yellow'))
        gdf = gpd.GeoDataFrame(columns=['geometry', 'name'], crs=settings.crs)
        return gdf
    else:
        print(colored(f"Found {len(shapefiles)} shapefiles:", 'green'))
    for shapefile in shapefiles:
        feature = read_shapefile(shapefile)
        for _, row in feature.iterrows():
            print(f"File: {shapefile}, Area: {row['geometry'].area}")

    features = [read_shapefile(shapefile)
                for shapefile in shapefiles]
    gdf = pd.concat(features, ignore_index=True)
    gdf = gdf.to_crs(settings.crs)

    return gdf


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


def save_to_shapefile(features, filename, output_dir):
    """
    This function saves a GeoDataFrame to a shapefile.

    Parameters:
    features (GeoDataFrame): The GeoDataFrame to save.
    filename (str): The name of the shapefile.

    The shapefile is saved in the OUTPUT_DIR directory.
    """
    print(colored(
        f"Saving {filename} to shapefile {output_dir}/{filename}.shp", 'yellow', attrs=['dark']))
    features.to_file(os.path.join(output_dir, filename),
                     driver='ESRI Shapefile')


def write_output_json_and_excel(total_score, data, filename='output', output_dir="output"):
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

    with open(os.path.join(output_dir, settings.project_name + '_' + filename + '.json'), 'w') as file:
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

    df.to_excel(os.path.join(output_dir, settings.project_name + '_' +
                filename + '.xlsx'), index=False)
