import re
from termcolor import colored
from gis_lagefaktor.debugging import get_calling_function_name, get_calling_line_number


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
