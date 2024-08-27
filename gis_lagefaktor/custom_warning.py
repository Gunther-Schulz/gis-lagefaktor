import re
from termcolor import colored
import traceback


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

    if match_no_buffer:
        # Silently ignore this warning
        pass
    elif not match_keepdims:
        # Get the full traceback
        # Remove the last entry (this function call)
        tb = traceback.extract_stack()[:-1]
        tb_str = ''.join(traceback.format_list(tb))

        print(colored('Warning:', 'red'))
        print(tb_str.strip())
        print(colored(str(message), 'yellow'))
        print()  # Add a blank line for better readability
