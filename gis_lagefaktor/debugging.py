import inspect
from termcolor import colored
import warnings
import inspect
import os
import sys
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd


def get_calling_function_name():
    """
    This function returns the name of the function in the main module from where this function was called.

    Returns:
    str: The name of the function in the main module from where this function was called, or None if the function was not called from the main module.
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
    This function returns the line number in the main module from where this function was called.

    Returns:
    int: The line number in the main module from where this function was called, or None if the function was not called from the main module.
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


def show_plot(gdf, title):
    """
    This function shows a plot of a GeoDataFrame.

    Parameters:
    gdf (GeoDataFrame): The GeoDataFrame to plot.
    title (str): The title of the plot.

    Returns:
    None
    """
    fig, ax = plt.subplots()
    gdf.plot(ax=ax)
    ax.set_title(title)  # Add this line to set the title
    plt.show()


def do_debug(debug, debug_counter_dict, DEBUG_DIR, gdf, prefix='', show_plot_option=False, include_line_numbers=False):
    """
    This function writes a GeoDataFrame to a shapefile for debugging purposes.

    Parameters:
    gdf (GeoDataFrame): The GeoDataFrame to write.
    prefix (str, optional): An optional prefix to add to the filename.
    include_line_numbers (bool, optional): Whether to include line numbers in the filename.

    Returns:
    None
    """
    if debug:
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
        if show_plot_option:
            show_plot(gdf, prefix)


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
        if not np.isnan(max_length):
            print(colors[i % len(colors)] +
                  name.ljust(int(max_length) + 1, ' '), end='')
        else:
            print("Problem: max_length is NaN")
            # Handle the case when max_length is NaN
    print('\033[0m')  # Reset color

    # Print a line of dashes
    if pd.isnull(max_length):
        print("max_length is NaN")
    else:
        print('-' * int((max_length + 1) * len(fixed_width_df.columns)))

    # Print each row with fixed width columns
    for index, row in fixed_width_df.iterrows():
        for i, value in enumerate(row):
            print(colors[i % len(colors)] + value, end=' ')
        print('\033[0m')  # Reset color after each row

    print()


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
