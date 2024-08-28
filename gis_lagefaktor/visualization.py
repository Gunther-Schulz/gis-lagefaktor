import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from gis_lagefaktor.config import settings
import os
from gis_lagefaktor.debugging import pt
import random
import numpy as np
import itertools


def random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


def create_plot(construction_features, compensation_features, interference, scope, output_path, show_plot=False):
    """
    This function creates a plot with different layers of geospatial data.

    Parameters:
    construction_features (GeoDataFrame): The construction features to plot.
    compensation_features (GeoDataFrame): The compensation features to plot.
    interference (GeoDataFrame): The interference features to plot.
    scope (GeoDataFrame): The scope features to plot.
    show_plot (bool): Whether to display the plot. Defaults to False.
    """

    # Assuming 'features' is a GeoDataFrame
    # Increase the figure size (adjust these values as needed)
    # Increased from default size
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))

    column_name = ''
    handles = []
    labels = []
    plot_area = False

    # if 'buffer_dis' in construction_features.columns:

    import colorsys

    if not construction_features.empty and 'buffer_dis' in construction_features.columns and 'base_name' in construction_features.columns:
        # Add a group title for construction features (top-most group)
        handles.append(
            Patch(facecolor='none', edgecolor='none', label='Flächentypen'))
        labels.append('Flächentypen')

        column_name = 'buffer_dis'
        buffer_dis_values = construction_features[column_name].unique()

        # Define a base color for each unique value in 'buffer_dis'
        base_colors = sns.color_palette("hsv", len(buffer_dis_values))
        buffer_dis_to_color = {
            buffer_dis: base_colors[i] for i, buffer_dis in enumerate(buffer_dis_values)}

        # Plot 'buffer_dis' with the base color, using different shades for each 'base_name'
        for buffer_dis, group in construction_features.groupby(column_name):
            base_names = group['base_name'].unique()
            base_color = buffer_dis_to_color[buffer_dis]
            hue = colorsys.rgb_to_hsv(*base_color)[0]
            cmap = sns.cubehelix_palette(
                len(base_names), start=hue, dark=0.5, light=0.8)
            base_name_to_color = {base_name: cmap[i]
                                  for i, base_name in enumerate(base_names)}
            for base_name, subgroup in group.groupby('base_name'):
                subgroup.plot(
                    ax=ax, color=base_name_to_color[base_name], edgecolor='black', linewidth=0.5)
                handles.append(Patch(color=base_name_to_color[base_name]))
                labels.append(f"{buffer_dis} - {base_name}")

        # Add area labels if needed
        if plot_area:
            for x, y, label in zip(construction_features.geometry.centroid.x, construction_features.geometry.centroid.y, construction_features.geometry.area):
                ax.annotate(text=f'{int(label)}', xy=(x, y), fontsize=4)

    # Add single line spacing after Flächentypen group
    handles.append(Patch(facecolor='none', edgecolor='none', label=''))
    labels.append('')

    if not compensation_features.empty and 'compensat' in compensation_features.columns:
        # Add group title for compensation features
        handles.append(Patch(facecolor='none', edgecolor='none',
                       label='Kompensationswerte'))
        labels.append('Kompensationswerte')

        # Generate a unique color for each unique combination of compensat and buffer_dis
        unique_combinations = compensation_features.drop_duplicates(
            subset=['compensat', 'buffer_dis'])

        # Create a cyclic color palette
        color_palette = itertools.cycle(plt.cm.tab20.colors)

        # Create a dictionary to store colors for each unique combination
        color_dict = {}

        for _, row in unique_combinations.iterrows():
            key = (row['compensat'], row['buffer_dis'])
            if key not in color_dict:
                color_dict[key] = next(color_palette)

            color = color_dict[key]
            label = f"{row['compensat']} - Buffer: {row['buffer_dis']}"
            handles.append(Patch(color=color))
            labels.append(label)

        # Plot all compensation features
        for _, row in compensation_features.iterrows():
            key = (row['compensat'], row['buffer_dis'])
            color = color_dict[key]
            ax.fill(row.geometry.exterior.xy[0], row.geometry.exterior.xy[1],
                    color=color, edgecolor='black', linewidth=0.5)

    # Add single line spacing after Kompensationswerte group
    handles.append(Patch(facecolor='none', edgecolor='none', label=''))
    labels.append('')

    if not interference.empty:
        # Plot 'interference' on the same axes
        interference.plot(ax=ax, color='cyan')

        # Create a legend entry for 'interference'
        interference_patch = Patch(color='cyan', label='Störungsquelle')
        handles.append(interference_patch)
        labels.append('Störungsquelle')

    if not scope.empty:
        pt(scope)
        print("Scope geometry types:")
        print(scope.geometry.geom_type.value_counts())

        # Define a set of strong, distinct colors
        strong_colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#FFFF00', '#00FFFF',
                         '#FF8000', '#8000FF', '#0080FF', '#FF0080', '#80FF00', '#00FF80']

        # Plot all geometries at once
        for idx, row in scope.iterrows():
            color = strong_colors[idx % len(strong_colors)]
            ax.plot(*row.geometry.boundary.xy, color=color, linestyle='dashed',
                    linewidth=2, dashes=(5, 5))  # Thicker lines with adjusted dash pattern

            # Create a patch for the legend
            scope_patch = Patch(color=color, linestyle='dashed', fill=False)

            # Add to handles and labels
            handles.append(scope_patch)
            labels.append(
                row['name'] if 'name' in row else f'Geltungsbereich {idx+1}')

    # Create a single legend for all entries
    plt.legend(handles=handles, labels=labels,
               loc='upper left', bbox_to_anchor=(1, 1))

    plt.title(settings.project_name)

    # Adjust the layout to make room for the annotation
    plt.tight_layout()

    # Add centered coordinate system information
    crs = construction_features.crs
    if crs:
        fig.text(0.5, -0.02, f"Coordinate System: {crs.to_string()}",
                 fontsize=8, ha='center', va='top')

    # write plot to file
    plt.savefig(os.path.join(output_path, settings.project_name + '_plot.png'),
                dpi=600, bbox_inches='tight', pad_inches=0.1)
    if show_plot:
        plt.show()
