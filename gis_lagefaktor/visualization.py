import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from gis_lagefaktor.config import settings
import os
from gis_lagefaktor.debugging import pt
import random


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
        column_name = 'buffer_dis'
        buffer_dis_values = construction_features[column_name].unique()

        # Define a base color for each unique value in 'buffer_dis'
        base_colors = sns.color_palette("hsv", len(buffer_dis_values))
        buffer_dis_to_color = {
            buffer_dis: base_colors[i] for i, buffer_dis in enumerate(buffer_dis_values)}

        # Plot 'buffer_dis' with the base color, using different shades for each 'base_name'
        handles = []
        labels = []
        for buffer_dis, group in construction_features.groupby(column_name):
            base_names = group['base_name'].unique()
            # Create a colormap for each 'base_name' within the same 'buffer_dis'
            base_color = buffer_dis_to_color[buffer_dis]
            hue = colorsys.rgb_to_hsv(*base_color)[0]  # Get the hue component
            cmap = sns.cubehelix_palette(
                len(base_names), start=hue, dark=0.5, light=0.8)
            base_name_to_color = {base_name: cmap[i]
                                  for i, base_name in enumerate(base_names)}
            for base_name, subgroup in group.groupby('base_name'):
                subgroup.plot(
                    ax=ax, color=base_name_to_color[base_name], edgecolor='black', linewidth=0.5)
                # Add a legend entry for each unique 'base_name'
                handles.append(Patch(color=base_name_to_color[base_name]))
                labels.append(f"{buffer_dis} - {base_name}")

        # Add area labels
        if plot_area:
            for x, y, label in zip(construction_features.geometry.centroid.x, construction_features.geometry.centroid.y, construction_features.geometry.area):
                ax.annotate(text=f'{int(label)}', xy=(x, y), fontsize=4)

        ax.legend(handles=handles, labels=labels, loc='best')

    if not compensation_features.empty and 'compensat' in compensation_features.columns:
        column_name = 'compensat'

        # Generate a unique color for each polygon
        num_polygons = len(compensation_features)
        color_palette = sns.color_palette("husl", num_polygons)

        # Plot each polygon with its unique color
        for idx, row in compensation_features.iterrows():
            color = color_palette[idx]
            ax.fill(row.geometry.exterior.xy[0], row.geometry.exterior.xy[1],
                    color=color, edgecolor='black', linewidth=0.5)

        # Create legend entries
        compensat_patches = [Patch(color=color_palette[idx],
                                   label=f"{row[column_name]}")
                             for idx, row in compensation_features.iterrows()]

        handles.append(Patch(facecolor='none', edgecolor='none',
                       label='Kompensationswerte'))
        handles.extend(compensat_patches)
        labels.append('Kompensationswerte')
        labels.extend([patch.get_label() for patch in compensat_patches])

        # Add area labels if needed
        if plot_area:
            for x, y, label in zip(compensation_features.geometry.centroid.x, compensation_features.geometry.centroid.y, compensation_features.geometry.area):
                ax.annotate(text=f'{int(label)}', xy=(x, y), fontsize=4)

    if not interference.empty:
        # Plot 'interference' on the same axes
        interference.plot(ax=ax, color='cyan')

        # Create a legend entry for 'interference'
        interference_patch = Patch(color='cyan', label='Störungsquelle')

        handles.append(
            Patch(facecolor='none', edgecolor='none', label='Störungsquelle'))
        handles.append(interference_patch)
        labels.append('')
        labels.append('Störungsquelle')

    if not scope.empty:
        pt(scope)
        print("Scope geometry types:")
        print(scope.geometry.geom_type.value_counts())

        # Plot all geometries at once
        for idx, row in scope.iterrows():
            color = random_color()
            ax.plot(*row.geometry.boundary.xy, color=color, linestyle='dashed')

            # Create a patch for the legend
            scope_patch = Patch(color=color, linestyle='dashed', fill=False)

            # Add to handles and labels
            handles.append(scope_patch)
            labels.append(
                row['name'] if 'name' in row else f'Geltungsbereich {idx+1}')

        handles.append(
            Patch(facecolor='none', edgecolor='none', label='Geltungsbereich'))
        labels.append('')

    # Create a single legend for all entries
    plt.legend(handles=handles, labels=labels,
               loc='upper left', bbox_to_anchor=(1, 1))

    plt.title(settings.project_name)
    # write plot to file
    plt.savefig(os.path.join(output_path, settings.project_name + '_plot.png'),
                dpi=600, bbox_inches='tight')
    if show_plot:
        plt.show()
