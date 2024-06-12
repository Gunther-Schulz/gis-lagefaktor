import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from gis_lagefaktor.config import settings
import os


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
    fig, ax = plt.subplots(1, 1)

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
        # if 'compensat' in compensation_features.columns:
        column_name = 'compensat'
        # Define the color map and norm
        colors = ['red', 'green', 'blue']  # replace with the colors you want
        bounds = [1, 2, 3, 4]  # replace with the boundaries you want
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bounds, cmap.N)

        # Plot the GeoDataFrame without the legend
        compensation_features.plot(
            column=column_name, ax=ax, cmap=cmap, norm=norm, edgecolor='black', linewidth=0.5)

        # Add area labels
        if plot_area:
            for x, y, label in zip(construction_features.geometry.centroid.x, construction_features.geometry.centroid.y, construction_features.geometry.area):
                ax.annotate(text=f'{int(label)}', xy=(x, y), fontsize=4)
        # Get the unique values in the column
        unique_values = compensation_features[column_name].unique()

        # Create a legend entry for each unique value
        compensat_patches = [Patch(color=cmap(
            norm(value)), label=value) for value in unique_values]

        handles.append(Patch(facecolor='none', edgecolor='none',
                       label='Kompensationswert'))
        handles.extend(compensat_patches)
        labels.append('Kompensationswert')
        labels.extend(unique_values)

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
        scope.boundary.plot(ax=ax, color='black', linestyle='dashed')

        scope_patch = Patch(
            color='black', label='Geltungsbereich', linestyle='dashed', fill=False)

        handles.append(
            Patch(facecolor='none', edgecolor='none', label='Geltungsbereich'))
        handles.append(scope_patch)
        labels.append('')
        labels.append('Geltungsbereich')

    # Create a single legend for all entries
    plt.legend(handles=handles, labels=labels,
               loc='upper left', bbox_to_anchor=(1, 1))

    plt.title(settings.project_name)
    # write plot to file
    plt.savefig(os.path.join(output_path, settings.project_name +
                '_plot.png'), dpi=600, bbox_inches='tight')
    if show_plot:
        plt.show()
