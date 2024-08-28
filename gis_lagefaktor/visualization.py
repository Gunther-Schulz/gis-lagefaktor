import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from gis_lagefaktor.config import settings
import os
import numpy as np


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

    handles = []
    labels = []

    # Create color palettes
    construction_palette = {
        '<100': plt.cm.get_cmap('Blues')(np.linspace(0.3, 0.8, 2)),
        '>100<625': plt.cm.get_cmap('Oranges')(np.linspace(0.3, 0.8, 2)),
        '>625': plt.cm.get_cmap('Greens')(np.linspace(0.3, 0.8, 2))
    }
    compensation_palette = plt.cm.get_cmap('Greens')(np.linspace(0.3, 0.8, 3))

    if not construction_features.empty and 'buffer_dis' in construction_features.columns and 'base_name' in construction_features.columns:
        handles.append(
            Patch(facecolor='none', edgecolor='none', label='Flächentypen'))
        labels.append('Flächentypen')

        for buffer_dis, group in construction_features.groupby('buffer_dis'):
            colors = construction_palette[buffer_dis]
            base_names = group['base_name'].unique()
            for i, base_name in enumerate(base_names):
                subgroup = group[group['base_name'] == base_name]
                color = colors[i % len(colors)]
                subgroup.plot(ax=ax, color=color,
                              edgecolor='black', linewidth=0.5)
                handles.append(Patch(color=color))
                labels.append(f"{buffer_dis} - {base_name}")

    handles.append(Patch(facecolor='none', edgecolor='none', label=''))
    labels.append('')

    if not compensation_features.empty and 'compensat' in compensation_features.columns:
        handles.append(Patch(facecolor='none', edgecolor='none',
                       label='Kompensationswerte'))
        labels.append('Kompensationswerte')

        for i, (compensat, group) in enumerate(compensation_features.groupby('compensat')):
            for j, (buffer_dis, subgroup) in enumerate(group.groupby('buffer_dis')):
                color = compensation_palette[j]
                subgroup.plot(ax=ax, color=color,
                              edgecolor='black', linewidth=0.5)
                handles.append(Patch(color=color))
                labels.append(f"{compensat} - Buffer: {buffer_dis}")

    handles.append(Patch(facecolor='none', edgecolor='none', label=''))
    labels.append('')

    if not interference.empty:
        interference.plot(ax=ax, color='cyan')
        handles.append(Patch(color='cyan', label='Störungsquelle'))
        labels.append('Störungsquelle')

    if not scope.empty:
        strong_colors = ['black', 'red', 'blue', 'green',
                         'purple', 'orange']  # Add more if needed
        for idx, row in scope.iterrows():
            color = strong_colors[idx % len(strong_colors)]
            ax.plot(*row.geometry.boundary.xy, color=color,
                    linestyle='dashed', linewidth=2, dashes=(5, 5))
            handles.append(Patch(color=color, linestyle='dashed', fill=False))
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
