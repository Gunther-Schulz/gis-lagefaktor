import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import colorsys
from gis_lagefaktor.config import settings
import os
import numpy as np
import random


def generate_distinct_colors(n):
    hues = np.linspace(0, 1, n, endpoint=False)
    np.random.shuffle(hues)
    colors = [colorsys.hsv_to_rgb(h, 0.8, 0.8) for h in hues]
    return colors


def generate_similar_colors(base_color, n):
    h, s, v = colorsys.rgb_to_hsv(*base_color)
    colors = [colorsys.hsv_to_rgb((h + 0.05 * i) % 1,
                                  max(0.4, min(1, s - 0.1 + 0.2 * i / n)),
                                  max(0.4, min(1, v - 0.1 + 0.2 * i / n)))
              for i in range(n)]
    return colors


def create_plot(construction_features, compensation_features, interference, scope, output_path, show_plot=False):
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    handles = []
    labels = []

    base_names = construction_features['base_name'].unique()
    buffer_distances = construction_features['buffer_dis'].unique()

    # Generate distinct base colors for each group
    all_base_colors = generate_distinct_colors(
        len(base_names) + 1)  # +1 for compensation
    flachentyp_base_colors = {name: color for name,
                              color in zip(base_names, all_base_colors[:-1])}
    compensation_base_color = all_base_colors[-1]

    if not construction_features.empty and 'buffer_dis' in construction_features.columns and 'base_name' in construction_features.columns:
        handles.append(
            Patch(facecolor='none', edgecolor='none', label='Flächentypen'))
        labels.append('Flächentypen')

        for base_name in base_names:
            colors = generate_similar_colors(
                flachentyp_base_colors[base_name], len(buffer_distances))
            for i, buffer_dis in enumerate(buffer_distances):
                subgroup = construction_features[(construction_features['base_name'] == base_name) &
                                                 (construction_features['buffer_dis'] == buffer_dis)]
                if not subgroup.empty:
                    color = colors[i]
                    subgroup.plot(ax=ax, color=color,
                                  edgecolor='black', linewidth=0.5)
                    handles.append(Patch(color=color))
                    labels.append(f"{base_name} - {buffer_dis}")

    handles.append(Patch(facecolor='none', edgecolor='none', label=''))
    labels.append('')

    if not compensation_features.empty and 'compensat' in compensation_features.columns:
        handles.append(Patch(facecolor='none', edgecolor='none',
                       label='Kompensationswerte'))
        labels.append('Kompensationswerte')

        compensation_values = compensation_features['compensat'].unique()
        colors = generate_similar_colors(
            compensation_base_color, len(buffer_distances))

        for compensat in compensation_values:
            for i, buffer_dis in enumerate(buffer_distances):
                subgroup = compensation_features[(compensation_features['compensat'] == compensat) &
                                                 (compensation_features['buffer_dis'] == buffer_dis)]
                if not subgroup.empty:
                    color = colors[i]
                    subgroup.plot(ax=ax, color=color,
                                  edgecolor='black', linewidth=0.5)
                    handles.append(Patch(color=color))
                    labels.append(f"{compensat} - {buffer_dis}")

    handles.append(Patch(facecolor='none', edgecolor='none', label=''))
    labels.append('')

    if not interference.empty:
        interference.plot(ax=ax, color='cyan')
        handles.append(Patch(color='cyan', label='Störungsquelle'))
        labels.append('Störungsquelle')

    if not scope.empty:
        strong_colors = ['black', 'red', 'blue', 'green', 'purple', 'orange']
        for idx, row in scope.iterrows():
            color = strong_colors[idx % len(strong_colors)]
            ax.plot(*row.geometry.boundary.xy, color=color,
                    linestyle='dashed', linewidth=2, dashes=(5, 5))
            handles.append(Patch(color=color, linestyle='dashed', fill=False))
            labels.append(
                row['name'] if 'name' in row else f'Geltungsbereich {idx+1}')

    plt.legend(handles=handles, labels=labels,
               loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(settings.project_name)
    plt.tight_layout()

    crs = construction_features.crs
    if crs:
        fig.text(
            0.5, -0.02, f"Coordinate System: {crs.to_string()}", fontsize=8, ha='center', va='top')

    plt.savefig(os.path.join(output_path, settings.project_name + '_plot.png'),
                dpi=600, bbox_inches='tight', pad_inches=0.1)
    if show_plot:
        plt.show()
