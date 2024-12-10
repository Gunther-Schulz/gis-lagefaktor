# config.py
import yaml
from box import Box
import os
import sys

# TODO: Don't keep all projects in settings, only the one that is currently used and have it in top level like "project" -> "path"

# General default settings
# default_settings = {
# "output_dir": "output",
# "project_name": "ProjectName",
# "crs": "epsg:",
# "buffer_gen_distances": (100, 625),
# "buffer_distances": {
#     '<100': '<100',
#     '>100<625': '>100<625',
#     '>625': '>625'
# },
# "grz_factors": {
#     "0.5": (0.5, 0.2, 0.6),
#     "0.75": (0.75, 0.5, 0.8)
# },
# "default_sliver": 0.001,
# "filter_small_areas": True,
# "filter_small_areas_limit": 1,
# "count_small_compensatory_if_adjacent": False,
# "projects": {
#     "ProjectName": {
#         "path": "/home/user/Projekts/Solar",
#         "grz": 0.5,
#         "changing_construction_base_values": {
#             "Acker": 1,
#             "Grünland": 2
#         },
#         "changing_compensatory_base_values": {
#             "Acker": 0,
#             "Acker Comp": 0,
#             "Grünland": 0
#         },
#         "compensatory_measure_values": {
#             "Wiese": 3
#         },
#         "compensatory_measure_minimum_area": {
#             "Wiese": 2000
#         },
#         "compensatory_protected_values": {
#             "VSG": 1.1,
#             "GGB": 1.1,
#             "Test": 2
#         },
#         "construction_lagefaktor_values": {
#             "<100": 0.75,
#             ">100<625": 1,
#             ">625": 1.25
#         },
#         "construction_protected_values": {
#             "NSG": 1.5,
#             "VSG": 1.25,
#             "GGB": 1.25
#         }
#     }
# }
# }

settings = Box({})


def load_config(base_config_path='config.yaml'):
    try:
        with open(base_config_path, 'r') as file:
            loaded_settings = yaml.safe_load(file) or {}
            settings.update(Box(loaded_settings))
            # Only set output_dir if project_name is set
            if hasattr(settings, 'project_name') and settings.project_name:
                settings.output_dir = settings.projects[settings.project_name].path + "/output"
    except FileNotFoundError:
        print(f"Error: Base configuration file '{base_config_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading base configuration: {str(e)}")
        sys.exit(1)


def load_project_config():
    # Skip if project_name is not set yet
    if not hasattr(settings, 'project_name') or not settings.project_name:
        return
        
    project_name = settings.project_name
    project_config_path = f'{settings.projects[project_name].path}/config.yaml'
    
    # Skip if project config doesn't exist (it will be created if --new flag is used)
    if not os.path.exists(project_config_path):
        return
        
    with open(project_config_path, 'r') as file:
        project_settings = yaml.safe_load(file) or {}
        # Merge project-specific defaults with loaded settings for the specific project
        complete_project_settings = {
            **settings.projects[project_name], **project_settings}
        settings.projects[project_name].update(Box(complete_project_settings))
