# config.py
import yaml
from box import Box

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
    with open(base_config_path, 'r') as file:
        loaded_settings = yaml.safe_load(file) or {}
        settings.update(Box(loaded_settings))
        settings.output_dir = settings.projects[settings.project_name].path + "/output"


def load_project_config():
    project_name = settings.project_name
    project_config_path = f'{settings.projects[project_name].path}/config.yaml'
    with open(project_config_path, 'r') as file:
        project_settings = yaml.safe_load(file) or {}
        # Merge project-specific defaults with loaded settings for the specific project
        complete_project_settings = {
            **settings.projects[project_name], **project_settings}
        settings.projects[project_name].update(Box(complete_project_settings))
