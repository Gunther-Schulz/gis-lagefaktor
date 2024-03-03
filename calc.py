import os
import geopandas as gpd
import pandas as pd

INPUT_DIR = './input'
OUTPUT_DIR = './output'
# craete dirs if not exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# remove all files in output
for file in os.listdir(OUTPUT_DIR):
    os.remove(f'{OUTPUT_DIR}/{file}')


# Load the shapefiles
A1 = gpd.read_file(f'{INPUT_DIR}/A1.shp')
A2 = gpd.read_file(f'{INPUT_DIR}/A2.shp')
B1 = gpd.read_file(f'{INPUT_DIR}/B1.shp')
B2 = gpd.read_file(f'{INPUT_DIR}/B2.shp')

# Combine A1 and A2 into a single GeoDataFrame
A_combined = gpd.GeoDataFrame(pd.concat([A1, A2], ignore_index=True))

# Calculate intersections
A1_B1_intersection = gpd.overlay(A1, B1, how='intersection')
A1_B2_intersection = gpd.overlay(A1, B2, how='intersection')
A2_B1_intersection = gpd.overlay(A2, B1, how='intersection')
A2_B2_intersection = gpd.overlay(A2, B2, how='intersection')

# Subtract A1_B1_intersection from A1_B2_intersection
A1_B2_not_B1 = gpd.overlay(
    A1_B2_intersection, A1_B1_intersection, how='difference')

# Subtract A2_B1_intersection from A2_B2_intersection
A2_B2_not_B1 = gpd.overlay(
    A2_B2_intersection, A2_B1_intersection, how='difference')

# Calculate areas of intersections
A1_B1_area = A1_B1_intersection.area.sum()
A1_B2_not_B1_area = A1_B2_not_B1.area.sum()
A2_B1_area = A2_B1_intersection.area.sum()
A2_B2_not_B1_area = A2_B2_not_B1.area.sum()

# Calculate area outside B2 for each A1 and A2
A1_outside_B2 = gpd.overlay(A1, B2, how='difference')
A2_outside_B2 = gpd.overlay(A2, B2, how='difference')

# Filter to only include polygons
A1_outside_B2 = A1_outside_B2[A1_outside_B2.geometry.type == 'Polygon']
A2_outside_B2 = A2_outside_B2[A2_outside_B2.geometry.type == 'Polygon']

# Calculate areas
A1_outside_B2_area = A1_outside_B2.area.sum()
A2_outside_B2_area = A2_outside_B2.area.sum()

# Print the results
print(f"Area of A1 intersecting with B1: {A1_B1_area}")
print(f"Area of A1 intersecting with B2 but not B1: {A1_B2_not_B1_area}")
print(f"Area of A2 intersecting with B1: {A2_B1_area}")
print(f"Area of A2 intersecting with B2 but not B1: {A2_B2_not_B1_area}")
print(f"Area of A1 outside B2: {A1_outside_B2_area}")
print(f"Area of A2 outside B2: {A2_outside_B2_area}")


# Create new shapes of all of the above
if A1_B1_intersection.area.sum() > 0:
    A1_B1_intersection.to_file(f'{OUTPUT_DIR}/A1_B1_intersection.shp')
if A1_B2_not_B1.area.sum() > 0:
    A1_B2_not_B1.to_file(f'{OUTPUT_DIR}/A1_B2_not_B1.shp')
if A2_B1_intersection.area.sum() > 0:
    A2_B1_intersection.to_file(f'{OUTPUT_DIR}/A2_B1_intersection.shp')
if A2_B2_not_B1.area.sum() > 0:
    A2_B2_not_B1.to_file(f'{OUTPUT_DIR}/A2_B2_not_B1.shp')
if A1_outside_B2_area > 0:
    A1_outside_B2.to_file(f'{OUTPUT_DIR}/A1_outside_B2.shp')
if A2_outside_B2_area > 0:
    A2_outside_B2.to_file(f'{OUTPUT_DIR}/A2_outside_B2.shp')
