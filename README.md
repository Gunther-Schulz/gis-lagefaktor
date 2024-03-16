# gis-lagefaktor

## Usage

### Running the script

    ```bash
    python3 compensation.py --new PROJECT_NAME
    ```
    where `PROJECT_NAME` is the name of the project. This will create a new directory structure for the project.
    You can then copy the input shape files into the appropriate directories and run the script again without the `--new` flag.

    ```bash
    python3 compensation.py
    ```

### Directory structure

DATA/Project Name
|
+-- changing
|
+-- compensatory
|
+-- construction
|
+-- protected
|
+-- interference
|
+-- scope
|
+-- unchanging
|
+-- output
|
+-- debug

## Input shape files

### changing

Shapefiles that define areas that are part of the planned development

### compensatory

Shapefiles that define areas that are part of the compensatory measures

### construction

Shapefiles that define areas that are part of the construction areas

### unchanging

Shapefiles that define areas that are not part of the planned development. This is optional. If they cover construction or compensatory areas, those parts will be removed from the calculations and the output shapes.

### protected

Shapefiles that define areas that are part of the protected areas such as nature reserves

### interference

Shapefiles that define areas that are part of the interference areas are used to calculate the interference factors as part of the Lagefaktor calculations for the construction areas

### scope

Shapefiles that define the scope of the planned development. This is optional. If it is not provided, the scope is calculated as the union of the changing, compensatory and construction areas.

## Output shape files

### output

The output directory contains the compensatory and construction shapes with all relevant calculations.

### debug

The debug directory contains debug shape files that are used to visualize the intermediate results of the calculations.

## Testing scores

### Bäbelitz:

Total final value: 196169.43
Total final value for compensatory features: 278066.28

### Vorbeck

Total final value: 294992.93
Total final value for compensatory features: 267277.98

### Friedrichshof

Total final value: 160418.42
Total final value for compensatory features: 31148.58
