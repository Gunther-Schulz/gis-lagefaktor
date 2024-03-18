# gis-lagefaktor

## TODO

- do linear contruction and compensatory work? Test it
- was bedeuten die area einschränkungen bei den schutzgebieten in der lagefaktor tabelle?
- lagefaktor <100 ist falsch in bänbelitz (1) ????
- es gibt noch sliver bei construction in bäbelitz

Hendrik:

- schutz gebiet und interference area lagefakltoren beide oder nur einer?
- vorbeck anschauen, mindestfläceh für kompensation im schutzgebiet?
- zählen dann die separate oder wird die mindestfläche anhand der gesamtfläceh (schutzgebiet und nicht schutzgebiet tzsammen)
- 2000 qm mindestfläche bei kompensation? auch bei schutzgebiten?

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

### Friedrichshof

Total final value: 160418.42
Total final value for compensatory features: 31148.58

### Vorbeck

Total final value: 294992.93
Total final value for compensatory features: 310305.83
(The old compensatory value was: 267277.98. I am guessing that only the large compensatory area was used for the old value)
