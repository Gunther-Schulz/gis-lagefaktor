from gis_lagefaktor.config import settings

CONSTRUCTION_LAGEFAKTOR_VALUES = settings.projects[
    settings.project_name].construction_lagefaktor_values


def add_lagefaktor_values(feature, lagefaktor_value):
    """
    This function adds 'lagefaktor' values to the given feature GeoDataFrame.

    Parameters:
    feature (GeoDataFrame): The GeoDataFrame to which to add 'lagefaktor' values.
    lagefaktor_value (float): The 'lagefaktor' value to add.

    Returns:
    GeoDataFrame: The updated GeoDataFrame with 'lagefaktor' values.
    """

    if 'prot_cons' in feature.columns:
        # Check if 'prot_cons' is not null
        is_protected_not_null = feature['prot_cons'].notnull()

        feature['lagefaktor'] = feature['prot_cons'].fillna(lagefaktor_value)
        if lagefaktor_value == CONSTRUCTION_LAGEFAKTOR_VALUES.get('<100'):
            # Only subtract 0.25 from 'lagefaktor' if 'prot_cons' is not null
            feature.loc[is_protected_not_null, 'lagefaktor'] -= 0.25
    else:
        feature['lagefaktor'] = lagefaktor_value

    # remove column prot_comp if it exists
    if 'prot_comp' in feature.columns:
        feature = feature.drop(columns='prot_comp')

    return feature
