import pandas as pd

def highest_abv_by_group(df, group_type):
    """ This function calculates the max beer_abv given a dataframe with a column named 'beer_abv'

    Args:
        df (Pandas Dataframe): Dataframe containing beer data
        group_type (String): String representing column name of value we will compare beer abv on

    Returns:
        group_abv (Pandas Series ): Series containing all groups and their average beer abv
        max_value (Float): Float representing the max average beer abv from a group
        max_value_name (String): String representing the group name having the max average beer abv
    """

    highest_abv = 0
    highest_abv_group_type_name = None

    # ----- Get the average beer abv from all groups -----
    group_abv = df.groupby(group_type)['beer_abv'].mean()

    # ----- Get the max beer abv and the name of the group containing the max abv value -----
    max_value = group_abv.max()
    max_value_name = group_abv.idxmax()

    # ----- Sort group_abv series -----
    group_abv.sort_values(ascending=False, inplace=True)

    return group_abv, max_value, max_value_name
