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

    # ----- Get the average beer abv from all groups -----
    group_abv = df.groupby(group_type)['beer_abv'].mean()

    # ----- Get the max beer abv and the name of the group containing the max abv value -----
    max_value = group_abv.max()
    max_value_name = group_abv.idxmax()

    # ----- Sort group_abv series -----
    group_abv.sort_values(ascending=False, inplace=True)

    return group_abv, max_value, max_value_name

def get_mean_review_scores_dataframe(df):
    """ This function returns a DataFrame that has had its reviews transformed to be the mean of all reviews grouped by beer

    Args:
        df (Pandas DataFrame): A DataFrame containing beer review data

    Returns:
        filtered_mean_beer_data (Pandas DataFrame): A DataFrame containing the mean of the beer reviews
    """
    # ----- Create dataframe consisting only of mean review data -----
    mean_review_data = df.groupby('beer_beerid')[['review_overall','review_aroma', 'review_appearance', 'review_palate', 'review_taste']].mean()
    mean_review_data.reset_index(level=0, inplace=True)

    # ----- Create dataframe consisting of the rest of the beer data -----
    additional_beer_data = df[['brewery_id', 'brewery_name','beer_style','beer_name', 'beer_abv', 'beer_beerid']]

    # ----- Merge the two dataframes -----
    filtered_mean_beer_data = pd.merge(mean_review_data,additional_beer_data,how='left',on="beer_beerid").drop_duplicates()

    return filtered_mean_beer_data

def get_top_n_beers(df, num_beers, weights=None):
    """ This function returns the top n beers based on average ratings. 
    Weights can be given to give importance to review types if needed

    Args:
        df (Pandas DataFrame): A DataFrame containing the mean of the beer reviews
        num_beers (int): Integer representing the top n beers that should be returned
        weights (dict): Dictionary containing weights to the review types. Defaults to None.

    Returns:
        (Pandas DataFrame): A DataFrame containing the top n beers and their information.
    """
    # ----- If weights have been provided, add them to their respective reviews -----
    if weights:
        for w_name, w in weights.items():
            df[w_name] = df[w_name] + w*2

    # ----- Sum the beer score -----
    df['beer_score'] = df[['review_aroma', 'review_appearance', 'review_palate', 'review_taste']].sum(axis=1)
    
    # ----- Sort dataframe by beer score -----
    df.sort_values(by=['beer_score'], inplace=True, ascending=False)

    # ----- Return top n results -----
    return df.head(num_beers)



