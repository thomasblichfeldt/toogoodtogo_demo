import pandas as pd
import numpy as np
import random

def highest_abv_by_group(input_df, group_type):
    """ This function calculates the max beer_abv given a dataframe with a column named 'beer_abv'

    Args:
        input_df (Pandas Dataframe): Dataframe containing beer data
        group_type (String): String representing column name of value we will compare beer abv on

    Returns:
        group_abv (Pandas Series ): Series containing all groups and their average beer abv
        max_value (Float): Float representing the max average beer abv from a group
        max_value_name (String): String representing the group name having the max average beer abv
    """

    # ----- Copy input dataframe -----
    df = input_df.copy()

    # ----- Drop duplicates of beer names to get accurate average -----
    df.drop_duplicates(subset='beer_name', keep="first", inplace=True)

    # ----- Get the average beer abv from all groups -----
    group_abv = df.groupby(group_type)['beer_abv'].mean()

    # ----- Get the max beer abv and the name of the group containing the max abv value -----
    max_value = group_abv.max()
    max_value_name = group_abv.idxmax()

    # ----- Sort group_abv series -----
    group_abv.sort_values(ascending=False, inplace=True)

    return group_abv, max_value, max_value_name

def get_mean_review_scores_dataframe(input_df):
    """ This function returns a DataFrame that has had its reviews transformed to be the mean of all reviews grouped by beer

    Args:
        input_df (Pandas DataFrame): A DataFrame containing beer review data

    Returns:
        filtered_mean_beer_data (Pandas DataFrame): A DataFrame containing the mean of the beer reviews
    """

    # ----- Copy input dataframe -----
    df = input_df.copy()

    # ----- Create dataframe consisting only of mean review data -----
    mean_review_data = df.groupby('beer_beerid')[['review_overall','review_aroma', 'review_appearance', 'review_palate', 'review_taste']].mean()
    mean_review_data.reset_index(level=0, inplace=True)

    # ----- Create dataframe consisting of the rest of the beer data -----
    additional_beer_data = df[['brewery_id', 'brewery_name','beer_style','beer_name', 'beer_abv', 'beer_beerid']]

    # ----- Merge the two dataframes -----
    filtered_mean_beer_data = pd.merge(mean_review_data,additional_beer_data,how='left',on="beer_beerid").drop_duplicates()

    return filtered_mean_beer_data

def get_top_n_beers(input_df, num_beers, weights=None):
    """ This function returns the top n beers based on average ratings. 
    Weights can be given to give importance to review types if needed

    Args:
        input_df (Pandas DataFrame): A DataFrame containing the mean of the beer reviews
        num_beers (int): Integer representing the top n beers that should be returned
        weights (dict): Dictionary containing weights to the review types. Defaults to None.

    Returns:
        (Pandas DataFrame): A DataFrame containing the top n beers and their information.
    """

    # ----- Copy input dataframe -----
    df = input_df.copy()

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

def generate_recommendation(user_id, model, beer_metadata, thresh=4):
    
    """
    Generates a beer recommendation for a user based on a rating threshold. Only
    beers with a predicted rating at or above the threshold will be recommended

    based on code from following article: https://towardsdatascience.com/how-you-can-build-simple-recommender-systems-with-surprise-b0d32a8e4802
    
    Args:
        user_id (str): String representing username
        model (Object): Model object trained to give beer recommendations
        beer_metadata (dict): Dictionary containing information about a beer
        thresh (float): Float representing minimum predicted rating a beer should have. Defaults to 4.

    Returns:
        (Pandas DataFrame): A DataFrame containing the top n beers and their information.

    """
    
    # ----- Get beer names as a list -----
    beer_names = list(beer_metadata.keys())
    beers_above_thresh = []
    
    # ----- Get all beers above the theshold -----
    for beer_name in beer_names:
        beer_id = beer_metadata[beer_name]
        rating = model.predict(uid=user_id, iid=beer_id)
        if rating.est >= thresh:
            beers_above_thresh.append([beer_name, rating.est])

    beers_above_thresh = np.array(beers_above_thresh)

    # ----- sort results and return the highest rated beer -----
    beers_above_thresh = beers_above_thresh[np.argsort(beers_above_thresh[:, 1].astype(float))]

    return beers_above_thresh[-1]


