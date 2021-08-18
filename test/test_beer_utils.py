import pytest

import numpy as np
import pandas as pd

from src.beer_utils import *

TEST_DATA_PATH = 'data/beer_reviews_test_data.csv'
TEST_DATA_FILTERED_PATH = 'data/beer_reviews_filtered_test_data.csv'

beer_test_data = pd.read_csv(TEST_DATA_PATH)
beer_test_data_with_null = beer_test_data.loc[beer_test_data.brewery_name == 'Vecchio Birraio', :] = np.nan 
beer_filtered_test_data = pd.read_csv(TEST_DATA_FILTERED_PATH)

def test_highest_abv_by_group():

    # ----- Get highest beer abv by brewery name -----
    brewery_average_abv, max_average_brewery_abv, max_average_brewery_abv_name = highest_abv_by_group(beer_test_data, 'brewery_name')

    # ----- Assert that output are expected types -----
    assert(isinstance(brewery_average_abv, pd.Series))
    assert(isinstance(max_average_brewery_abv, float))
    assert(isinstance(max_average_brewery_abv_name, str))

    # ----- Assert that returned max abv is the max value from the series -----
    assert (max_average_brewery_abv == brewery_average_abv.max())

    # ----- Assert that function fails if a group is null -----
    with pytest.raises(AttributeError) as atr_err:
        brewery_average_abv, max_average_brewery_abv, max_average_brewery_abv_name = highest_abv_by_group(beer_test_data_with_null, 'brewery_name')

def test_get_mean_review_scores_dataframe():
    
    # ----- Get dataframe with mean reviews -----
    mean_reviews_dataframe = get_mean_review_scores_dataframe(df=beer_test_data)

    # ----- Assert that a dataframe is returned -----
    assert(isinstance(mean_reviews_dataframe, pd.DataFrame))

    # ----- Assert that returned dataframe has even or less rows than inserted -----
    assert(beer_test_data.shape[0] >= mean_reviews_dataframe.shape[0])

    # ----- Assert that mean is correct -----

    # Get all reviews from beer Rauch Ür Bock
    beer_rauch = beer_test_data[beer_test_data['beer_name'] == 'Rauch Ür Bock']

    # Calculate mean
    beer_rauch_mean = beer_rauch[['review_overall','review_aroma', 'review_appearance', 'review_palate', 'review_taste']].mean()
    
    # Get mean values from already transformed dataframe
    beer_rauch_from_transformed_df = mean_reviews_dataframe[mean_reviews_dataframe['beer_name'] == 'Rauch Ür Bock']

    assert(beer_rauch_mean.equals(beer_rauch_from_transformed_df[['review_overall','review_aroma', 'review_appearance', 'review_palate', 'review_taste']].iloc[0]))


def test_get_top_n_beers():

    beer_weights = {
        'review_aroma': 0.6075657393404831,
        'review_appearance': 0.49466611599955945,
        'review_palate': 0.6955274387613946,
        'review_taste': 0.7840815193033372
        }

    beer_weights_2 = {
        'review_aroma': 0.6075657393404831,
        'review_appearance': 0.49466611599955945
        }

    bad_weights = {
        'review_aroma': 0.6075657393404831,
        'bad_weights': 0.1234
    }

    # ----- Get top 3 beers given no weigts -----
    top_3_beers = get_top_n_beers(df=beer_filtered_test_data, num_beers=3)

    # ----- Assert that a dataframe with 3 rows is returned -----
    assert(isinstance(top_3_beers, pd.DataFrame))
    assert(top_3_beers.shape[0] == 3)

    # ----- Get top 5 beers given weights on all parameters -----
    top_5_beers_with_weights = get_top_n_beers(df=beer_filtered_test_data, num_beers=5, weights=beer_weights)

    # ----- Assert that the beer score is the sum of reviews -----
    assert top_5_beers_with_weights['beer_score'].equals(top_5_beers_with_weights[['review_aroma', 'review_appearance', 'review_palate', 'review_taste']].sum(axis=1))

    # ----- Get top 5 beers given weights on only aroma and appearence -----
    top_5_beers_with_selected_weights = get_top_n_beers(df=beer_filtered_test_data, num_beers=5, weights=beer_weights_2)

    # ----- Assert that 5 beers are returned -----
    assert(top_5_beers_with_weights.shape[0] == 5)
    assert(top_5_beers_with_selected_weights.shape[0] == 5)

    # ----- Assert that function fails if a non-existent weight is given -----
    with pytest.raises(KeyError) as key_err:
        get_top_n_beers(df=beer_filtered_test_data, num_beers=3, weights=bad_weights)

    

