import pytest

import numpy as np
import pandas as pd

from src.beer_utils import *

TEST_DATA_PATH = 'data/beer_reviews_test_data.csv'
beer_test_data = pd.read_csv(TEST_DATA_PATH)
beer_test_data_with_null = beer_test_data.loc[beer_test_data.brewery_name == 'Vecchio Birraio', :] = np.nan 

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