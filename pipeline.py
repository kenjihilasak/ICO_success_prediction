import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler


# Import custom imputers
from customImputers import (
    YearMeanImputer,
    CountryMeanImputer,
    DropNAColumns,
    DummyModeImputer,
    RowCounter
)

# ---- Column definitions ----
numerical_cols_robust_impute = ['fundraising_duration_days']
numerical_cols_median_impute = [
    'days_since_start',
    'ico_end_month_sin',
    'ico_end_month_cos',
    'ico_end_weekday_sin',
    'ico_end_weekday_cos'
]
numerical_cols_standard_scale = ['rating']
dummy_cols_mode_impute = ['kyc', 'bonus', 'whitelist', 'mvp', 'ERC20']
binary_cols = [
    'ERC20_missing', 'whitelist_missing', 'has_pre_ico',
    'top_5_countries', 'usa_restricted',
    'link_white_paper_flag', 'linkedin_link_flag',
    'github_link_flag', 'website_flag',
    'accept_only_BTC_ETH_LTC', 'accept_fiat',
    'bull_run_2017', 'crash_2018'
]
# Additional numeric variables to passthrough without processing
passthrough_numeric = [
    'min_investment_usd_converted',
    'total_restricted_areas',
    'cryptocurrencies_accepted'
]

# ---- Preprocessing ColumnTransformer ----
preprocessor = ColumnTransformer(
    transformers=[
        # 1) Drop rows with missing 'rating'
        ('drop_rating_na', DropNAColumns(['rating']), ['rating']),

        # 2) Year-mean imputation for 'price_usd_converted'
        ('impute_price', YearMeanImputer('ico_year', 'price_usd_converted'),
         ['ico_year', 'price_usd_converted']),

        # 3) Year-mean imputation for 'distributed_in_ico'
        ('impute_dist', YearMeanImputer('ico_year', 'distributed_in_ico'),
         ['ico_year', 'distributed_in_ico']),

        # 4) Country-mean imputation for 'teamsize'
        ('impute_team', CountryMeanImputer('country', 'teamsize'),
         ['country', 'teamsize']),

        # 5) Drop column 'country' after imputation
        ('drop_country', 'drop', ['country']),

        # 6) Median imputation for 'ico_year' missing values
        ('impute_ico_year', SimpleImputer(strategy='median'), ['ico_year']),

        # 7) Mode imputation for dummy variables
        ('impute_dummies', DummyModeImputer(dummy_cols_mode_impute),
         dummy_cols_mode_impute),

        # 8) Robust scaling for skewed numeric columns
        ('scale_robust', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ]), numerical_cols_robust_impute),

        # 9) Median imputation + standard scaling for cyclic features
        ('handle_numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols_median_impute),

        # 10) Standard scaling for 'rating'
        ('scale_rating', StandardScaler(), numerical_cols_standard_scale),

        # 11) Passthrough processed binary columns
        ('pass_binary', 'passthrough', binary_cols),

        # 12) Passthrough additional numeric columns
        ('pass_numeric', 'passthrough', passthrough_numeric)
    ],
    remainder='drop'
)

# Export preprocessor and RowCounter
__all__ = ['preprocessor', 'RowCounter']