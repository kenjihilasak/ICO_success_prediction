
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ── 1) Custom imputers ──────────────────────────────────────────────────────────

class YearMeanImputer(BaseEstimator, TransformerMixin):
    """Impute missing values by the mean computed per ico_year."""
    def __init__(self, year_col: str, target_col: str):
        self.year_col = year_col
        self.target_col = target_col
        self.year_means_ = {}

    def fit(self, X, y=None):
        df = X[[self.year_col, self.target_col]].copy()
        # Compute per-year mean, excluding NaNs
        self.year_means_ = df.dropna(subset=[self.target_col]) \
                              .groupby(self.year_col)[self.target_col] \
                              .mean().to_dict()
        # Global fallback mean
        self.global_mean_ = df[self.target_col].mean()
        return self

    def transform(self, X):
        X = X.copy()
        def impute(row):
            if pd.isna(row[self.target_col]):
                return self.year_means_.get(row[self.year_col], self.global_mean_)
            return row[self.target_col]

        X[self.target_col] = X.apply(impute, axis=1)
        return X[[self.target_col]]


class CountryMeanImputer(BaseEstimator, TransformerMixin):
    """Impute missing values by the mean computed per country; fallback to global median."""
    def __init__(self, country_col: str, target_col: str):
        self.country_col = country_col
        self.target_col = target_col
        self.country_means_ = {}

    def fit(self, X, y=None):
        df = X[[self.country_col, self.target_col]].copy()
        self.country_means_ = df.dropna(subset=[self.target_col]) \
                                .groupby(self.country_col)[self.target_col] \
                                .mean().to_dict()
        self.global_median_ = df[self.target_col].median()
        return self

    def transform(self, X):
        X = X.copy()
        def impute(row):
            if pd.isna(row[self.target_col]):
                return self.country_means_.get(row[self.country_col], self.global_median_)
            return row[self.target_col]

        X[self.target_col] = X.apply(impute, axis=1)
        return X[[self.target_col]]



class DummyModeImputer(BaseEstimator, TransformerMixin):
    """Impute missing values in dummy/binary columns using mode (most frequent value)."""
    
    def __init__(self, cols_to_impute: list):
        """
        Parameters:
        -----------
        cols_to_impute : list
            List of column names to impute
        """
        self.cols_to_impute = cols_to_impute
        self.modes_ = {}
    
    def fit(self, X, y=None):
        """
        Compute the mode for each column in cols_to_impute.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data
        y : None
            Ignored
        """
        for col in self.cols_to_impute:
            self.modes_[col] = X[col].mode()[0]  # Get most frequent value
        return self
    
    def transform(self, X):
        """
        Impute missing values using pre-computed modes.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data to transform
        
        Returns:
        --------
        pd.DataFrame
            Data with imputed values
        """
        X = X.copy()
        for col in self.cols_to_impute:
            X[col] = X[col].fillna(self.modes_[col])
            X[col] = X[col].astype(int)  # Ensure integer type for dummy variables
        return X[self.cols_to_impute]


class DropNAColumns(BaseEstimator, TransformerMixin):
    """
    Drops rows where any of the specified columns are NaN.
    """
    def __init__(self, cols_to_drop: list):
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.dropna(subset=self.cols_to_drop).reset_index(drop=True)


class RowCounter(BaseEstimator, TransformerMixin):
    """Transformer that counts rows before and after preprocessing."""
    def __init__(self, name=''):
        self.name = name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print(f"{self.name} shape: {X.shape[0]} rows, {X.shape[1]} columns")
        return X