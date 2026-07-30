"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import numpy as np

def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0)."""

    # Convert to NumPy float array
    X = np.asarray(X, dtype=float)

    # Make a copy
    result = X.copy()

    # Compute column means ignoring NaNs
    col_means = np.nanmean(result, axis=0)

    # Replace NaN means (all-NaN columns) with 0
    col_means[np.isnan(col_means)] = 0.0

    # Find NaN positions
    mask = np.isnan(result)

    # Replace NaNs with corresponding column means
    result[mask] = np.take(col_means, np.where(mask)[1])

    return result

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    """
    Compute per-column lower and upper bounds using the IQR rule.

    Args:
        X : (N, F) array-like of numeric values.
        k : Multiplier for the IQR (default = 1.5).

    Returns:
        lower : (F,) ndarray of lower bounds.
        upper : (F,) ndarray of upper bounds.
    """

    # Convert input to NumPy float array
    X= np.asarray(X, dtype=float)

    # Compute the 25th and 75th percentiles for each column
    q1= np.percentile(X,25, axis=0)
    q3= np.percentile(X, 75, axis=0)

    # Compute the Interquartile Range (IQR)
    iqr=q3-q1

    # Compute lower and upper bounds
    lower= q1-k*iqr
    upper=q3+k*iqr

    return lower , upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    """
    Clip every entry of a feature matrix to per-column lower and upper bounds.

    Args:
        X : (N, F) array-like of numeric values.
        lower : (F,) array of lower bounds.
        upper : (F,) array of upper bounds.

    Returns:
        (N, F) ndarray with values clipped to the specified bounds.
    """

    # Convert inputs to NumPy arrays

    X= np.asarray(X, dtype= float)
    lower = np.asarray(lower, dtype= float)
    upper = np.asarray(upper, dtype=float)

    ##
   ## np.clip() compares every value in each column with its corresponding lower and upper bound.
##If a value is:
##less than lower → replace with lower
##greater than upper → replace with upper
##otherwise → keep the original value.

    # Return a new clipped array (does not modify X)
    return np.clip(X, lower, upper)

# Step 4 - make_ratio_feature
import numpy as np

def make_ratio_feature(numerator, denominator, eps=1e-8):
    """
    Form a derived ratio feature from two 1-D arrays with safe division.

    Args:
        numerator: (N,) array-like
        denominator: (N,) array-like
        eps: Small value added to denominator to avoid division by zero.

    Returns:
        (N,) ndarray containing numerator / (denominator + eps)
    """

    # Convert inputs to NumPy float arrays
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    # Safe division
    return numerator / (denominator + eps)

# Step 5 - append_column
import numpy as np

def append_column(X, col):
    # Convert inputs to NumPy arrays
    X = np.asarray(X, dtype=float)
    col = np.asarray(col, dtype=float)

    # Convert to column vector
    col = col.reshape(-1, 1)

    # Append the column
    return np.hstack((X, col))

# Step 6 - one_hot_encode (not yet solved)
# TODO: implement

# Step 7 - fit_standardizer (not yet solved)
# TODO: implement

# Step 8 - apply_standardizer (not yet solved)
# TODO: implement

# Step 9 - add_bias_column (not yet solved)
# TODO: implement

# Step 10 - make_shuffled_indices (not yet solved)
# TODO: implement

# Step 11 - partition_indices (not yet solved)
# TODO: implement

# Step 12 - subset_xy (not yet solved)
# TODO: implement

# Step 13 - ols_fit (not yet solved)
# TODO: implement

# Step 14 - ols_predict (not yet solved)
# TODO: implement

# Step 15 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 16 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 17 - r_squared (not yet solved)
# TODO: implement

# Step 18 - residual_summary (not yet solved)
# TODO: implement

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

