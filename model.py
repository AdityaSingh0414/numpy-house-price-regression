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

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
     # Convert to NumPy array
     label= np.asarray(labels)

     # Find unique labels and inverse indices
     unique_labels, indices= np.unique(labels,return_inverse=True)

     # Number of samples and categories
     N= len(labels)
     C= len(unique_labels)

     # Create output matrix
     one_hot= np.zeros((N,C), dtype=float)

     # Set the appropriate positions to 1
     one_hot[np.arange(N), indices]=1.0

     return one_hot

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    # Convert to NumPy array
    X= np.asarray(X, dtype=float)

    # Compute column-wise mean
    mean=np.mean(X,axis=0)

    # Compute column-wise standard deviation
    std= np.std(X,axis=0)

    # Replace zero std with 1
    std[std==0]= 1.0

    return mean,std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
     # Convert inputs to NumPy arrays
    X = np.asarray(X, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)

    # Standardize
    return (X - mean) / std

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
     # Convert to NumPy array
     X= np.asarray(X, dtype=float)

     # Number of rows
     N= X.shape[0]


     # Create bias column
     bias= np.ones((N,1), dtype=float)


     # Append bias column to the left
     return np.hstack((bias, X))

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    """
    Create a reproducibly shuffled permutation of row indices.

    Args:
        n_samples : int
            Number of samples.
        seed : int
            Random seed.

    Returns:
        (n_samples,) ndarray of shuffled indices.
    """

    # Create random number generator
    rng= np.random.default_rng(seed)

    # Create shuffled indices
    indices= rng.permutation(n_samples)

    return indices

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    """
    Split shuffled indices into train, validation, and test sets.

    Args:
        indices : (N,) array-like
        train_ratio : float
        val_ratio : float

    Returns:
        train_idx, val_idx, test_idx
    """

    # Convert to NumPy array
    indices = np.asarray(indices, dtype=int)

     # Total samples
    N = len(indices)

    # Compute split sizes
    train_size = int(N * train_ratio)
    val_size = int(N * val_ratio)

     # Slice the array
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    return train_idx, val_idx, test_idx

# Step 12 - subset_xy
import numpy as np

def subset_xy(X, y, indices):
    """
    Select rows of X and y at the given indices.

    Args:
        X : (N, F) array-like
        y : (N,) array-like
        indices : array-like of row indices

    Returns:
        X_sub : (M, F) ndarray
        y_sub : (M,) ndarray
    """

    # Convert to NumPy arrays
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    indices = np.asarray(indices, dtype=int)

    # Select rows
    X_sub = X[indices]
    y_sub = y[indices]

    return X_sub, y_sub

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    """
    Compute Ordinary Least Squares (OLS) weights.

    Args:
        X : (N, D) design matrix (includes bias column)
        y : (N,) target vector

    Returns:
        theta : (D,) weight vector
    """

    # Convert inputs to NumPy arrays
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)


    # Normal equation components
    A = X.T @ X
    b = X.T @ y

    # Solve A * theta = b
    theta = np.linalg.solve(A, b)

    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    # Convert inputs to NumPy arrays
    X = np.asarray(X, dtype=float)
    theta = np.asarray(theta, dtype=float)

    # Compute predictions
    predictions = np.dot(X, theta)
    return predictions

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

