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
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # Use Moore-Penrose pseudo-inverse
    theta = np.linalg.pinv(X) @ y

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

# Step 15 - mean_absolute_error
import numpy as np

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Step 16 - root_mean_squared_error
import numpy as np

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Step 17 - r_squared
import numpy as np

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)



    ##1. Residual Sum of Squares (SS_res)
    ##SSres​=∑(ytrue​−ypred​)2

    ##2. Total Sum of Squares (SS_tot)
    #Measures how much the actual values vary from their mean.
    #SStot​=∑(ytrue​−yˉ​)2

# Step 18 - residual_summary
import numpy as np

def residual_summary(y_true, y_pred):
    residuals = y_true - y_pred

    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "median_abs": float(np.median(np.abs(residuals)))
    }

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    # Step 1: Replace NaN values
    X_clean = impute_nan_with_mean(X)

    # Step 2: Compute IQR bounds
    lower, upper = compute_iqr_bounds(X_clean, iqr_k)

    # Step 3: Clip outliers
    X_clean = clip_columns(X_clean, lower, upper)

    # Step 4: Return cleaned data
    return X_clean

# Step 20 - assemble_feature_matrix
import numpy as np

def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # Step 1: Extract numerator and denominator columns
    numerator = X_num[:, ratio_num_idx]
    denominator = X_num[:, ratio_den_idx]

    # Step 2: Create ratio feature
    ratio = make_ratio_feature(numerator, denominator)

    # Step 3: Append ratio column
    X = append_column(X_num, ratio)

    # Step 4: Add one-hot encoded categorical block if provided
    if cat_labels is not None:
        cat_block = one_hot_encode(cat_labels)
        X = np.hstack((X, cat_block))

    return X

# Step 21 - make_train_val_test
import numpy as np

def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # Step 1: Total samples
    N = len(X)

    # Step 2: Shuffle indices
    np.random.seed(seed)
    indices = np.random.permutation(N)

    # Step 3: Shuffle X and y together
    X = X[indices]
    y = y[indices]

    # Step 4: Compute split sizes
    train_size = int(N * train_ratio)
    val_size = int(N * val_ratio)

    # Step 5: Split data
    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]

    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    # Step 6: Return dictionary
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }

# Step 22 - standardize_and_add_bias
import numpy as np

def standardize_and_add_bias(splits):
    # Training features
    X_train = splits['X_train']

    # Compute mean and std only from training data
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Avoid division by zero
    std = np.where(std == 0, 1, std)

    std_splits = {}

    # Standardize each feature matrix and add bias column
    for key in ['X_train', 'X_val', 'X_test']:
        X = splits[key]
        X_standardized = (X - mean) / std

        # Prepend bias column of ones
        bias = np.ones((X_standardized.shape[0], 1))
        X_with_bias = np.hstack((bias, X_standardized))

        std_splits[key] = X_with_bias

    # Keep target vectors unchanged
    std_splits['y_train'] = splits['y_train']
    std_splits['y_val'] = splits['y_val']
    std_splits['y_test'] = splits['y_test']

    return std_splits, mean, std

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'r2': r_squared(y_true, y_pred),
        'residual_summary': residual_summary(y_true, y_pred)
    }

# Step 24 - house_price_pipeline
def house_price_pipeline(
    X,
    y,
    ratio_num_idx,
    ratio_den_idx,
    cat_labels=None,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42,
    iqr_k=1.5
):
    # Step 1: Clean numeric features
    X = prepare_cleaned_features(X, iqr_k)

    # Step 2: Assemble feature matrix
    X = assemble_feature_matrix(
        X,
        ratio_num_idx,
        ratio_den_idx,
        cat_labels
    )

    # Step 3: Split data
    splits = make_train_val_test(
        X,
        y,
        train_ratio,
        val_ratio,
        seed
    )

    # Step 4: Standardize and add bias
    std_splits, _, _ = standardize_and_add_bias(splits)

    # Step 5: Fit OLS model
    theta = ols_fit(
        std_splits["X_train"],
        std_splits["y_train"]
    )

    # Step 6: Predict
    y_val_pred = ols_predict(std_splits["X_val"], theta)
    y_test_pred = ols_predict(std_splits["X_test"], theta)

    # Step 7: Evaluate
    val_metrics = evaluate_predictions(
        std_splits["y_val"],
        y_val_pred
    )

    test_metrics = evaluate_predictions(
        std_splits["y_test"],
        y_test_pred
    )

    # Step 8: Return results
    return {
        "theta": theta,
        "y_test": std_splits["y_test"],
        "y_test_pred": y_test_pred,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics
    }

