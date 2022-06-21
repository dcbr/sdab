import numpy as np
import pandas as pd


def top_metrics(metrics, frac_skip=0.8, top_idxs=None):
    """Select the top K elements (at most) from each metric Series in the given metrics array."""
    K = 5
    if top_idxs is None:
        top_idxs = [None]*len(metrics)
    top_metrics = []
    # Determine minimum timestep to consider (data from previous timesteps will be neglected)
    k_skip = int(np.ceil(frac_skip*pd.concat(metrics, axis=1).index.max()))
    for i, metric in enumerate(metrics):
        if top_idxs[i] is None:
            # For the current metric array, determine the minimum index to consider (data from previous indices will be neglected)
            idx_skip = np.searchsorted(metric.index >= k_skip, True)
            # k is the minimum of K (the maximum amount of elements to select) and the amount of remaining datapoints in the metric array
            k = min(K, len(metric.iloc[idx_skip:]))
            # Get the idxs of the top k datapoints in the metric array
            top_idxs[i] = idx_skip + np.argpartition(metric.iloc[idx_skip:], -k)[-k:]
        # Append the top metrics from the current array to the top_metrics array
        top_metrics.extend(metric.iloc[top_idxs[i]])
    return top_metrics, top_idxs


def process_metrics(metrics, k_delta=1000, alpha=0.1):
    """Preprocess the given metrics array (each metric being a pandas Series)."""
    # Concatenate them into 1 dataframe:
    metrics_original = pd.concat(metrics, axis=1)
    metrics = metrics_original
    # Create empty dataframe with regularly spaced index:
    N = int(np.ceil(metrics.index.max() / k_delta))
    df_delta = pd.DataFrame(index=k_delta*np.arange(N+1))
    # Concatenate it with original metrics:
    metrics = pd.concat([metrics, df_delta], axis=1)
    # Interpolate missing data:
    metrics = metrics.interpolate(method="index", limit_area="inside")
    # Reindex using the regularly spaced index of df_delta:
    metrics = metrics.reindex(df_delta.index)
    # We now have a dataframe with regular timestep data (every k_delta timesteps), drop the initial missing datapoints:
    metrics = metrics.dropna()
    # Aggregate metrics, calculating the minimum, maximum and mean values across the different columns:
    metrics = metrics.agg(["min", "max", "mean"], axis=1)
    # Apply exponential smoothing filter to each column:
    metrics = metrics.ewm(alpha=alpha).mean()
    return metrics
