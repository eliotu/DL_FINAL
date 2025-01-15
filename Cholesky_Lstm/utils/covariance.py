import numpy as np
import pandas as pd
from numpy.linalg import cholesky

def calculate_hourly_realized_covariance(returns: pd.DataFrame, freq='h', return_cholesky=False, transform = True) -> dict:
    """
    Calculate hourly realized covariance matrices from 1-minute returns.

    Inputs:
    - returns: pd.DataFrame
        A DataFrame of 1-minute returns with datetime index and columns for each asset (or 1h returns).
    - freq : string
        frequencies for which we want to calculate covariances. Default is 'h' for 1 hour. The only different possible value is 'd' for daily.
    - return_cholesky : bool
        Indicates whether to return array of vectorized cholesky decompositions or array of covariance matrices.
    - transform : bool
        Whether to apply multplier for covariances (maybe adds numerical stability, but controversial)

    Outputs:
    - hourly_covariance: numpy array
        A 2D numpy array with vectorized Cholesky decompositions of covariance matrices with shape (T x N) in case return_cholesky is True.
            T - # of timestamps
            N - length of Cholesky vector
        
        A 3D numpy array with covariance matrices with shape (T x n x n) in case return_cholesky is False.
            T - # of timestamps
            n - # of stocks
    """
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("The index of the returns DataFrame must be a DatetimeIndex.")
    
    if freq not in ['h', 'd']:
        raise ValueError(f"Invalid frequency: '{freq}'. Allowed values are 'h' or 'd'.")

    hourly_groups = returns.resample(freq, closed='right', label='right')

 
    hourly_covariance = []


    for timestamp, group in hourly_groups:
        if not group.empty:
            cov_matrix = group.cov().values
            if transform:
              cov_matrix = cov_matrix*22

            if return_cholesky:

                try:
                    cholesky_decomposition = cholesky(cov_matrix)
                    cholesky_decomposition_vec = cholesky_decomposition[np.tril_indices_from(cholesky_decomposition)]
                    hourly_covariance.append(cholesky_decomposition_vec)
                except np.linalg.LinAlgError:
                   
                    cov_matrix += 1e-8 * np.eye(cov_matrix.shape[0])
                    cholesky_decomposition = cholesky(cov_matrix)
                    cholesky_decomposition_vec = cholesky_decomposition[np.tril_indices_from(cholesky_decomposition)]
                    hourly_covariance.append(cholesky_decomposition_vec)

            
            else:
                hourly_covariance.append(cov_matrix)

  
    return np.array(hourly_covariance)
