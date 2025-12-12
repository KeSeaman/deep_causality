"""
Deep Causality - Python bindings for Rust causal inference engine

A safety engine for AI that asks "Why?" before "What?"

Example:
    >>> import deep_causality
    >>> import polars as pl
    >>> 
    >>> df = pl.read_parquet("data.parquet")
    >>> features = deep_causality.run_mrmr(df, target="SepsisLabel", max_features=10)
    >>> for f in features:
    ...     print(f"{f.name}: {f.score:.4f}")
"""

from deep_causality._core import (
    FeatureRanking,
    SurdResult,
    run_mrmr,
    run_mrmr_from_dict,
    version,
)

__version__ = version()
__all__ = [
    "FeatureRanking",
    "SurdResult", 
    "run_mrmr",
    "run_mrmr_from_dict",
    "run_mrmr_polars",
    "version",
]


def run_mrmr_polars(df, target: str, max_features: int = 10):
    """
    Run mRMR feature selection on a Polars DataFrame.
    
    Args:
        df: Polars DataFrame with numeric columns
        target: Name of the target column
        max_features: Maximum number of features to select (default: 10)
    
    Returns:
        List of FeatureRanking objects with .name and .score attributes
    
    Example:
        >>> import polars as pl
        >>> import deep_causality
        >>> 
        >>> df = pl.read_parquet("icu_data.parquet")
        >>> features = deep_causality.run_mrmr_polars(df, target="SepsisLabel")
        >>> print(features[0])
        FeatureRanking(name='ICULOS', score=1.0000)
    """
    # Convert Polars DataFrame to dict of lists
    df_dict = {col: df[col].to_list() for col in df.columns}
    return run_mrmr_from_dict(df_dict, target, max_features)
