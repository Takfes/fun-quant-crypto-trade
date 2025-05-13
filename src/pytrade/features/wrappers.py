import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def transformer_wrapper(transform_function):
    """
    A wrapper to convert a transformation function into a scikit-learn compatible Transformer.
    """

    class WrappedTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y=None):
            # No fitting necessary for this transformer
            return self

        def transform(self, X):
            # Ensure input is a DataFrame
            if not isinstance(X, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")

            X_ = X.copy()
            return transform_function(X_, **self.kwargs)

    return WrappedTransformer
