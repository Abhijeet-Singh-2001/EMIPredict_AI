"""Feature-selection helpers for the EMI eligibility dataset."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def rank_features_by_chi_square(
    dataframe: pd.DataFrame,
    target: str = "emi_eligibility",
    *,
    top_n: int | None = None,
    numeric_bins: int = 10,
) -> pd.DataFrame:
    """Rank columns by their association with a categorical target.

    Each feature is tested independently against ``target`` using a
    chi-square test of independence. Numeric features are converted to
    quantile-based categories before testing, while categorical features
    retain their categories.

    Parameters
    ----------
    dataframe:
        DataFrame containing the candidate features and target.
    target:
        Name of the categorical target column.
    top_n:
        If provided, return only the first ``top_n`` ranked features.
    numeric_bins:
        Maximum number of quantile bins used for numeric features.

    Returns
    -------
    pandas.DataFrame
        Rows sorted by ascending p-value and descending chi-square
        statistic. The result contains ``feature``, ``chi2``, ``p_value``,
        ``degrees_of_freedom``, and Cramer's ``v`` effect size.

    Raises
    ------
    TypeError
        If ``dataframe`` is not a pandas DataFrame.
    ValueError
        If the target is missing, invalid arguments are supplied, or the
        target has fewer than two non-null classes.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    if target not in dataframe.columns:
        raise ValueError(f"Target column {target!r} was not found")
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be greater than zero or None")
    if numeric_bins < 2:
        raise ValueError("numeric_bins must be at least 2")

    target_values = dataframe[target].dropna()
    if target_values.nunique() < 2:
        raise ValueError("The target must contain at least two non-null classes")

    results: list[dict[str, Any]] = []
    for feature in dataframe.columns:
        if feature == target:
            continue

        pair = dataframe[[feature, target]].dropna(subset=[target]).copy()
        if pair.empty or pair[feature].nunique(dropna=False) < 2:
            continue

        values = pair[feature]
        numeric_values = pd.to_numeric(values, errors="coerce")
        is_numeric_like = (
            pd.api.types.is_numeric_dtype(values)
            or numeric_values.notna().mean() >= 0.95
        )
        if is_numeric_like:
            # Binning avoids treating every continuous value as a separate
            # category, which can make the chi-square statistic misleading.
            categories = pd.qcut(
                numeric_values,
                q=min(numeric_bins, numeric_values.nunique()),
                duplicates="drop",
            )
        else:
            categories = values.astype("string").fillna("<missing>")

        feature_codes, _ = pd.factorize(categories, sort=True)
        target_codes, _ = pd.factorize(pair[target], sort=True)
        contingency = pd.crosstab(feature_codes, target_codes)

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue

        chi2, p_value, degrees_of_freedom, _ = chi2_contingency(contingency)
        sample_size = contingency.to_numpy().sum()
        cramers_v = (
            np.sqrt(chi2 / (sample_size * min(contingency.shape[0] - 1, contingency.shape[1] - 1)))
            if sample_size and min(contingency.shape[0] - 1, contingency.shape[1] - 1) > 0
            else 0.0
        )
        results.append(
            {
                "feature": feature,
                "chi2": chi2,
                "p_value": p_value,
                "degrees_of_freedom": degrees_of_freedom,
                "cramers_v": cramers_v,
            }
        )

    ranking = pd.DataFrame(
        results,
        columns=["feature", "chi2", "p_value", "degrees_of_freedom", "cramers_v"],
    ).sort_values(["p_value", "chi2"], ascending=[True, False], ignore_index=True)

    return ranking if top_n is None else ranking.head(top_n)
