from typing import Callable, Any
import inspect
import re
import numpy as np
import pandas as pd

from pair_trading.constants import OFFSET_MAPPING, OFFSET_DEFAULTS, DATE_REGEX


def kwargs_allowed(obj: Callable) -> bool:
    signature = inspect.signature(obj)
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param
        in signature.parameters.values()
    )


def get_allowed_keyword_arguments(obj: Callable) -> set[str]:
    signature = inspect.signature(obj)
    keyword_arguemnts = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    return {
        param.name
        for param
        in signature.parameters.values()
        if param.kind in keyword_arguemnts
    }


def filter_allowed_kwargs(
    obj: Callable,
    kwargs: dict[str, Any],
    disable_kwargs: bool = False,
) -> dict[str, Any]:
    if not disable_kwargs and kwargs_allowed(obj):
        return kwargs

    valid_kwargs = get_allowed_keyword_arguments(obj)
    return {
        keyword: argument
        for keyword, argument
        in kwargs.items()
        if keyword in valid_kwargs
    }


def winsorize(
    df: pd.DataFrame,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
    inplace: bool = False,
) -> pd.DataFrame:
    if not inplace:
        df = df.copy()

    numeric_cols = df.select_dtypes(include=np.number).columns
    lower = df[numeric_cols].quantile(lower_pct)
    upper = df[numeric_cols].quantile(upper_pct)

    df[numeric_cols] = df[numeric_cols].clip(lower=lower, upper=upper, axis=1)

    return df


def date_offset(
    date: str | pd.Timestamp,
    offset: str,
) -> pd.Timestamp:

    date = pd.Timestamp(date)
    offset = offset.replace(" ", "").lower()

    for match in re.finditer(DATE_REGEX, offset):
        if (
            (replace_param := match.group("replace_param"))
            and (replace_value := match.group("replace_value"))
        ):
            date += pd.DateOffset(**{replace_param: int(replace_value)})

        elif (
            (add_param := match.group("add_param"))
            and (add_value := match.group("add_value"))
        ):
            date += pd.DateOffset(**{add_param: int(add_value)})

        elif (offset_param := match.group("offset_param")):
            offset_type = OFFSET_MAPPING[offset_param]
            date += offset_type(
                n=int(match.group("offset_value") or 1),
                **OFFSET_DEFAULTS.get(offset_param, {})
            )

    return date
