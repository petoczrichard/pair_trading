import pandas.tseries.offsets as offsets


FORMATION_BACKGROUND_COLOR = "lightgreen"
TRADING_BACKGROUND_COLOR = "pink"

FORMATION_COLOR_SCALE = [
    [0.0,  "#1a3a5c"],
    [0.15, "#1a5c6e"],
    [0.3,  "#1a7a6e"],
    [0.45, "#1f9e6e"],
    [0.6,  "#31b554"],
    [0.72, "#6cc735"],
    [0.83, "#a8d826"],
    [0.92, "#d4e829"],
    [1.0,  "#fde725"],
]
TRADING_COLOR_SCALE = [
    [0.0,  "#f5ccd4"],
    [0.15, "#f0a0b0"],
    [0.3,  "#e8607a"],
    [0.45, "#d42a48"],
    [0.6,  "#b01030"],
    [0.72, "#880a20"],
    [0.83, "#600518"],
    [0.92, "#3d0210"],
    [1.0,  "#1a0008"],
]

LONG_COLOR = "orange"
SHORT_COLOR = "green"

SPREAD_COLOR = "black"
SPREAD_LINE_STYLE = "solid"

EQUITY_CURVE_COLOR = "black"
EQUITY_CURVE_LINE_STYLE = "solid"

PRICE1_COLOR = "blue"
PRICE2_COLOR = "red"

ENTRY_LINE_STYLE = "dash"
EXIT_LINE_STYLE = "dot"
STOPLOSS_LINE_STYLE = "solid"

ISO8601_DATE_FORMAT = "%Y-%m-%d"
REPLACE_PARAMS = [
    "year",
    "mnth",
    "day",
    "weekday",
    "hour",
    "minute",
    "second",
    "microsecond",
]

ADD_PARAMS = [
    "years",
    "months",
    "weeks",
    "days",
    "hours",
    "minutes",
    "seconds",
    "milliseconds",
    "microseconds",
    "nanoseconds",
]

OFFSET_MAPPING = {
    "monthbegin": offsets.MonthBegin,
    "monthend": offsets.MonthEnd,
    "semimonthbegin": offsets.SemiMonthBegin,
    "semimonthend": offsets.SemiMonthEnd,
    "quarterbegin": offsets.QuarterBegin,
    "quarterend": offsets.QuarterEnd,
    "yearbegin": offsets.YearBegin,
    "yearend": offsets.YearEnd,
}

OFFSET_DEFAULTS = {
    "quarterbegin": {"startingMonth": 1},
    "quarterend": {"startingMonth": 3},
}

DEFAULT_OFFSET = {"days": 0}

REPLACE_PATTERN = rf"(?P<replace_param>{'|'.join(REPLACE_PARAMS)})=(?P<replace_value>\d+)"  # noqa: E501
ADD_PATTERN = rf"(?P<add_value>[+-]?\d+)(?P<add_param>{'|'.join(ADD_PARAMS)})"
OFFSET_PATTERN = rf"(?P<offset_value>[+-]?\d+?)?(?P<offset_param>{'|'.join(OFFSET_MAPPING.keys())})"  # noqa: E501

DATE_REGEX = f"{REPLACE_PATTERN}|{ADD_PATTERN}|{OFFSET_PATTERN}"
