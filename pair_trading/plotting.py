import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pair_trading.utils import filter_allowed_kwargs
from pair_trading.constants import (
    FORMATION_BACKGROUND_COLOR,
    TRADING_BACKGROUND_COLOR,
    FORMATION_COLOR_SCALE,
    TRADING_COLOR_SCALE,
    LONG_COLOR,
    SHORT_COLOR,
    SPREAD_COLOR,
    SPREAD_LINE_STYLE,
    EQUITY_CURVE_COLOR,
    EQUITY_CURVE_LINE_STYLE,
    PRICE1_COLOR,
    PRICE2_COLOR,
    ENTRY_LINE_STYLE,
    EXIT_LINE_STYLE,
    STOPLOSS_LINE_STYLE,
    ISO8601_DATE_FORMAT,
)


def _has_secondary_y(fig: go.Figure) -> bool:
    return any(getattr(trace, "yaxis", None) == "y2" for trace in fig.data)


def _clean_trace(trace):
    trace_dict = trace.to_plotly_json()
    trace_dict.pop("xaxis", None)
    trace_dict.pop("yaxis", None)
    return trace_dict


def _clean_axis(axis):
    axis_dict = axis.to_plotly_json()
    axis_dict.pop("domain", None)
    axis_dict.pop("anchor", None)
    axis_dict.pop("overlaying", None)
    return axis_dict


def _build_specs(figures, rows, columns):
    specs = []
    for r in range(rows):
        row = []
        for c in range(columns):
            idx = r * columns + c
            if idx < len(figures):
                row.append({"secondary_y": _has_secondary_y(figures[idx])})
            else:
                row.append({})
        specs.append(row)
    return specs


def _copy_figure_to_subfigure(figure, subfigure, **pos):
    for trace in figure.data:
        is_secondary = getattr(trace, "yaxis", None) == "y2"
        subfigure.add_trace(
            _clean_trace(trace),
            secondary_y=is_secondary,
            **pos,
        )

    for annotation in figure.layout.annotations:
        subfigure.add_annotation(annotation.to_plotly_json(), **pos)
    for shape in figure.layout.shapes:
        subfigure.add_shape(shape.to_plotly_json(), **pos)
    if figure.layout.xaxis:
        subfigure.update_xaxes(_clean_axis(figure.layout.xaxis), **pos)

    subfigure.update_yaxes(_clean_axis(figure.layout.yaxis), **pos)

    if (secondary_y_axis := getattr(figure.layout, "yaxis2", None)):
        subfigure.update_yaxes(
            _clean_axis(secondary_y_axis),
            secondary_y=True,
            **pos,
        )


def figures_to_subplot(
    figures: list[go.Figure],
    rows: int,
    columns: int,
    **kwargs,
) -> go.Figure:

    specs = _build_specs(figures, rows, columns)

    make_subplots_kwargs = filter_allowed_kwargs(
        make_subplots,
        kwargs,
        disable_kwargs=True,
    )

    subplot = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=[figure.layout.title.text for figure in figures],
        specs=specs,
        **make_subplots_kwargs,
    )

    for index, figure in enumerate(figures):
        pos = {
            "row": index // columns + 1,
            "col": index % columns + 1,
        }

        _copy_figure_to_subfigure(figure, subplot, **pos)

    update_layout_kwargs = {
        k: v for k, v in kwargs.items() if k not in make_subplots_kwargs
    }
    subplot.update_layout(**update_layout_kwargs)

    return subplot


def prices_plot(
    price1: pd.Series,
    price2: pd.Series,
    formation_start: pd.Timestamp | None = None,
    formation_end: pd.Timestamp | None = None,
    trading_start: pd.Timestamp | None = None,
    trading_end: pd.Timestamp | None = None,
    **kwargs,
):

    name1 = price1.name
    name2 = price2.name
    index = price1.index
    color1 = {"color": PRICE1_COLOR}
    color2 = {"color": PRICE2_COLOR}

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=index,
            y=price1,
            name=name1,
            line=color1,
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                f"{name1}: %{{y:.2f}}"
                "<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=index,
            y=price2,
            name=name2,
            line=color2,
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                f"{name2}: %{{y:.2f}}"
                "<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    fig.update_layout(**kwargs)
    fig.update_yaxes(
        title_text=name1,
        title_font=color1,
        tickfont=color1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text=name2,
        title_font=color2,
        tickfont=color2,
        secondary_y=True,
    )

    if formation_start and formation_end:
        fig.add_vrect(
            x0=formation_start,
            x1=formation_end,
            fillcolor=FORMATION_BACKGROUND_COLOR,
            opacity=0.2,
            layer="below",
            line_width=0,
        )

    if trading_start and trading_end:
        fig.add_vrect(
            x0=trading_start,
            x1=trading_end,
            fillcolor=TRADING_BACKGROUND_COLOR,
            opacity=0.2,
            layer="below",
            line_width=0,
        )

    return fig


def scatter_plot(
    price1: pd.Series,
    price2: pd.Series,
    formation_start: pd.Timestamp,
    formation_end: pd.Timestamp,
    trading_start: pd.Timestamp,
    trading_end: pd.Timestamp,
    hedge_ratio: float,
    spread_mean: float,
    spread_std: float,
    long_entry: float = -2,
    long_exit: float = 0,
    long_stoploss: float = -5,
    short_entry: float = 2,
    short_exit: float = 0,
    short_stoploss: float = 5,
    **kwargs,
):
    color_segments = 6
    scatter_size = 5

    color1 = {"color": PRICE1_COLOR}
    color2 = {"color": PRICE2_COLOR}

    name1 = price1.name
    name2 = price2.name

    formation_price1 = price1.loc[formation_start:formation_end]
    formation_price2 = price2.loc[formation_start:formation_end]
    trading_price1 = price1.loc[trading_start:trading_end]
    trading_price2 = price2.loc[trading_start:trading_end]

    formation_index = formation_price1.index
    formation_color_vals = formation_index.view("int64")

    trading_index = trading_price1.index
    trading_color_vals = trading_index.view("int64")

    x1 = np.linspace(price2.min() * 0.98, price2.max() * 1.02, 100)
    y1 = spread_mean + hedge_ratio * x1

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x1,
            y=y1 + long_entry * spread_std,
            name="Long Entry",
            mode="lines",
            line=dict(
                color=LONG_COLOR,
                dash=ENTRY_LINE_STYLE,
            ),
            hovertemplate=(
                f"Long Entry (Spread {'-' if long_entry < 0 else '+'} {abs(long_entry)} Std)<br>"  # noqa: E501
                f"{name1}: %{{y:.2f}}<br>"
                f"{name2}: %{{x:.2f}}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x1,
            y=y1 + long_exit * spread_std,
            name="Long Exit",
            mode="lines",
            line=dict(
                color=LONG_COLOR,
                dash=EXIT_LINE_STYLE,
            ),
            hovertemplate=(
                f"Long Exit (Spread {'-' if long_exit < 0 else '+'} {abs(long_exit)} Std)<br>"  # noqa: E501
                f"{name1}: %{{y:.2f}}<br>"
                f"{name2}: %{{x:.2f}}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x1,
            y=y1 + long_stoploss * spread_std,
            name="Long Stoploss",
            mode="lines",
            line=dict(
                color=LONG_COLOR,
                dash=STOPLOSS_LINE_STYLE,
            ),
            hovertemplate=(
                f"Long Stoploss (Spread {'-' if long_stoploss < 0 else '+'} {abs(long_stoploss)} Std)<br>"  # noqa: E501
                f"{name1}: %{{y:.2f}}<br>"
                f"{name2}: %{{x:.2f}}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x1,
            y=y1 + short_entry * spread_std,
            name="Short Entry",
            mode="lines",
            line=dict(
                color=SHORT_COLOR,
                dash=ENTRY_LINE_STYLE,
            ),
            hovertemplate=(
                f"Short Entry (Spread {'-' if short_entry < 0 else '+'} {abs(short_entry)} Std)<br>"  # noqa: E501
                f"{name1}: %{{y:.2f}}<br>"
                f"{name2}: %{{x:.2f}}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x1,
            y=y1 + short_exit * spread_std,
            name="Short Exit",
            mode="lines",
            line=dict(
                color=SHORT_COLOR,
                dash=EXIT_LINE_STYLE,
            ),
            hovertemplate=(
                f"Short Exit (Spread {'-' if short_exit < 0 else '+'} {abs(short_exit)} Std)<br>"  # noqa: E501
                f"{name1}: %{{y:.2f}}<br>"
                f"{name2}: %{{x:.2f}}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x1,
            y=y1 + short_stoploss * spread_std,
            name="Short Stoploss",
            mode="lines",
            line=dict(
                color=SHORT_COLOR,
                dash=STOPLOSS_LINE_STYLE,
            ),
            hovertemplate=(
                f"Short Stoploss (Spread {'-' if short_stoploss < 0 else '+'} {abs(short_stoploss)} Std)<br>"  # noqa: E501
                f"{name1}: %{{y:.2f}}<br>"
                f"{name2}: %{{x:.2f}}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=formation_price2,
            y=formation_price1,
            mode="markers",
            name="Formation Period",
            showlegend=True,
            opacity=0.9,
            customdata=formation_index,
            hovertemplate=(
                "Formation Period<br>"
                "Date: %{customdata|%Y-%m-%d}<br>"
                f"{name1}: %{{y:.2f}}<br>"
                f"{name2}: %{{x:.2f}}"
                "<extra></extra>"
            ),
            marker=dict(
                size=scatter_size,
                color=formation_color_vals,
                colorscale=FORMATION_COLOR_SCALE,
                showscale=False,
                line=dict(color="black", width=0.5),
                colorbar=dict(
                    title="Formation Date",
                    x=1.02,
                    tickvals=np.linspace(
                        formation_color_vals.min(),
                        formation_color_vals.max(),
                        color_segments,
                    ),
                    ticktext=[
                        formation_index[loc].strftime(ISO8601_DATE_FORMAT)
                        for loc in
                        np.linspace(
                            0, len(formation_index) - 1, color_segments,
                        ).astype(int)
                    ]
                )
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=trading_price2,
            y=trading_price1,
            mode="markers",
            name="Trading Period",
            showlegend=True,
            opacity=0.9,
            customdata=trading_index,
            hovertemplate=(
                "Trading Period<br>"
                "Date: %{customdata|%Y-%m-%d}<br>"
                f"{name1}: %{{y:.2f}}<br>"
                f"{name2}: %{{x:.2f}}"
                "<extra></extra>"
            ),
            marker=dict(
                size=scatter_size,
                color=trading_color_vals,
                colorscale=TRADING_COLOR_SCALE,
                showscale=False,
                line=dict(color="black", width=0.5),
                colorbar=dict(
                    title="Trading Date",
                    x=1.15,
                    tickvals=np.linspace(
                        trading_color_vals.min(),
                        trading_color_vals.max(),
                        color_segments,
                    ),
                    ticktext=[
                        trading_index[loc].strftime(ISO8601_DATE_FORMAT)
                        for loc in np.linspace(
                            0, len(trading_index) - 1, color_segments,
                        ).astype(int)
                    ]
                )
            )
        )
    )

    price1_max = price1.max()
    price1_min = price1.min()
    yaxis_range = price1_max - price1_min
    default_kwargs = {
        "legend": dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        "yaxis": dict(
            range=[
                price1_min - 0.1 * yaxis_range,
                price1_max + 0.1 * yaxis_range,
            ],
        ),
    }

    kwargs = default_kwargs | kwargs

    fig.update_layout(**kwargs)
    fig.update_xaxes(
        title_text=name2,
        title_font=color2,
        tickfont=color2,
    )
    fig.update_yaxes(
        title_text=name1,
        title_font=color1,
        tickfont=color1,
    )

    return fig


def spread_plot(
    spread,
    formation_start,
    formation_end,
    trading_start,
    trading_end,
    spread_mean,
    spread_std,
    long_entry: float = -2,
    long_exit: float = 0,
    long_stoploss: float = -5,
    short_entry: float = 2,
    short_exit: float = 0,
    short_stoploss: float = 5,
    **kwargs,
):
    spread_len = len(spread)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=[spread_mean + long_entry * spread_std] * spread_len,
            mode="lines",
            name="Long Entry",
            line=dict(color=LONG_COLOR, dash=ENTRY_LINE_STYLE),
            hovertemplate=(
                "Long Entry<br>"
                "Date: %{x|%Y-%m-%d}<br>"
                f"Spread {'-' if long_entry < 0 else '+'} {abs(long_entry)} Std: %{{y:.2f}}"  # noqa: E501
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=[spread_mean + long_exit * spread_std] * spread_len,
            mode="lines",
            name="Long Exit",
            line=dict(color=LONG_COLOR, dash=EXIT_LINE_STYLE),
            hovertemplate=(
                "Long Exit<br>"
                "Date: %{x|%Y-%m-%d}<br>"
                f"Spread {'-' if long_exit < 0 else '+'} {abs(long_exit)} Std: %{{y:.2f}}"  # noqa: E501
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=[spread_mean + long_stoploss * spread_std] * spread_len,
            mode="lines",
            name="Long Stoploss",
            line=dict(color=LONG_COLOR, dash=STOPLOSS_LINE_STYLE),
            hovertemplate=(
                "Long Stoploss<br>"
                "Date: %{x|%Y-%m-%d}<br>"
                f"Spread {'-' if long_stoploss < 0 else '+'} {abs(long_stoploss)} Std: %{{y:.2f}}"  # noqa: E501
                "<extra></extra>"
            ),

        )
    )

    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=[spread_mean + short_entry * spread_std] * spread_len,
            mode="lines",
            name="Short Entry",
            line=dict(color=SHORT_COLOR, dash=ENTRY_LINE_STYLE),
            hovertemplate=(
                "Short Entry<br>"
                "Date: %{x|%Y-%m-%d}<br>"
                f"Spread {'-' if short_entry < 0 else '+'} {abs(short_entry)} Std: %{{y:.2f}}"  # noqa: E501
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=[spread_mean + short_exit * spread_std] * spread_len,
            mode="lines",
            name="Short Exit",
            line=dict(color=SHORT_COLOR, dash=EXIT_LINE_STYLE),
            hovertemplate=(
                "Short Exit<br>"
                "Date: %{x|%Y-%m-%d}<br>"
                f"Spread {'-' if short_exit < 0 else '+'} {abs(short_exit)} Std: %{{y:.2f}}"  # noqa: E501
                "<extra></extra>"
            ),

        )
    )
    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=[spread_mean + short_stoploss * spread_std] * spread_len,
            mode="lines",
            name="Short Stoploss",
            line=dict(color=SHORT_COLOR, dash=STOPLOSS_LINE_STYLE),
            hovertemplate=(
                "Short Stoploss<br>"
                "Date: %{x|%Y-%m-%d}<br>"
                f"Spread {'-' if short_stoploss < 0 else '+'} {abs(short_stoploss)} Std: %{{y:.2f}}"  # noqa: E501
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=spread.values,
            mode="lines",
            name="Spread",
            line=dict(color=SPREAD_COLOR, dash=SPREAD_LINE_STYLE),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Spread: %{y:.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vrect(
        x0=formation_start,
        x1=formation_end,
        fillcolor=FORMATION_BACKGROUND_COLOR,
        opacity=0.2,
        layer="below",
        line_width=0,
    )

    fig.add_vrect(
        x0=trading_start,
        x1=trading_end,
        fillcolor=TRADING_BACKGROUND_COLOR,
        opacity=0.2,
        layer="below",
        line_width=0,
    )

    fig.update_yaxes(
        title_text='Spread',
    )

    fig.update_layout(**kwargs)

    return fig


def equity_curve_plot(
    portfolio_value,
    formation_start,
    formation_end,
    trading_start,
    trading_end,
    **kwargs,
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=portfolio_value.index,
            y=portfolio_value.values,
            mode="lines",
            name="Portfolio",
            line=dict(color=EQUITY_CURVE_COLOR, dash=EQUITY_CURVE_LINE_STYLE),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Portfolio: %{y:.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vrect(
        x0=formation_start,
        x1=formation_end,
        fillcolor=FORMATION_BACKGROUND_COLOR,
        opacity=0.2,
        layer="below",
        line_width=0,
    )

    fig.add_vrect(
        x0=trading_start,
        x1=trading_end,
        fillcolor=TRADING_BACKGROUND_COLOR,
        opacity=0.2,
        layer="below",
        line_width=0,
    )

    fig.update_yaxes(
        title_text='Portfolio',
    )

    fig.update_layout(**kwargs)

    return fig
