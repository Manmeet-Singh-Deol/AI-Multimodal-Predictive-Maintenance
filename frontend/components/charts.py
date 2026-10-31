"""Reusable Plotly visualization components styled with an industrial dark theme.

Provides standardized plotting routines for telemetry time-series, operational gauges,
fault distribution bars, thermal heatmaps, and correlation scatter plots.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Theme constants
CHART_BG = "rgba(0,0,0,0)"
GRID_COLOR = "#222c3a"
TEXT_COLOR = "#e2e8f0"
ACCENT_PALETTE = [
    "#00d2ff",  # Cyan
    "#3b82f6",  # Blue
    "#10b981",  # Emerald Green
    "#f59e0b",  # Amber
    "#ef4444",  # Crimson Red
    "#8b5cf6",  # Violet
    "#ec4899",  # Pink
]


def _apply_dark_layout(
    fig: go.Figure,
    title: str,
    height: int = 380,
    show_legend: bool = True,
) -> go.Figure:
    """Apply consistent industrial dark mode styling to a Plotly figure."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "y": 0.95,
            "x": 0.02,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 15, "color": "#ffffff"},
        },
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font={"color": TEXT_COLOR, "family": "-apple-system, Roboto, sans-serif"},
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        height=height,
        showlegend=show_legend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "bgcolor": "rgba(15, 23, 42, 0.6)",
            "bordercolor": "rgba(255, 255, 255, 0.15)",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        hoverlabel={
            "bgcolor": "rgba(15, 23, 42, 0.88)",
            "bordercolor": "rgba(0, 210, 255, 0.5)",
            "font": {"color": "#ffffff", "size": 12, "family": "-apple-system, Roboto, sans-serif"},
        },
        hovermode="x unified",
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        tickfont={"size": 10, "color": "#94a3b8"},
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        tickfont={"size": 10, "color": "#94a3b8"},
    )
    return fig


def create_time_series_chart(
    df: pd.DataFrame,
    x_col: str,
    y_cols: Union[str, List[str]],
    title: str,
    unit: Optional[str] = None,
) -> go.Figure:
    """Create a multi-line time-series telemetry chart."""
    if isinstance(y_cols, str):
        y_cols = [y_cols]

    fig = go.Figure()
    for idx, col in enumerate(y_cols):
        if col in df.columns:
            color = ACCENT_PALETTE[idx % len(ACCENT_PALETTE)]
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[col],
                    name=col.replace("_", " ").title(),
                    mode="lines",
                    line={"color": color, "width": 2},
                    hovertemplate=f"%{{y:.2f}} {unit or ''}<extra></extra>",
                )
            )

    fig = _apply_dark_layout(fig, title)
    if unit:
        fig.update_yaxes(title_text=unit, title_font={"size": 11})
    return fig


def create_gauge_chart(
    value: float,
    title: str,
    min_val: float = 0.0,
    max_val: float = 100.0,
    thresholds: Optional[Dict[str, float]] = None,
) -> go.Figure:
    """Create an industrial gauge indicator chart (e.g. Health Score or RUL)."""
    if thresholds is None:
        thresholds = {"danger": 40.0, "warning": 70.0}

    # Determine needle bar color based on thresholds
    if value < thresholds.get("danger", 40.0):
        bar_color = "#ef4444"
    elif value < thresholds.get("warning", 70.0):
        bar_color = "#f59e0b"
    else:
        bar_color = "#10b981"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"<b>{title}</b>", "font": {"size": 15, "color": "#ffffff"}},
            number={"suffix": "%", "font": {"size": 32, "color": bar_color}},
            gauge={
                "axis": {
                    "range": [min_val, max_val],
                    "tickwidth": 1,
                    "tickcolor": "#64748b",
                    "tickfont": {"color": "#94a3b8", "size": 11},
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(30, 41, 59, 0.5)",
                "borderwidth": 1,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [min_val, thresholds.get("danger", 40.0)], "color": "rgba(239, 68, 68, 0.15)"},
                    {
                        "range": [thresholds.get("danger", 40.0), thresholds.get("warning", 70.0)],
                        "color": "rgba(245, 158, 11, 0.15)",
                    },
                    {"range": [thresholds.get("warning", 70.0), max_val], "color": "rgba(16, 185, 129, 0.15)"},
                ],
            },
        )
    )

    fig.update_layout(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font={"color": TEXT_COLOR, "family": "-apple-system, sans-serif"},
        margin={"l": 25, "r": 25, "t": 35, "b": 15},
        height=260,
    )
    return fig


def create_bar_chart(
    categories: List[str],
    values: List[float],
    title: str,
    colors: Optional[List[str]] = None,
    horizontal: bool = False,
) -> go.Figure:
    """Create a categorized bar chart for fault distributions or metrics."""
    if colors is None:
        colors = [ACCENT_PALETTE[i % len(ACCENT_PALETTE)] for i in range(len(categories))]

    fig = go.Figure()
    if horizontal:
        fig.add_trace(
            go.Bar(
                x=values,
                y=categories,
                orientation="h",
                marker={"color": colors, "line": {"width": 1, "color": "#1e293b"}},
                text=[f"{v:.1f}" if isinstance(v, float) else str(v) for v in values],
                textposition="auto",
            )
        )
    else:
        fig.add_trace(
            go.Bar(
                x=categories,
                y=values,
                marker={"color": colors, "line": {"width": 1, "color": "#1e293b"}},
                text=[f"{v:.1f}" if isinstance(v, float) else str(v) for v in values],
                textposition="auto",
            )
        )

    fig = _apply_dark_layout(fig, title, show_legend=False)
    return fig


def create_pie_chart(
    labels: List[str],
    values: List[float],
    title: str,
    hole: float = 0.45,
) -> go.Figure:
    """Create a donut or pie chart for proportional categorical metrics."""
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=hole,
                marker={"colors": ACCENT_PALETTE[: len(labels)], "line": {"color": "#0f172a", "width": 2}},
                textinfo="percent+label",
                insidetextorientation="radial",
                hoverinfo="label+value+percent",
            )
        ]
    )
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "y": 0.95,
            "x": 0.05,
            "font": {"size": 15, "color": "#ffffff"},
        },
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font={"color": TEXT_COLOR, "family": "-apple-system, sans-serif"},
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
        height=320,
        showlegend=False,
    )
    return fig


def create_heatmap(
    data: Union[List[List[float]], np.ndarray],
    x_labels: List[str],
    y_labels: List[str],
    title: str,
) -> go.Figure:
    """Create a correlation or telemetry spatial heatmap."""
    fig = go.Figure(
        data=go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            colorscale="Viridis",
            showscale=True,
            colorbar={"tickfont": {"color": "#94a3b8"}, "outlinecolor": "#334155"},
        )
    )
    fig = _apply_dark_layout(fig, title, show_legend=False)
    return fig


def create_scatter_plot(
    x: List[float],
    y: List[float],
    title: str,
    color_by: Optional[List[Any]] = None,
    x_label: str = "X",
    y_label: str = "Y",
) -> go.Figure:
    """Create a scatter plot for cross-sensor correlation and clustering analysis."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker={
                "size": 8,
                "color": color_by if color_by is not None else "#00d2ff",
                "colorscale": "Plasma" if color_by is not None else None,
                "showscale": color_by is not None,
                "line": {"color": "#ffffff", "width": 0.5},
            },
        )
    )
    fig = _apply_dark_layout(fig, title, show_legend=False)
    fig.update_xaxes(title_text=x_label, title_font={"size": 11})
    fig.update_yaxes(title_text=y_label, title_font={"size": 11})
    return fig
