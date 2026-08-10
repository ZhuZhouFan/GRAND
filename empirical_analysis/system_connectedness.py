"""Plot system-level connectedness against the smoothed market index.

Computes the volatility-based ``DGC_{t,sigma}`` and quantile-based
``DGC_{t,VaR}`` time series from saved adjacency matrices, smooths them
together with the SSEC weekly close via a trailing moving average, and
renders a 3-row x 3-column figure spanning three out-of-sample periods
with annotated event markers A-O.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, MaxNLocator

sys.path.append('.')
from config import project_path

Q_PATH = f'{project_path}/Q_graph.npy'
V_PATH = f'{project_path}/V_graph.npy'
INDEX_PATH = f'{project_path}/kline_week_index/000001.XSHG.csv'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
MA_WINDOW = 4

PERIODS = [
    ('2019-01-01', '2019-09-01'),
    ('2020-01-01', '2021-01-15'),
    ('2021-11-01', '2022-06-01'),
]

EVENTS = {
    'A': '2019-01-31',
    'B': '2019-04-19',
    'C': '2019-05-06',
    'D': '2019-05-24',
    'E': '2019-06-29',
    'F': '2020-01-23',
    'G': '2020-03-11',
    'H': '2020-04-20',
    'I': '2020-10-29',
    'J': '2021-01-08',
    'K': '2021-11-03',
    'L': '2022-02-24',
    'M': '2022-03-25',
    'N': '2022-04-22',
    'O': '2022-05-13',
}

ROWS = [
    {
        'col': 'market',
        'color': '#C62828',
        'ylabel': 'Market index',
        'decimals': 0,
    },
    {
        'col': 'DGC_sigma',
        'color': '#555555',
        'ylabel': r'$\mathrm{DGC}_{t,\sigma}$ (%)',
        'decimals': 2,
    },
    {
        'col': 'DGC_VaR',
        'color': '#111111',
        'ylabel': r'$\mathrm{DGC}_{t,\mathrm{VaR}}$ (%)',
        'decimals': 2,
    },
]

def load_adjacency(path):
    return np.load(path, allow_pickle=True).item()


def compute_dgc(adj_dict):
    """``DGC_t = #{(i, j) : i != j, A_t^{(i,j)} != 0} / (N (N - 1))``."""
    records = {}
    for date, mat in adj_dict.items():
        n = mat.shape[0]
        off = mat.copy()
        np.fill_diagonal(off, 0)
        records[date] = float(np.count_nonzero(off)) / (n * (n - 1))
    series = pd.Series(records, name='DGC')
    series.index = pd.to_datetime(series.index)
    series = series.sort_index().shift(-1).dropna()
    return series


def load_market_index(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Market-index file not found: {path}. Update INDEX_PATH at the '
            f'top of this script to point to your weekly market-index CSV.')
    df = pd.read_csv(path, parse_dates=['date'])
    return df.set_index('date')['close'].sort_index()


def smooth(series, window=MA_WINDOW):
    return series.rolling(window=window, min_periods=1).mean()

def nearest_index_date(target, index):
    diffs = np.abs(index - target)
    return index[int(diffs.argmin())]


def events_in_window(start, end):
    out = []
    for label, iso in EVENTS.items():
        d = pd.Timestamp(iso)
        if start <= d <= end:
            out.append((label, d))
    return out


def stagger_levels(events_in_panel, min_separation_days=28):
    """Assign each event the lowest annotation level that keeps it clear of
    neighbours on the same level."""
    sorted_evts = sorted(events_in_panel, key=lambda x: x[1])
    last_date_per_level = []
    result = []
    for label, d in sorted_evts:
        assigned = None
        for level, last_d in enumerate(last_date_per_level):
            if (d - last_d).days >= min_separation_days:
                last_date_per_level[level] = d
                assigned = level
                break
        if assigned is None:
            last_date_per_level.append(d)
            assigned = len(last_date_per_level) - 1
        result.append((label, d, assigned))
    return result


def annotate_events(ax, series, color, staggered):
    base_offset_pts = 16
    level_step_pts = 14
    for label, ev_date, level in staggered:
        d = nearest_index_date(ev_date, series.index)
        gap = abs((d - ev_date).days)
        if gap > 7:
            warnings.warn(
                f'Event {label} ({ev_date.date()}) is {gap} days from the '
                f'nearest weekly observation ({d.date()}).')
        y = float(series.loc[d])
        ax.plot([d], [y], 'o',
                color=color, markersize=4.5,
                markeredgecolor='white', markeredgewidth=0.8, zorder=5)
        ax.annotate(
            # f'Point {label}',
            f'{label}',
            xy=(d, y),
            xytext=(0, base_offset_pts + level_step_pts * level),
            textcoords='offset points',
            ha='center', va='bottom',
            fontsize=10,
            arrowprops=dict(arrowstyle='-', color='gray',
                            lw=0.7, shrinkA=0, shrinkB=2),
            zorder=4,
        )

def render_figure(smooth_df, periods, events):
    n_rows = len(ROWS)
    n_cols = len(periods)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(15, 10), constrained_layout=True)

    for c, (start_str, end_str) in enumerate(periods):
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str)
        title = (f'Period {("I", "II", "III")[c]}: '
                 f'{start.strftime("%b %Y")} - {end.strftime("%b %Y")}')
        axes[0, c].set_title(title, fontsize=13, pad=10)

    for r, row in enumerate(ROWS):
        col = row['col']
        color = row['color']
        decimals = row.get('decimals', 2)
        y_formatter = FuncFormatter(
            lambda x, _pos, d=decimals: f'{x:.{d}f}')

        # First pass: gather per-column windows / event staggers and aggregate
        # statistics so the row can share a common y-axis range.
        panels = []
        y_min_row, y_max_row = np.inf, -np.inf
        max_levels = 0
        for c, (start_str, end_str) in enumerate(periods):
            start = pd.Timestamp(start_str)
            end = pd.Timestamp(end_str)
            window = smooth_df.loc[start:end, col].dropna()
            staggered = stagger_levels(events_in_window(start, end))
            n_levels = (max((lv for _, _, lv in staggered), default=-1) + 1
                        if staggered else 0)
            panels.append((c, start, end, window, staggered))
            if not window.empty:
                y_min_row = min(y_min_row, float(window.min()))
                y_max_row = max(y_max_row, float(window.max()))
            max_levels = max(max_levels, n_levels)

        y_range = y_max_row - y_min_row
        if y_range == 0 or not np.isfinite(y_range):
            y_range = max(abs(y_max_row), 1.0)
        headroom = 0.08 + 0.08 * max_levels
        ylim = (y_min_row - 0.03 * y_range,
                y_max_row + headroom * y_range)

        for c, start, end, window, staggered in panels:
            ax = axes[r, c]

            ax.plot(window.index, window.values,
                    color=color, lw=1.9, solid_capstyle='round')

            ax.set_ylim(*ylim)

            annotate_events(ax, window, color, staggered)

            ax.set_xlim(start, end)
            ax.xaxis.set_major_locator(
                mdates.AutoDateLocator(minticks=3, maxticks=4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.yaxis.set_major_locator(
                MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
            ax.yaxis.set_major_formatter(y_formatter)
            ax.tick_params(axis='x', labelsize=11)
            ax.tick_params(axis='y', labelsize=11)

            ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.7)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)

            if c == 0:
                ax.set_ylabel(row['ylabel'], fontsize=14, labelpad=8)

    return fig

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    q = load_adjacency(Q_PATH)
    v = load_adjacency(V_PATH)
    dgc_var = compute_dgc(q) * 100.0
    dgc_sigma = compute_dgc(v) * 100.0

    market = load_market_index(INDEX_PATH)

    raw = pd.DataFrame({
        'market':    market,
        'DGC_sigma': dgc_sigma,
        'DGC_VaR':   dgc_var,
    }).sort_index()
    smooth_df = raw.apply(lambda s: smooth(s, MA_WINDOW))

    fig = render_figure(smooth_df, PERIODS, EVENTS)

    png = os.path.join(OUT_DIR, 'system_level_connectedness.png')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved: {png}')


if __name__ == '__main__':
    main()
