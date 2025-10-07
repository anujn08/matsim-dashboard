import os
import math
import glob
from typing import List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.dirname(__file__)
AROOT = os.path.join(BASE_DIR, 'analysis_run_outputs')
OUTDIR = os.path.join(BASE_DIR, 'analysis_results')
os.makedirs(OUTDIR, exist_ok=True)


def ci95(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors='coerce').dropna()
    if len(s) == 0:
        return float('nan')
    return 1.96 * s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else 0.0


def load_csvs(pattern: str, add_cols: List[Tuple[str, str]] = None) -> pd.DataFrame:
    files = glob.glob(pattern)
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if add_cols:
                for col_name, col_value in add_cols:
                    df[col_name] = col_value
            dfs.append(df)
        except Exception as e:
            print(f"WARN: Failed reading {f}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def aggregate_numeric(df: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude identifiers/iterations if present from aggregation by default
    exclude = {'iteration'}
    num_cols = [c for c in num_cols if c not in exclude]

    def agg_ci(x):
        return pd.Series({
            'mean': x.mean(),
            'std': x.std(ddof=1),
            'min': x.min(),
            'max': x.max(),
            'n': x.count(),
            'ci95': (1.96 * x.std(ddof=1) / math.sqrt(x.count())) if x.count() > 1 else 0.0,
        })

    pieces = []
    for col in num_cols:
        tmp = df.groupby(group_by)[col].apply(lambda s: agg_ci(pd.to_numeric(s, errors='coerce')))
        tmp = tmp.unstack(-1)
        tmp.columns = [f"{col}_{stat}" for stat in tmp.columns]
        pieces.append(tmp)
    out = pd.concat(pieces, axis=1).reset_index()
    return out


def main():
    print('Scanning inputs under:', AROOT)

    # 1) Overall (all_days) summary across runs
    overall_path = os.path.join(AROOT, 'all_days', 'analysis_summary.csv')
    overall_df = pd.read_csv(overall_path) if os.path.exists(overall_path) else pd.DataFrame()
    if not overall_df.empty:
        overall_df['scope'] = 'all_days'
        overall_df.to_csv(os.path.join(OUTDIR, 'overall_all_runs_raw.csv'), index=False)
        overall_agg = aggregate_numeric(overall_df, group_by=['scope'])
        overall_agg.to_csv(os.path.join(OUTDIR, 'overall_all_runs_aggregates.csv'), index=False)
        print('Saved overall aggregates:', os.path.join(OUTDIR, 'overall_all_runs_aggregates.csv'))
    else:
        print('No overall all_days summary found.')

    # 2) Per-day summaries across runs
    day_dirs = [d for d in glob.glob(os.path.join(AROOT, 'day*')) if os.path.isdir(d)]
    day_rows = []
    for d in sorted(day_dirs):
        day_name = os.path.basename(d)
        f = os.path.join(d, 'analysis_summary.csv')
        if os.path.exists(f):
            try:
                tmp = pd.read_csv(f)
                tmp['day'] = day_name
                day_rows.append(tmp)
            except Exception as e:
                print(f"WARN: failed reading {f}: {e}")
    daily_df = pd.concat(day_rows, ignore_index=True) if day_rows else pd.DataFrame()
    if not daily_df.empty:
        daily_df.to_csv(os.path.join(OUTDIR, 'daily_all_runs_raw.csv'), index=False)
        daily_agg = aggregate_numeric(daily_df, group_by=['day'])
        daily_agg.to_csv(os.path.join(OUTDIR, 'daily_all_runs_aggregates.csv'), index=False)
        print('Saved daily aggregates:', os.path.join(OUTDIR, 'daily_all_runs_aggregates.csv'))

        # Plot: daily mean average_travel_time_s with CI
        if 'average_travel_time_s' in daily_df.columns:
            plot_df = daily_df.copy()
            plot_df['average_travel_time_s'] = pd.to_numeric(plot_df['average_travel_time_s'], errors='coerce')
            plot_df = plot_df.dropna(subset=['average_travel_time_s'])
            plt.figure(figsize=(10, 5))
            sns.barplot(data=plot_df, x='day', y='average_travel_time_s', estimator=np.mean, ci=95, order=sorted(plot_df['day'].unique()))
            plt.xticks(rotation=45, ha='right')
            plt.title('Daily mean average_travel_time_s across runs')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTDIR, 'daily_avg_travel_time_mean.png'), dpi=150)
            plt.close()

        # Plot: distribution of avg leg travel time per day
        if 'average_leg_travel_time_s' in daily_df.columns:
            plot_df = daily_df.copy()
            plot_df['average_leg_travel_time_s'] = pd.to_numeric(plot_df['average_leg_travel_time_s'], errors='coerce')
            plot_df = plot_df.dropna(subset=['average_leg_travel_time_s'])
            plt.figure(figsize=(10, 5))
            sns.boxplot(data=plot_df, x='day', y='average_leg_travel_time_s', order=sorted(plot_df['day'].unique()))
            plt.xticks(rotation=45, ha='right')
            plt.title('Daily distribution of average_leg_travel_time_s across runs')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTDIR, 'daily_avg_leg_travel_time_box.png'), dpi=150)
            plt.close()

    else:
        print('No per-day analysis_summary.csv files found.')

    # 3) Dham activity aggregation across runs and days (if present per day)
    dham_rows = []
    for d in sorted(day_dirs):
        day_name = os.path.basename(d)
        f = os.path.join(d, 'dham_activity_analysis.csv')
        if os.path.exists(f):
            try:
                tmp = pd.read_csv(f)
                tmp['day'] = day_name
                dham_rows.append(tmp)
            except Exception as e:
                print(f"WARN: failed reading {f}: {e}")
    dham_df = pd.concat(dham_rows, ignore_index=True) if dham_rows else pd.DataFrame()
    if not dham_df.empty:
        dham_df.to_csv(os.path.join(OUTDIR, 'dham_activity_raw.csv'), index=False)
        # Aggregate unique persons by day and dham_activity_type
        dham_agg = dham_df.groupby(['day', 'dham_activity_type'])['unique_dham_persons'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
        # 95% CI
        dham_agg['ci95'] = 1.96 * (dham_agg['std'] / np.sqrt(dham_agg['count'].clip(lower=1)))
        dham_agg.to_csv(os.path.join(OUTDIR, 'dham_activity_aggregates.csv'), index=False)

        # Plot per-dham trend
        plt.figure(figsize=(11, 6))
        order = sorted(dham_df['day'].unique())
        sns.lineplot(data=dham_df, x='day', y='unique_dham_persons', hue='dham_activity_type', estimator='mean', ci=95, sort=False)
        plt.xticks(rotation=45, ha='right')
        plt.title('Mean unique_dham_persons per day by dham')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, 'dham_activity_trend.png'), dpi=150)
        plt.close()
    else:
        print('No dham_activity_analysis.csv files found.')

    # 4) Facility rest aggregation
    rest_rows = []
    for d in sorted(day_dirs):
        day_name = os.path.basename(d)
        f = os.path.join(d, 'facility_rest_analysis.csv')
        if os.path.exists(f):
            try:
                tmp = pd.read_csv(f)
                tmp['day'] = day_name
                rest_rows.append(tmp)
            except Exception as e:
                print(f"WARN: failed reading {f}: {e}")
    rest_df = pd.concat(rest_rows, ignore_index=True) if rest_rows else pd.DataFrame()
    if not rest_df.empty:
        rest_df.to_csv(os.path.join(OUTDIR, 'facility_rest_raw.csv'), index=False)
        rest_agg = rest_df.groupby(['day', 'facility_id'])['unique_rest_persons'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
        rest_agg['ci95'] = 1.96 * (rest_agg['std'] / np.sqrt(rest_agg['count'].clip(lower=1)))
        rest_agg.to_csv(os.path.join(OUTDIR, 'facility_rest_aggregates.csv'), index=False)

        # Top facilities by mean rest per day (overall mean)
        top_fac_overall = rest_agg.groupby('facility_id')['mean'].mean().sort_values(ascending=False).head(15).index.tolist()
        plt.figure(figsize=(12, 6))
        top_df = rest_df[rest_df['facility_id'].isin(top_fac_overall)].copy()
        sns.barplot(data=top_df, x='facility_id', y='unique_rest_persons', estimator=np.mean, ci=95, order=top_fac_overall)
        plt.xticks(rotation=60, ha='right')
        plt.title('Top facilities by mean unique_rest_persons (across runs, averaged per day)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, 'facility_rest_top_facilities.png'), dpi=150)
        plt.close()
    else:
        print('No facility_rest_analysis.csv files found.')

    # 5) Simple HTML summary
    html_lines = [
        '<html><head><meta charset="utf-8"><title>MATSim Analysis Summary</title></head><body>',
        '<h1>MATSim Analysis Summary</h1>',
        f'<p>Root: {AROOT}</p>',
        '<ul>',
    ]
    for fn in [
        'overall_all_runs_aggregates.csv',
        'daily_all_runs_aggregates.csv',
        'dham_activity_aggregates.csv',
        'facility_rest_aggregates.csv',
        'daily_avg_travel_time_mean.png',
        'daily_avg_leg_travel_time_box.png',
        'dham_activity_trend.png',
        'facility_rest_top_facilities.png',
    ]:
        fpath = os.path.join(OUTDIR, fn)
        if os.path.exists(fpath):
            if fn.endswith('.png'):
                html_lines.append(f'<li><b>Figure:</b> {fn}<br><img src="{fn}" style="max-width: 900px;"></li>')
            else:
                html_lines.append(f'<li><b>Data:</b> <a href="{fn}">{fn}</a></li>')
    html_lines.append('</ul></body></html>')

    with open(os.path.join(OUTDIR, 'summary.html'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_lines))

    print('Done. See outputs in:', OUTDIR)


if __name__ == '__main__':
    main()
