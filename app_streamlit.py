import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, 'analysis_results')
RUN_OUT_DIR = os.path.join(BASE_DIR, 'analysis_run_outputs')


def sort_day_labels(vals):
    """Return a list of day labels sorted by their numeric suffix.
    Accepts labels like 'day1', 'day02', 'day10'. Falls back to string sort if no numbers.
    """
    import re
    def key_fn(v):
        if v is None:
            return float('inf')
        m = re.search(r"(\d+)", str(v))
        return int(m.group(1)) if m else float('inf')
    return sorted(vals, key=key_fn)


@st.cache_data(show_spinner=False)
def load_datasets() -> Dict[str, pd.DataFrame]:
    data = {}
    # Aggregates produced by analyze_matsim.py
    files = {
        'Overall aggregates (all days)': os.path.join(RESULTS_DIR, 'overall_all_runs_aggregates.csv'),
        'Daily aggregates': os.path.join(RESULTS_DIR, 'daily_all_runs_aggregates.csv'),
        'Dham activity aggregates': os.path.join(RESULTS_DIR, 'dham_activity_aggregates.csv'),
        'Facility rest aggregates': os.path.join(RESULTS_DIR, 'facility_rest_aggregates.csv'),
        'Daily raw (merged)': os.path.join(RESULTS_DIR, 'daily_all_runs_raw.csv'),
    }
    for name, path in files.items():
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                data[name] = df
            except Exception as e:
                st.warning(f'Failed to read {name}: {e}')

    # Fallback: directly read a couple of raw per-day CSVs if results missing
    if os.path.isdir(RUN_OUT_DIR):
        # Daily raw (merged)
        if 'Daily raw (merged)' not in data:
            rows = []
            for d in sorted(x for x in os.listdir(RUN_OUT_DIR) if x.startswith('day')):
                f = os.path.join(RUN_OUT_DIR, d, 'analysis_summary.csv')
                if os.path.exists(f):
                    try:
                        tmp = pd.read_csv(f)
                        tmp['day'] = d
                        rows.append(tmp)
                    except Exception:
                        pass
            if rows:
                data['Daily raw (merged)'] = pd.concat(rows, ignore_index=True)

        # All-days raw summary
        all_days_raw = os.path.join(RUN_OUT_DIR, 'all_days', 'analysis_summary.csv')
        if os.path.exists(all_days_raw):
            try:
                data['All-days raw summary'] = pd.read_csv(all_days_raw)
            except Exception:
                pass

        # Dham activity raw (merged)
        dham_rows = []
        for d in sorted(x for x in os.listdir(RUN_OUT_DIR) if x.startswith('day')):
            f = os.path.join(RUN_OUT_DIR, d, 'dham_activity_analysis.csv')
            if os.path.exists(f):
                try:
                    tmp = pd.read_csv(f)
                    tmp['day'] = d
                    dham_rows.append(tmp)
                except Exception:
                    pass
        if dham_rows:
            data['Dham activity raw (merged)'] = pd.concat(dham_rows, ignore_index=True)

        # Facility rest raw (merged)
        rest_rows = []
        for d in sorted(x for x in os.listdir(RUN_OUT_DIR) if x.startswith('day')):
            f = os.path.join(RUN_OUT_DIR, d, 'facility_rest_analysis.csv')
            if os.path.exists(f):
                try:
                    tmp = pd.read_csv(f)
                    tmp['day'] = d
                    rest_rows.append(tmp)
                except Exception:
                    pass
        if rest_rows:
            data['Facility rest raw (merged)'] = pd.concat(rest_rows, ignore_index=True)
    return data


def select_columns_ui(df: pd.DataFrame):
    st.subheader('Column selection')
    cols = df.columns.tolist()

    # Attempt to guess default x column
    default_x = None
    for candidate in ['day', 'iteration', 'scope']:
        if candidate in cols:
            default_x = candidate
            break

    x_col = st.selectbox('X-axis (categorical/time)', options=['<none>'] + cols, index=(cols.index(default_x) + 1) if default_x in cols else 0)

    # Numeric columns only for Y
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    # If none detected, try convert object to numeric soft
    if not numeric_cols:
        for c in cols:
            try:
                pd.to_numeric(df[c])
                numeric_cols.append(c)
            except Exception:
                pass
        numeric_cols = list(dict.fromkeys(numeric_cols))

    y_cols = st.multiselect('Y-axis metric(s)', options=numeric_cols, default=numeric_cols[:1])
    return (None if x_col == '<none>' else x_col), y_cols


def filter_ui(df: pd.DataFrame) -> pd.DataFrame:
    with st.expander('Filters', expanded=False):
        filtered = df.copy()
        # Provide simple equality filters for a few common columns if present
        for col in ['day', 'run_id', 'dham_activity_type', 'facility_id']:
            if col in filtered.columns:
                values = sorted(filtered[col].dropna().unique().tolist())
                if values and len(values) <= 1000:
                    selection = st.multiselect(f'{col} filter', values)
                    if selection:
                        filtered = filtered[filtered[col].isin(selection)]
        return filtered


def plot_ui(df: pd.DataFrame, x_col: Optional[str], y_cols):
    st.subheader('Plot')
    chart_type = st.selectbox('Chart type', ['line', 'bar', 'scatter', 'box', 'histogram'])
    color_col = st.selectbox('Color (grouping)', options=['<none>'] + df.columns.tolist())
    color = None if color_col == '<none>' else color_col

    if not y_cols:
        st.info('Select at least one Y-axis metric to plot.')
        return

    # For multiple y, melt to long format for consistent plotting
    long_df = df.copy()
    if len(y_cols) > 1:
        id_vars = [c for c in df.columns if c not in y_cols]
        long_df = df.melt(id_vars=id_vars, value_vars=y_cols, var_name='metric', value_name='value')
        y = 'value'
        if color is None:
            color = 'metric'
        elif color != 'metric':
            # combine color with metric for distinct traces
            long_df['metric_group'] = long_df['metric'].astype(str)
            color = 'metric_group' if color == 'metric' else color
    else:
        y = y_cols[0]

    # Build chart
    if chart_type == 'line':
        fig = px.line(long_df, x=x_col, y=y, color=color)
    elif chart_type == 'bar':
        fig = px.bar(long_df, x=x_col, y=y, color=color, barmode='group')
    elif chart_type == 'scatter':
        fig = px.scatter(long_df, x=x_col, y=y, color=color)
    elif chart_type == 'box':
        fig = px.box(long_df, x=x_col, y=y, color=color)
    else:
        # histogram ignores x if not specified
        if x_col and y == 'value':
            fig = px.histogram(long_df, x=y, color=color)
        else:
            fig = px.histogram(long_df, x=(y if x_col is None else x_col), color=color)

    fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title='MATSim Dashboard', layout='wide')
    st.title('MATSim Results Dashboard')
    st.caption(f'Root: {RUN_OUT_DIR}')

    # Guidance
    if not os.path.exists(RESULTS_DIR):
        st.info('Tip: run analyze_matsim.py first to generate aggregated CSVs and figures into analysis_results/.')

    data = load_datasets()
    if not data:
        st.error('No datasets found. Please ensure CSVs exist in analysis_run_outputs/.')
        return

    tabs = st.tabs([
        'Overview', 'Run Comparison (All-Days)', 'Daily Progression', 'Dham Activity', 'Facility Rest', 'Build Your Own'
    ])

    def get_df(name: str) -> Optional[pd.DataFrame]:
        return data.get(name).copy() if name in data else None

    # Overview
    with tabs[0]:
        st.subheader('Overview KPIs by run (all-days)')
        ad = get_df('All-days raw summary')
        if ad is None:
            ad = get_df('Overall aggregates (all days)')
        if ad is None:
            st.info('All-days summary not found. Place analysis_summary.csv under analysis_run_outputs/all_days/.')
        else:
            kpi_cols = [
                'average_travel_time_s', 'average_leg_travel_time_s', 'total_legs',
                'percentage_legs_car', 'percentage_legs_bus', 'percentage_legs_walk'
            ]
            present = [c for c in kpi_cols if c in ad.columns]
            if 'run_id' in ad.columns and present:
                show = ad[['run_id'] + present].copy()
                st.dataframe(show)
                if 'average_travel_time_s' in present:
                    fig = px.bar(show, x='run_id', y='average_travel_time_s', title='Average travel time by run', height=420)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('Expected columns not found in all-days summary.')

    # Run Comparison (All-Days)
    with tabs[1]:
        st.subheader('Compare runs on all-days summary')
        ad = get_df('All-days raw summary')
        if ad is None:
            st.info('All-days summary not found.')
        else:
            runs = sorted(ad['run_id'].unique()) if 'run_id' in ad.columns else []
            sel_runs = st.multiselect('Select runs', runs, default=runs[:10], key='runs_all_days') if runs else []
            ad_f = ad[ad['run_id'].isin(sel_runs)] if sel_runs else ad
            numeric_cols = [c for c in ad_f.columns if pd.api.types.is_numeric_dtype(ad_f[c])]
            if numeric_cols:
                default_idx = numeric_cols.index('average_travel_time_s') if 'average_travel_time_s' in numeric_cols else 0
                metric = st.selectbox('Metric', numeric_cols, index=default_idx, key='metric_all_days')
                fig = px.bar(ad_f, x='run_id', y=metric, title=f'{metric} by run', height=460)
                st.plotly_chart(fig, use_container_width=True)

            modal_cols = [c for c in ad.columns if c.startswith('percentage_legs_')]
            if modal_cols and 'run_id' in ad.columns:
                modal_df = ad_f[['run_id'] + modal_cols].melt(id_vars='run_id', var_name='mode', value_name='pct')
                fig2 = px.bar(modal_df, x='run_id', y='pct', color='mode', barmode='stack', title='Modal split (% legs)', height=460)
                st.plotly_chart(fig2, use_container_width=True)

    # Daily Progression
    with tabs[2]:
        st.subheader('Metrics over days')
        daily = get_df('Daily raw (merged)')
        if daily is None:
            st.info('Daily merged data not found.')
        else:
            if 'day' in daily.columns:
                order = sort_day_labels(daily['day'].unique().tolist())
                daily['day'] = pd.Categorical(daily['day'], categories=order, ordered=True)
            runs = sorted(daily['run_id'].unique()) if 'run_id' in daily.columns else []
            sel_runs = st.multiselect('Runs', runs, default=runs[:5], key='runs_daily') if runs else []
            daily_f = daily[daily['run_id'].isin(sel_runs)] if sel_runs else daily
            numeric_cols = [c for c in daily_f.columns if pd.api.types.is_numeric_dtype(daily_f[c])]
            if numeric_cols:
                default_idx = numeric_cols.index('average_travel_time_s') if 'average_travel_time_s' in numeric_cols else 0
                metric = st.selectbox('Metric (daily)', numeric_cols, index=default_idx, key='metric_daily')
                color = 'run_id' if 'run_id' in daily_f.columns else None
                fig = px.line(
                    daily_f,
                    x=('day' if 'day' in daily_f.columns else None),
                    y=metric,
                    color=color,
                    markers=True,
                    title=f'{metric} over days',
                    height=480,
                    category_orders={'day': order} if 'day' in daily_f.columns else None,
                )
                st.plotly_chart(fig, use_container_width=True)

                if 'day' in daily_f.columns:
                    figb = px.box(
                        daily_f, x='day', y=metric, points=False,
                        title=f'Distribution of {metric} per day (all runs)',
                        category_orders={'day': order}
                    )
                    st.plotly_chart(figb, use_container_width=True)

    # Dham Activity
    with tabs[3]:
        st.subheader('Dham activity across runs and days')
        dham = get_df('Dham activity raw (merged)')
        if dham is None:
            dham = get_df('Dham activity aggregates')
        if dham is None:
            st.info('No dham activity files found.')
        else:
            if 'run_id' in dham.columns:
                sel_runs = st.multiselect('Runs', sorted(dham['run_id'].unique()), default=None, key='runs_dham')
                if sel_runs:
                    dham = dham[dham['run_id'].isin(sel_runs)]
            if 'dham_activity_type' in dham.columns:
                sel_dhams = st.multiselect('Dham types', sorted(dham['dham_activity_type'].unique()), default=None, key='dham_types')
                if sel_dhams:
                    dham = dham[dham['dham_activity_type'].isin(sel_dhams)]

            ycol = 'unique_dham_persons' if 'unique_dham_persons' in dham.columns else ('mean' if 'mean' in dham.columns else None)
            if ycol is not None:
                x = 'day' if 'day' in dham.columns else 'run_id'
                color = 'dham_activity_type' if 'dham_activity_type' in dham.columns else None
                category_orders = None
                if x == 'day':
                    day_order = sort_day_labels(dham['day'].unique().tolist())
                    dham['day'] = pd.Categorical(dham['day'], categories=day_order, ordered=True)
                    category_orders = {'day': day_order}
                fig = px.line(dham, x=x, y=ycol, color=color, markers=True, title=f'{ycol} by {x}', category_orders=category_orders)
                st.plotly_chart(fig, use_container_width=True)

            if ycol is not None and {'run_id','dham_activity_type'}.issubset(dham.columns):
                pivot = dham.copy()
                if 'day' in pivot.columns:
                    pivot = pivot.groupby(['run_id', 'dham_activity_type'])[ycol].mean().reset_index()
                ptab = pivot.pivot(index='dham_activity_type', columns='run_id', values=ycol)
                figh = px.imshow(ptab, color_continuous_scale='Blues', title='Dham persons (mean) per run')
                st.plotly_chart(figh, use_container_width=True)

    # Facility Rest
    with tabs[4]:
        st.subheader('Facility rest across runs and days')
        rest = get_df('Facility rest raw (merged)')
        if rest is None:
            rest = get_df('Facility rest aggregates')
        if rest is None:
            st.info('No facility rest files found.')
        else:
            if 'run_id' in rest.columns:
                sel_runs = st.multiselect('Runs', sorted(rest['run_id'].unique()), default=None, key='runs_rest')
                if sel_runs:
                    rest = rest[rest['run_id'].isin(sel_runs)]
            if 'facility_id' in rest.columns:
                # show only first 50 by default for usability
                all_fac = sorted(rest['facility_id'].unique())
                sel_fac = st.multiselect('Facilities', all_fac[:50], help='Selecting none shows all (first 50 listed).', key='facilities_rest')
                if sel_fac:
                    rest = rest[rest['facility_id'].isin(sel_fac)]

            ycol = 'unique_rest_persons' if 'unique_rest_persons' in rest.columns else ('mean' if 'mean' in rest.columns else None)
            if ycol is not None:
                if 'facility_id' in rest.columns:
                    top = rest.groupby('facility_id')[ycol].mean().sort_values(ascending=False).head(20).index
                    rest_top = rest[rest['facility_id'].isin(top)].copy()
                    x = 'facility_id'
                    color = 'run_id' if 'run_id' in rest_top.columns else None
                    fig = px.bar(rest_top, x=x, y=ycol, color=color, barmode='group', title='Top facilities by mean rest persons')
                    st.plotly_chart(fig, use_container_width=True)

                if {'day','facility_id'}.issubset(rest.columns):
                    pivot = rest.groupby(['day','facility_id'])[ycol].mean().reset_index()
                    # order day columns naturally
                    day_order = sort_day_labels(pivot['day'].unique().tolist())
                    ptab = pivot.pivot(index='facility_id', columns='day', values=ycol)
                    ptab = ptab.reindex(columns=day_order)
                    figh = px.imshow(ptab, color_continuous_scale='OrRd', title='Facility rest heatmap (mean)')
                    st.plotly_chart(figh, use_container_width=True)

    # Build Your Own
    with tabs[5]:
        st.subheader('Build Your Own')
        ds_name = st.selectbox('Dataset', options=list(data.keys()), key='dataset_custom')
        df = data[ds_name].copy()
        st.write(f'Rows: {len(df):,} | Columns: {len(df.columns)}')
        with st.expander('Preview data', expanded=False):
            st.dataframe(df.head(200))
            st.download_button('Download CSV', df.to_csv(index=False).encode('utf-8'), file_name=f'{ds_name.replace(" ", "_")}.csv')
        df_f = filter_ui(df)
        x_col, y_cols = select_columns_ui(df_f)
        plot_ui(df_f, x_col, y_cols)


if __name__ == '__main__':
    main()
