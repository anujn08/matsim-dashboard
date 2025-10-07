How to run the dashboard

You don’t need to run any analysis script. Just install deps and start Streamlit.

Steps (Windows PowerShell)

Open PowerShell.

Run these commands exactly:

python -m pip install -r "c:\Users\anujn\OneDrive\Desktop\Matsim_Analysis\requirements.txt"

python -m streamlit run "c:\Users\anujn\OneDrive\Desktop\Matsim_Analysis\app_streamlit.py" --server.port 8503 --server.headless true

Open in your browser:

http://localhost:8503

If localhost doesn’t load, try http://10.61.38.219:8503

Using the dashboard

Tabs:

Overview: KPIs per run from analysis_run_outputs/all_days/analysis_summary.csv

Run Comparison: compare runs on any metric

Daily Progression: trends across days

Dham Activity: dham persons by run/day

Facility Rest: facility rest hotspots by run/day

Build Your Own: pick dataset, filters, X/Y, and chart type

Filters (where shown): run_id, day, dham_activity_type, facility_id.

Data preview: expand “Preview data” and use Download to export CSV.

Troubleshooting

Streamlit not found:

You already used python -m streamlit, which avoids PATH issues.

Port already in use:

Use another port, e.g.:

python -m streamlit run "c:\\Users\\anujn\\OneDrive\\Desktop\\Matsim_Analysis\\app_streamlit.py" --server.port 8504 --server.headless true

Stale page or widgets:

Streamlit menu → Clear cache → Rerun, or refresh the browser.

Firewall prompts:

Allow access for Python so the local server can bind to the port
