import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import io
import os
from datetime import datetime, timedelta
import numpy as np

# --- CONFIGURATION ---
STATE = 'OR'
AREA = 'Mt. Bachelor Area'
# Mt Bachelor approx coordinates for forecast
LAT = 43.98
LON = -121.69 
SITE_IDS = [815, 545, 388, 619] 
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Documents", "Snow")

START_YEAR = 2001
CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month
CURRENT_WATER_YEAR = CURRENT_YEAR + 1 if CURRENT_MONTH >= 10 else CURRENT_YEAR

# --- DATA FETCHING ---

def fetch_single_site(site_id, start_year):
    """Fetches SWE and Snow Depth data."""
    print(f"Fetching data for Site {site_id}...")
    url = (
        f"https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/"
        f"customSingleStationReport/daily/start_of_period/"
        f"{site_id}:{STATE}:SNTL|id=\"\"|name/"
        f"{start_year}-10-01,{datetime.now().strftime('%Y-%m-%d')}/"
        f"WTEQ::value,SNWD::value"
    )
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        content = response.content.decode('utf-8')
        
        if len(content) < 500: return None

        df = pd.read_csv(io.StringIO(content), comment='#')
        
        rename_map = {}
        for col in df.columns:
            if 'Snow Water Equivalent' in col or 'WTEQ' in col:
                rename_map[col] = 'SWE'
            elif 'Snow Depth' in col or 'SNWD' in col:
                rename_map[col] = 'Depth'
        
        df = df.rename(columns=rename_map)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            df = df[~df.index.duplicated(keep='first')]
            
            for col in ['SWE', 'Depth']:
                if col in df.columns:
                    df.loc[df[col] > 400, col] = np.nan 
                    df.loc[df[col] < 0, col] = np.nan
            
            return df[['SWE', 'Depth']].add_suffix(f'_{site_id}')
        return None
    except Exception as e:
        print(f"Failed site {site_id}: {e}")
        return None

def fetch_snotel_data(site_ids, start_year):
    combined_df = pd.DataFrame()
    for site_id in site_ids:
        df = fetch_single_site(site_id, start_year)
        if df is not None:
            if combined_df.empty: combined_df = df
            else: combined_df = combined_df.join(df, how='outer')
    
    if combined_df.empty: return pd.DataFrame()
    
    swe_cols = [c for c in combined_df.columns if 'SWE_' in c]
    depth_cols = [c for c in combined_df.columns if 'Depth_' in c]
    
    combined_df['Average_SWE'] = combined_df[swe_cols].mean(axis=1)
    combined_df['Average_Depth'] = combined_df[depth_cols].mean(axis=1)
    
    return combined_df

def fetch_forecasts(lat, lon, current_val):
    """
    Fetches forecasts from Open-Meteo.
    Automatically handles AI models by falling back to Precip/Temp calc 
    because they often lack explicit 'snowfall' variables.
    """
    print("Fetching Forecast models (Snowfall)...")
    
    models = {
        'GFS': 'gfs_seamless',
        'ECMWF (Hi-Res)': 'ecmwf_ifs025',  
        'ECMWF (AI)': 'ecmwf_aifs025',
        'GraphCast (AI)': 'gfs_graphcast025', # Re-enabled as a second source
        'ICON': 'icon_seamless',
        'GEM': 'gem_global',
        'MeteoFrance': 'meteofrance_arpege_world'
    }
    
    forecast_data = {}
    
    for name, model_id in models.items():
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            
            # Determine forecast days: AI models are often shorter range (10 days stable)
            f_days = 10 if 'AI' in name else 16

            # Default Parameters
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "snowfall_sum", 
                "hourly": "snowfall,precipitation,temperature_2m",
                "timezone": "auto",
                "models": model_id,
                "past_days": 0,
                "forecast_days": f_days,
                # Explicitly request mid-mountain elevation (approx 1950m / 6400ft for Bachelor)
                "elevation": 1950  
            }

            # --- FIX FOR AI MODELS ---
            # AI models often fail on hourly specifics or explicit snowfall.
            # Strategy: Switch to DAILY Aggregates as the "Source" of truth.
            if 'AI' in name:
                # Remove daily snowfall request to prevent API error if missing
                if 'daily' in params: del params['daily']
                
                # Request DAILY PRECIP & TEMP instead of hourly
                # This acts as a different data source logic
                params['daily'] = "precipitation_sum,temperature_2m_mean"
                
                # Keep hourly just in case, but minimize it
                params['hourly'] = "temperature_2m" 

            r = requests.get(url, params=params, timeout=15)
            
            if r.status_code != 200:
                print(f"Error fetching {name}: {r.status_code} - {r.text}")
                continue

            data = r.json()
            
            daily_dates = None
            daily_values = None
            
            # --- METHOD 1: Explicit Daily Snowfall (Standard Models) ---
            if 'daily' in data and 'snowfall_sum' in data['daily']:
                raw = data['daily'].get('snowfall_sum')
                if raw and not all(v is None for v in raw):
                    daily_dates = pd.to_datetime(data['daily']['time'])
                    daily_values = np.array([x if x is not None else 0.0 for x in raw], dtype=float)
                    daily_values = daily_values * 0.393701
            
            # --- METHOD 2: Daily Smart Calc (AI Models Preferred) ---
            # Uses Daily Precip Sum + Daily Mean Temp
            if daily_values is None and 'daily' in data:
                d_precip = data['daily'].get('precipitation_sum')
                d_temp = data['daily'].get('temperature_2m_mean')
                
                if d_precip and d_temp:
                    print(f"  Note: {name} using Daily Aggregates (Precip + Mean Temp).")
                    daily_dates = pd.to_datetime(data['daily']['time'])
                    
                    p_arr = np.array([x if x is not None else 0.0 for x in d_precip], dtype=float)
                    t_arr = np.array([x if x is not None else 0.0 for x in d_temp], dtype=float)
                    
                    # LOGIC: If Daily Mean Temp <= 2.0°C (35.6°F), assume Snow.
                    # This is a robust approximation for daily aggregates.
                    daily_snow_cm = np.where(t_arr <= 2.0, p_arr, 0.0)
                    
                    # Unit cm -> inches
                    daily_values = daily_snow_cm * 0.393701

            # --- METHOD 3: Hourly Smart Calc (Fallback) ---
            if daily_values is None and 'hourly' in data:
                h_precip = data['hourly'].get('precipitation')
                h_temp = data['hourly'].get('temperature_2m')
                
                if h_precip and h_temp:
                    # print(f"  Note: {name} using Hourly Smart Calc.")
                    hourly_dates = pd.to_datetime(data['hourly']['time'])
                    
                    p_arr = np.array([x if x is not None else 0.0 for x in h_precip], dtype=float)
                    t_arr = np.array([x if x is not None else 0.0 for x in h_temp], dtype=float)
                    
                    snow_cm_arr = np.where(t_arr <= 2.5, p_arr, 0.0)
                    
                    temp_df = pd.DataFrame({'val': snow_cm_arr}, index=hourly_dates)
                    daily_agg = temp_df.resample('D').sum()
                    
                    daily_dates = daily_agg.index
                    daily_values = daily_agg['val'].values * 0.393701

            # --- FINALIZE ---
            if daily_values is None:
                print(f"Warning: {name} returned NO valid data.")
                continue
            
            # Accumulate on top of current depth
            cumulative_depth = np.cumsum(daily_values) + current_val
            
            df = pd.DataFrame({'Date': daily_dates, 'Forecast_Depth': cumulative_depth})
            df = df.set_index('Date')
            forecast_data[name] = df
            print(f"  -> Fetched {name} ({len(df)} days)")

        except Exception as e:
            print(f"Could not fetch {name}: {e}")
            
    return forecast_data

# --- DATA PROCESSING ---

def process_water_years(df, metric_col):
    df = df[~((df.index.month == 2) & (df.index.day == 29))]
    
    daily = df[[metric_col]].copy()
    daily['Year'] = daily.index.year
    daily['Month'] = daily.index.month
    daily['WaterYear'] = np.where(daily['Month'] >= 10, daily['Year'] + 1, daily['Year'])
    
    def get_plot_date(idx_date):
        m, d = idx_date.month, idx_date.day
        y = 2020 if m >= 10 else 2021
        try:
            return datetime(y, m, d)
        except:
            return None

    daily['PlotDate'] = daily.index.map(get_plot_date)
    
    return daily.pivot_table(index='PlotDate', columns='WaterYear', values=metric_col, aggfunc='mean')

def normalize_forecast_dates(forecast_df):
    """Maps future forecast dates to the 2020-2021 plot axis."""
    def get_plot_date(idx_date):
        m, d = idx_date.month, idx_date.day
        y = 2020 if m >= 10 else 2021
        return datetime(y, m, d)
    
    forecast_df['PlotDate'] = forecast_df.index.map(get_plot_date)
    return forecast_df.set_index('PlotDate')

# --- PLOTTING ---

def add_metric_traces(fig, pivot_df, metric_name, unit, is_visible_by_default, forecast_dfs=None):
    start_idx = len(fig.data)
    hist_years = pivot_df.loc[:, pivot_df.columns < CURRENT_WATER_YEAR]
    
    # --- STATISTICS ---
    stats_df = pd.DataFrame(index=pivot_df.index)
    stats_df['Min'] = hist_years.min(axis=1)
    stats_df['Max'] = hist_years.max(axis=1)
    stats_df['Median'] = hist_years.median(axis=1)
    stats_df['P10'] = hist_years.quantile(0.1, axis=1)
    stats_df['P30'] = hist_years.quantile(0.3, axis=1)
    stats_df['P70'] = hist_years.quantile(0.7, axis=1)
    stats_df['P90'] = hist_years.quantile(0.9, axis=1)

    base_viz = True if is_visible_by_default else False
    legend_group = f'stats_{metric_name}'

    # Stats Traces
    fig.add_trace(go.Scatter(x=stats_df.index, y=stats_df['Min'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip', visible=base_viz))
    fig.add_trace(go.Scatter(
        x=stats_df.index, y=stats_df['P10'], mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(255, 80, 80, 0.25)', # Brighter Red, higher opacity
        name=f'{metric_name} Low (<10%)', visible=base_viz, legendgroup=legend_group
    ))
    fig.add_trace(go.Scatter(x=stats_df.index, y=stats_df['P10'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip', visible=base_viz))
    fig.add_trace(go.Scatter(
        x=stats_df.index, y=stats_df['P30'], mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(255, 225, 50, 0.25)', # Brighter Gold, higher opacity
        name=f'{metric_name} Mod (10-30%)', visible=base_viz, legendgroup=legend_group
    ))
    fig.add_trace(go.Scatter(x=stats_df.index, y=stats_df['P30'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip', visible=base_viz))
    fig.add_trace(go.Scatter(
        x=stats_df.index, y=stats_df['P70'], mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(80, 235, 80, 0.25)', # Brighter Green, higher opacity
        name=f'{metric_name} Normal (30-70%)', visible=base_viz, legendgroup=legend_group
    ))
    fig.add_trace(go.Scatter(x=stats_df.index, y=stats_df['P70'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip', visible=base_viz))
    fig.add_trace(go.Scatter(
        x=stats_df.index, y=stats_df['Max'], mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(60, 160, 255, 0.25)', # Brighter Blue, higher opacity
        name=f'{metric_name} High (>70%)', visible=base_viz, legendgroup=legend_group
    ))
    fig.add_trace(go.Scatter(
        x=stats_df.index, y=stats_df['Median'], mode='lines',
        name=f'{metric_name} Median', line=dict(color='rgba(160, 255, 160, 1.0)', width=2, dash='dash'), # Brighter Median
        visible=base_viz, legendgroup=legend_group
    ))

    # --- HISTORICAL YEARS ---
    peaks = hist_years.max()
    year_min_name = peaks.idxmin()
    year_max_name = peaks.idxmax()

    years = sorted(hist_years.columns)
    colors = px.colors.sample_colorscale("Turbo", [n/(len(years)-1) for n in range(len(years))])
    
    year_indices = []
    
    for i, year in enumerate(years):
        label = f"{year-1}-{year}"
        is_highlight = (year == year_min_name or year == year_max_name)
        
        if is_visible_by_default:
            viz = True
            if is_highlight:
                opac, wid, leg = 0.6, 1.5, True
                hov = f'<b>{label}</b><br>{metric_name}: %{{y:.1f}} {unit}<extra></extra>'
            else:
                opac, wid, leg = 0.0, 1.0, False
                hov = None
        else:
            viz, opac, wid, leg, hov = False, 0.0, 1.0, False, None
        
        fig.add_trace(go.Scatter(
            x=pivot_df.index, y=pivot_df[year],
            mode='lines', name=label,
            line=dict(color=colors[i], width=wid),
            opacity=opac, visible=viz, showlegend=leg,
            hoverinfo='skip' if hov is None else 'all', hovertemplate=hov
        ))
        year_indices.append(len(fig.data) - 1)

    # --- CURRENT YEAR ---
    curr_indices = []
    if CURRENT_WATER_YEAR in pivot_df.columns:
        curr = pivot_df[CURRENT_WATER_YEAR].dropna()
        if not curr.empty:
            last_val = curr.iloc[-1]
            last_date = curr.index[-1]
            
            # Safe access to median
            try:
                med_val = stats_df['Median'].loc[last_date]
                pct_med = (last_val / med_val * 100) if med_val > 0 else 0
            except:
                pct_med = 0
            
            label = f"Current ({CURRENT_WATER_YEAR-1}-{CURRENT_WATER_YEAR})"
            
            fig.add_trace(go.Scatter(
                x=curr.index, y=curr, mode='lines', name=label,
                line=dict(color='#ff00ff', width=3.5), zorder=10,
                visible=base_viz,
                hovertemplate=f'<b>Current</b><br>{metric_name}: %{{y:.1f}} {unit}<extra></extra>'
            ))
            curr_indices.append(len(fig.data) - 1)
            
            fig.add_trace(go.Scatter(
                x=[last_date], y=[last_val],
                mode='markers+text',
                marker=dict(color='#ff00ff', size=10, line=dict(color='white', width=1)),
                text=[f"{last_val:.1f}<br>{pct_med:.0f}%"],
                textposition="middle right", textfont=dict(color='#ff00ff'),
                showlegend=False, zorder=10, visible=base_viz
            ))
            curr_indices.append(len(fig.data) - 1)
            
            # --- FORECAST TRACES (Only for DEPTH now) ---
            if metric_name == "Depth" and forecast_dfs:
                model_colors = {
                    'GFS': '#00d2ff',          # Cyan
                    'ECMWF (Hi-Res)': '#ff9f00', # Orange
                    'ECMWF (AI)': '#ff5500',   # Dark Orange
                    'GraphCast (AI)': '#cc0000', # Deep Red
                    'ICON': '#b19cd9',         # Light Purple
                    'GEM': '#32cd32',          # Lime Green
                    'MeteoFrance': '#ff69b4'   # Hot Pink
                }
                
                for model_name, f_df in forecast_dfs.items():
                    norm_f = normalize_forecast_dates(f_df)
                    
                    # Filter dates to only show future relative to last_date
                    # This prevents the "zig-zag" back to 2020 if the dates overlap strangely
                    
                    stitch_x = [last_date] + list(norm_f.index)
                    stitch_y = [last_val] + list(norm_f['Forecast_Depth'])
                    
                    fig.add_trace(go.Scatter(
                        x=stitch_x, y=stitch_y,
                        mode='lines',
                        name=f"{model_name} Forecast",
                        line=dict(color=model_colors.get(model_name, 'white'), width=2, dash='dot'),
                        visible=base_viz,
                        hovertemplate=f'<b>{model_name}</b><br>Forecast: %{{y:.1f}} {unit}<extra></extra>'
                    ))
                    curr_indices.append(len(fig.data) - 1)

    return {
        'stats': list(range(start_idx, start_idx + 9)),
        'years': year_indices,
        'year_names': years,
        'min_year_idx': start_idx + 9 + years.index(year_min_name) if year_min_name in years else -1,
        'max_year_idx': start_idx + 9 + years.index(year_max_name) if year_max_name in years else -1,
        'current': curr_indices
    }

def create_advanced_plot(df_raw):
    print("Generating High-Fidelity Plot...")
    
    pivot_swe = process_water_years(df_raw, 'Average_SWE')
    pivot_depth = process_water_years(df_raw, 'Average_Depth')
    
    current_depth_val = 0
    if CURRENT_WATER_YEAR in pivot_depth.columns:
        last_valid = pivot_depth[CURRENT_WATER_YEAR].dropna()
        if not last_valid.empty:
            current_depth_val = last_valid.iloc[-1]

    # Fetch Forecasts (using Snow Depth base)
    forecasts = fetch_forecasts(LAT, LON, current_depth_val)

    fig = go.Figure()
    
    # SWE: No Forecasts
    meta_swe = add_metric_traces(fig, pivot_swe, "SWE", "in", is_visible_by_default=True, forecast_dfs=None)
    # Depth: Forecasts Attached
    meta_depth = add_metric_traces(fig, pivot_depth, "Depth", "in", is_visible_by_default=False, forecast_dfs=forecasts)
    
    total_traces = len(fig.data)
    
    # --- VISIBILITY VECTORS ---
    def get_viz_vector(active_meta):
        viz = [False] * total_traces
        for idx in active_meta['stats'] + active_meta['current']:
            viz[idx] = True
        for idx in active_meta['years']:
            viz[idx] = True
        return viz

    swe_viz_vector = get_viz_vector(meta_swe)
    depth_viz_vector = get_viz_vector(meta_depth)

    # --- MENUS ---
    metric_buttons = [
        dict(
            label="Show SWE",
            method="update",
            args=[{"visible": swe_viz_vector}, {"yaxis.title.text": "SWE (Inches)"}]
        ),
        dict(
            label="Show Snow Depth + Forecasts",
            method="update",
            args=[{"visible": depth_viz_vector}, {"yaxis.title.text": "Snow Depth (Inches)"}]
        )
    ]

    # Grid Buttons
    years = meta_swe['year_names']
    all_year_indices = meta_swe['years'] + meta_depth['years']
    
    min_max_indices = {
        meta_swe['min_year_idx'], meta_swe['max_year_idx'],
        meta_depth['min_year_idx'], meta_depth['max_year_idx']
    }
    
    reset_opacity = []
    reset_showlegend = []
    reset_width = []
    reset_hover = []
    
    for idx in all_year_indices:
        if idx in min_max_indices:
            reset_opacity.append(0.6)
            reset_showlegend.append(True)
            reset_width.append(1.5)
            reset_hover.append("all")
        else:
            reset_opacity.append(0.0)
            reset_showlegend.append(False)
            reset_width.append(1.0)
            reset_hover.append("skip")

    clear_button = dict(
        label="Reset Highlights",
        method="restyle",
        args=[
            {"opacity": reset_opacity, "showlegend": reset_showlegend, "line.width": reset_width, "hoverinfo": reset_hover},
            all_year_indices
        ]
    )
    
    years_per_row = 8
    grid_menus = []
    row_height = 0.05
    year_chunks = [years[i:i + years_per_row] for i in range(0, len(years), years_per_row)]
    
    updatemenus = [
        dict(
            buttons=metric_buttons,
            type="buttons", direction="left", pad={"r": 10, "t": 10},
            showactive=True, x=0.0, xanchor="left", y=1.25, yanchor="top",
            bgcolor="#333", bordercolor="#555"
        )
    ]
    
    for r, chunk in enumerate(year_chunks):
        row_buttons = []
        for i, year in enumerate(chunk):
            global_idx = years.index(year)
            idx_swe = meta_swe['years'][global_idx]
            idx_depth = meta_depth['years'][global_idx]
            lbl = f"{str(year-1)[2:]}-{str(year)[2:]}"
            
            button = dict(
                label=lbl,
                method="restyle",
                args=[
                    {"line.width": 2.5, "opacity": 1.0, "showlegend": True, "hoverinfo": "all"},
                    [idx_swe, idx_depth] 
                ]
            )
            row_buttons.append(button)
            
        if r == 0: row_buttons.insert(0, clear_button)

        menu = dict(
            buttons=row_buttons,
            type="buttons", direction="left", pad={"r": 2, "t": 2},
            showactive=False, x=0.0, xanchor="left", y=1.12 - (r * row_height), yanchor="top",
            bgcolor="rgba(0,0,0,0)", font=dict(size=10)
        )
        updatemenus.append(menu)

    fig.update_layout(
        title=dict(text=f"<b>{AREA} Interactive Snowpack + Forecasts</b>", y=0.98, x=0.05),
        template="plotly_dark",
        # Updated Range: Oct 1st to Aug 1st
        xaxis=dict(tickformat='%b %d', range=['2020-10-01', '2021-08-01'], showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(
            title=dict(text='SWE (Inches)', font=dict(size=14)),
            showgrid=True, 
            gridcolor='rgba(255,255,255,0.1)',
            automargin=True
        ),
        height=850,
        margin=dict(t=180, r=20, l=80),
        legend=dict(x=0.01, y=0.85, bgcolor="rgba(0,0,0,0.5)"), 
        updatemenus=updatemenus
    )
    
    fig.add_annotation(text="Metric:", x=0.0, y=1.28, xref="paper", yref="paper", showarrow=False, font=dict(color="gray", size=10), xanchor="left")
    fig.add_annotation(text="Select Seasons:", x=0.0, y=1.15, xref="paper", yref="paper", showarrow=False, font=dict(color="gray", size=10), xanchor="left")

    # Configure the export button
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d'], 
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'{AREA.replace(" ", "_")}_Snowpack_Forecast',
            'height': 900,
            'width': 1600,
            'scale': 6
        }
    }

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    path = os.path.join(OUTPUT_DIR, f'snotel_forecast_{datetime.now().strftime("%Y%m%d")}.html')
    fig.write_html(path, config=config)
    print(f"Graph saved to: {path}")

if __name__ == "__main__":
    df_raw = fetch_snotel_data(SITE_IDS, START_YEAR)
    if not df_raw.empty:
        create_advanced_plot(df_raw)
