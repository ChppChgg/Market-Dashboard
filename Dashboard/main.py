"""
Stock Market Decision Support System (DSS) Dashboard
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from MarketData import SECTORS, COMMODITY_NAMES, get_all_tickers, load_ticker_data, get_ticker_sector, update_ticker_data, fetch_ticker_data
from HMMRegime import (
    detect_regimes, 
    get_regime_zones, 
    get_trend_label, 
    get_volatility_label,
    get_volatility_color,
    REGIME_COLORS,
    VOLATILITY_COLORS,
    validate_regime_detection,
    print_validation_report,
    calculate_regime_stress_indicators,
    get_current_regime_outlook
)


# =============================================================================
# CACHED HMM COMPUTATION
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def compute_hmm_regimes(ticker: str, df_json: str) -> dict:
    """
    Compute HMM regimes with caching.
    Results are cached per ticker to avoid recomputation.
    
    Args:
        ticker: Ticker symbol (used as cache key)
        df_json: DataFrame as JSON string (for cache invalidation if data changes)
    
    Returns:
        Dictionary with regime results
    """
    # Reconstruct DataFrame from JSON
    df = pd.read_json(df_json)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Run HMM detection
    regime_result = detect_regimes(df)
    all_zones = get_regime_zones(regime_result['df'])
    
    return {
        'trend': regime_result['current_trend'],
        'volatility': regime_result['current_volatility'],
        'stats': regime_result['regime_stats'],
        'all_zones': all_zones,
        'computed_at': datetime.now().isoformat()
    }


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Stock Market DSS",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# STATIC FILES LOADER
# =============================================================================

STATIC_DIR = Path(__file__).parent / "static"
CSS_DIR = STATIC_DIR / "css"
HTML_DIR = STATIC_DIR / "html"
CACHED_DIR = Path(__file__).parent / "cached"
USER_DATA_FILE = CACHED_DIR / "user_data.json"


# =============================================================================
# USER DATA PERSISTENCE
# =============================================================================

def load_user_data() -> dict:
    """Load user data (watchlist, preferences) from JSON file."""
    # Ensure cached directory exists
    CACHED_DIR.mkdir(exist_ok=True)
    
    if USER_DATA_FILE.exists():
        try:
            with open(USER_DATA_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {'watchlist': []}
    return {'watchlist': []}


def save_user_data(data: dict):
    """Save user data to JSON file."""
    # Ensure cached directory exists
    CACHED_DIR.mkdir(exist_ok=True)
    
    try:
        with open(USER_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        st.error(f"Could not save user data: {e}")


def get_watchlist() -> list:
    """Get watchlist from persistent storage."""
    data = load_user_data()
    return data.get('watchlist', [])


def save_watchlist(watchlist: list):
    """Save watchlist to persistent storage."""
    data = load_user_data()
    data['watchlist'] = watchlist
    save_user_data(data)


def load_css(filename: str) -> str:
    """Load CSS file from static/css directory."""
    css_path = CSS_DIR / filename
    if css_path.exists():
        return css_path.read_text(encoding='utf-8')
    return ""


def load_html(filename: str) -> str:
    """Load HTML template from static/html directory."""
    html_path = HTML_DIR / filename
    if html_path.exists():
        return html_path.read_text(encoding='utf-8')
    return ""


# =============================================================================
# LOAD AND APPLY CSS
# =============================================================================

css_content = load_css("styles.css")
st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================



def get_display_name(ticker: str) -> str:
    """Get a display-friendly name for a ticker."""
    if ticker in COMMODITY_NAMES:
        return COMMODITY_NAMES[ticker]
    return ticker


def get_ticker_from_display(display_name: str) -> str:
    """Get the original ticker symbol from display name."""
    # Check commodities first
    for ticker, name in COMMODITY_NAMES.items():
        if name == display_name:
            return ticker
    # Otherwise display name is the ticker
    return display_name


def get_sector_display_name(sector: str) -> str:
    """Convert sector key to display name."""
    return sector.replace("_", " ")


def create_sector_box(sector: str, tickers: list) -> str:
    """Create HTML for a sector box with stock chips."""
    display_name = get_sector_display_name(sector)
    
    # Load templates
    chip_template = load_html("stock_chip.html")
    box_template = load_html("sector_box.html")
    
    # Build chips HTML
    chips_html = ""
    for ticker in tickers:
        display_ticker = get_display_name(ticker)
        chips_html += chip_template.format(ticker=display_ticker)
    
    # Build sector box
    html = box_template.format(sector_name=display_name, stock_chips=chips_html)
    return html


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the sidebar with asset selection and date range."""
    
    with st.sidebar:
        st.markdown("## 🎯 Asset Selection")
        st.markdown("---")
        
        # Get all tickers
        available_tickers = get_all_tickers()
        
        # Asset selection with search
        st.markdown("### 🔍 Select Asset")
        selected_asset = st.selectbox(
            "Choose an asset to analyze",
            options=["-- Select Asset --"] + [get_display_name(t) for t in available_tickers],
            index=0,
            help="Select a stock, ETF, or commodity to analyze"
        )
        
        st.markdown("---")
        
        # Date range selection
        st.markdown("### 📅 Date Range")
        
        # Default date range (last 1 year)
        default_end = datetime.now().date()
        default_start = default_end - timedelta(days=365)
        min_date = datetime(1980, 1, 1).date()  # Allow historical data back to 1980
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=default_start,
                min_value=min_date,
                max_value=default_end
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=default_end,
                min_value=min_date,
                max_value=default_end
            )
        
        # Quick date range buttons
        st.markdown("**Quick Select:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("1M", use_container_width=True):
                start_date = default_end - timedelta(days=30)
        with col2:
            if st.button("6M", use_container_width=True):
                start_date = default_end - timedelta(days=180)
        with col3:
            if st.button("1Y", use_container_width=True):
                start_date = default_end - timedelta(days=365)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("3Y", use_container_width=True):
                start_date = default_end - timedelta(days=1095)
        with col2:
            if st.button("5Y", use_container_width=True):
                start_date = default_end - timedelta(days=1825)
        with col3:
            if st.button("MAX", use_container_width=True):
                start_date = datetime(2000, 1, 1).date()
        
        st.markdown("---")
        
        # Analysis button
        analyze_clicked = st.button(
            "🚀 Analyze Asset",
            type="primary",
            use_container_width=True,
            disabled=(selected_asset == "-- Select Asset --")
        )
        
        st.markdown("---")
        
        # Data status
        st.markdown("### 📊 Data Status")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            with st.spinner("Updating market data..."):
                # This would trigger data update
                st.success("Data refreshed!")
        
        # Store selections in session state
        if selected_asset != "-- Select Asset --":
            st.session_state['selected_asset'] = get_ticker_from_display(selected_asset)
        st.session_state['start_date'] = start_date
        st.session_state['end_date'] = end_date
        
        # Switch to analysis tab when analyze is clicked
        if analyze_clicked and selected_asset != "-- Select Asset --":
            st.session_state['active_tab'] = 1  # Asset Analysis tab
            st.rerun()
        
        return selected_asset, start_date, end_date


# =============================================================================
# MAIN CONTENT - MARKET OVERVIEW
# =============================================================================

def render_market_overview():
    """Render the Market Overview tab with sector boxes."""
    
    # Load header from template
    header_html = load_html("market_overview_header.html")
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Summary metrics
    total_sectors = len(SECTORS)
    total_assets = sum(len(tickers) for tickers in SECTORS.values())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("📁 Total Sectors", total_sectors)
    with col2:
        st.metric("📊 Total Assets", total_assets)
    
    # -------------------------------------------------------------------------
    # WATCHLIST SECTION
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### ⭐ My Watchlist")
    st.caption("Add up to 5 stocks to your watchlist for quick analysis")
    
    # Initialize watchlist from persistent storage
    if 'watchlist' not in st.session_state:
        st.session_state['watchlist'] = get_watchlist()
    
    # Add to watchlist UI
    col1, col2 = st.columns([3, 1])
    
    all_tickers = get_all_tickers()
    available_for_watchlist = [t for t in all_tickers if t not in st.session_state['watchlist']]
    
    with col1:
        new_ticker = st.selectbox(
            "Add stock to watchlist",
            options=["-- Select Stock --"] + [get_display_name(t) for t in available_for_watchlist],
            key="watchlist_select",
            label_visibility="collapsed"
        )
    
    with col2:
        add_disabled = (
            new_ticker == "-- Select Stock --" or 
            len(st.session_state['watchlist']) >= 5
        )
        if st.button("➕ Add", use_container_width=True, disabled=add_disabled):
            # Find the original ticker symbol
            for t in all_tickers:
                if get_display_name(t) == new_ticker:
                    if t not in st.session_state['watchlist']:
                        st.session_state['watchlist'].append(t)
                        save_watchlist(st.session_state['watchlist'])  # Persist
                    break
            st.rerun()
    
    # Show watchlist limit warning
    if len(st.session_state['watchlist']) >= 5:
        st.warning("⚠️ Watchlist full (maximum 5 stocks)")
    
    # Display watchlist
    if st.session_state['watchlist']:
        watchlist_cols = st.columns(5)
        
        for idx, ticker in enumerate(st.session_state['watchlist']):
            with watchlist_cols[idx]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%); 
                            border-radius: 10px; padding: 1rem; text-align: center;
                            border: 1px solid #3d7a4d; margin-bottom: 0.5rem;">
                    <div style="color: #8fc4a0; font-size: 0.8rem;">#{idx + 1}</div>
                    <div style="color: #ffffff; font-weight: bold; font-size: 1.1rem;">
                        {get_display_name(ticker)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("📈", key=f"analyze_{ticker}", use_container_width=True, help="Analyze"):
                        st.session_state['selected_asset'] = ticker
                        st.session_state['active_tab'] = 1
                        st.rerun()
                with col_b:
                    if st.button("🗑️", key=f"remove_{ticker}", use_container_width=True, help="Remove"):
                        st.session_state['watchlist'].remove(ticker)
                        save_watchlist(st.session_state['watchlist'])  # Persist
                        st.rerun()
    else:
        st.info("📝 Your watchlist is empty. Add stocks above to get started!")

    st.markdown("---")
    st.markdown("### 📋 Sectors & Assets")
    st.caption("Click on any stock to analyze it")
    
    # Display sector boxes in a grid (3 columns)
    sector_items = list(SECTORS.items())
    
    # Create rows of 3 sectors each
    for i in range(0, len(sector_items), 3):
        cols = st.columns(3)
        
        for j, col in enumerate(cols):
            if i + j < len(sector_items):
                sector, tickers = sector_items[i + j]
                with col:
                    # Sector header
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
                                border-radius: 12px 12px 0 0; padding: 0.8rem 1.2rem;
                                border: 1px solid #3d7ab5; border-bottom: 2px solid #4a9eff;">
                        <span style="color: #ffffff; font-size: 1rem; font-weight: 700;">📊 {get_sector_display_name(sector)}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Stock buttons in a container
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
                                border-radius: 0 0 12px 12px; padding: 0.5rem 1rem 1rem 1rem;
                                border: 1px solid #3d7ab5; border-top: none; margin-bottom: 1rem;">
                    """, unsafe_allow_html=True)
                    
                    # Create button grid for tickers (5 per row max)
                    ticker_cols = st.columns(5)
                    for t_idx, ticker in enumerate(tickers):
                        with ticker_cols[t_idx % 5]:
                            if st.button(get_display_name(ticker), key=f"btn_{sector}_{ticker}", use_container_width=True):
                                st.session_state['selected_asset'] = ticker
                                st.session_state['active_tab'] = 1
                                st.rerun()
                    
                    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# MAIN CONTENT - ASSET ANALYSIS (Placeholder)
# =============================================================================

def render_asset_analysis():
    """Render the Asset Analysis tab with price chart."""
    
    selected_asset = st.session_state.get('selected_asset')
    
    if not selected_asset:
        # Load header from template
        header_html = load_html("asset_analysis_header.html")
        st.markdown(header_html, unsafe_allow_html=True)
        st.warning("⚠️ Please select an asset from the sidebar or Market Overview to begin analysis.")
        return
    
    # Display asset header
    display_name = get_display_name(selected_asset)
    sector = get_ticker_sector(selected_asset)
    sector_display = get_sector_display_name(sector) if sector else "Unknown"
    
    st.markdown(f"""
    <div class="main-header">
        <h1>📈 {display_name}</h1>
        <p>{sector_display} | Ticker: {selected_asset}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get date range from session state
    start_date = st.session_state.get('start_date', datetime.now().date() - timedelta(days=365))
    end_date = st.session_state.get('end_date', datetime.now().date())
    
    # Fetch data
    with st.spinner(f"Loading data for {display_name}..."):
        # Try to load from saved data first
        if sector:
            df = load_ticker_data(selected_asset, sector)
        else:
            df = pd.DataFrame()
        
        # If no saved data, fetch from YFinance
        if df.empty:
            df = fetch_ticker_data(selected_asset)
        
        if df.empty:
            st.error(f"❌ Could not load data for {display_name}. Please try again later.")
            return
    
    # Filter by date range
    df['Date'] = pd.to_datetime(df['Date'])
    mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
    df_filtered = df[mask].copy()
    
    if df_filtered.empty:
        st.warning(f"No data available for the selected date range ({start_date} to {end_date})")
        return
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    latest = df_filtered.iloc[-1]
    first = df_filtered.iloc[0]
    price_change = latest['Close'] - first['Close']
    price_change_pct = (price_change / first['Close']) * 100
    
    with col1:
        st.metric("Current Price", f"${latest['Close']:.2f}")
    with col2:
        st.metric("Period Change", f"${price_change:.2f}", f"{price_change_pct:+.2f}%")
    with col3:
        st.metric("Period High", f"${df_filtered['High'].max():.2f}")
    with col4:
        st.metric("Period Low", f"${df_filtered['Low'].min():.2f}")
    
    st.markdown("---")
    
    # =========================================================================
    # HMM REGIME DETECTION - BUTTON TRIGGERED
    # =========================================================================
    
    # Initialize session state for HMM results (cached per ticker, not per date range)
    hmm_key = f"hmm_result_{st.session_state.selected_asset}"
    if hmm_key not in st.session_state:
        st.session_state[hmm_key] = None
    
    # Button to run HMM analysis
    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        run_hmm = st.button("🔍 Run Regime Detection", type="primary", help="Analyze market regimes using Hidden Markov Model on FULL history")
    
    regime_detected = False
    current_trend = None
    current_volatility = None
    regime_stats = None
    regime_zones = []
    
    if run_hmm:
        with st.spinner("Detecting market regimes on full dataset..."):
            try:
                # Use FULL dataset (df) for HMM, not the filtered one
                # Use FULL dataset for HMM analysis
                df_for_hmm = df.copy()
                df_for_hmm['Date'] = pd.to_datetime(df_for_hmm['Date'])
                
                if len(df_for_hmm) < 60:
                    st.error("❌ Not enough historical data for regime detection (need at least 60 days)")
                    st.session_state[hmm_key] = None
                else:
                    # Use cached computation - will be instant on repeat calls
                    df_json = df_for_hmm.to_json()
                    result = compute_hmm_regimes(selected_asset, df_json)
                    
                    st.session_state[hmm_key] = {
                        'trend': result['trend'],
                        'volatility': result['volatility'],
                        'stats': result['stats'],
                        'all_zones': result['all_zones'],
                        'computed_at': datetime.now()
                    }
            except Exception as e:
                st.error(f"❌ Could not detect regimes: {str(e)}")
                st.session_state[hmm_key] = None
    
    # Load cached HMM results if available
    if st.session_state[hmm_key] is not None:
        regime_detected = True
        current_trend = st.session_state[hmm_key]['trend']
        current_volatility = st.session_state[hmm_key]['volatility']
        regime_stats = st.session_state[hmm_key]['stats']
        all_zones = st.session_state[hmm_key]['all_zones']
        
        print(f"[DEBUG] Loaded {len(all_zones)} total zones from cache")
        
        # Filter zones to selected date range for display
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        regime_zones = []
        for zone in all_zones:
            zone_start = pd.to_datetime(zone['start'])
            zone_end = pd.to_datetime(zone['end'])
            
            # Check if zone overlaps with selected date range
            if zone_end >= start_dt and zone_start <= end_dt:
                # Clip zone to date range
                clipped_start = max(zone_start, start_dt)
                clipped_end = min(zone_end, end_dt)
                regime_zones.append({
                    'start': clipped_start,
                    'end': clipped_end,
                    'regime': zone['regime'],
                    'color': zone['color']
                })
        
        print(f"[DEBUG] Filtered to {len(regime_zones)} zones for date range {start_dt} to {end_dt}")
        
        with col_status:
            st.success(f"✅ **{current_trend.capitalize()}** trend | **{current_volatility.capitalize()}** vol | {len(regime_zones)} zones in view")
    
    # =========================================================================
    # MAIN CONTENT LAYOUT: Chart (left) + Metrics (right)
    # =========================================================================
    
    chart_col, metrics_col = st.columns([3, 1])
    
    with chart_col:
        st.markdown("### 📊 Price Chart & Volume")
        
        # Chart type selection
        chart_type = st.radio("Chart Type", ["Candlestick", "Line"], horizontal=True)
        
        # Create subplots with shared x-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
            subplot_titles=(None, None)
        )
        
        # Add price trace first, then we'll add zones
        price_data = df_filtered.copy()
        
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=df_filtered['Date'],
                open=df_filtered['Open'],
                high=df_filtered['High'],
                low=df_filtered['Low'],
                close=df_filtered['Close'],
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444',
                name='Price'
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=df_filtered['Date'],
                y=df_filtered['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#4a9eff', width=2)
            ), row=1, col=1)
        
        # Add volume bars
        # Color volume bars based on price change
        colors = ['#00ff88' if df_filtered['Close'].iloc[i] >= df_filtered['Open'].iloc[i] 
                  else '#ff4444' for i in range(len(df_filtered))]
        
        fig.add_trace(go.Bar(
            x=df_filtered['Date'],
            y=df_filtered['Volume'],
            marker_color=colors,
            name='Volume',
            showlegend=False
        ), row=2, col=1)
        
        fig.update_layout(
            title=f"{display_name} - {start_date} to {end_date}",
            template="plotly_dark",
            height=650,
            xaxis_rangeslider_visible=True,
            xaxis_rangeslider=dict(
                visible=True,
                thickness=0.08,
                bgcolor="#1e3a5f",
                bordercolor="#4a9eff",
                borderwidth=1,
                yaxis_rangemode="auto"
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False
        )
        
        # Remove gaps for weekends and holidays (when market is closed)
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # Hide weekends
            ],
            row=1, col=1
        )
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # Hide weekends
            ],
            row=2, col=1
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        # Add regime background zones if detected (after layout is set)
        if regime_detected and regime_zones:
            for zone in regime_zones:
                fig.add_shape(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=zone['start'],
                    x1=zone['end'],
                    y0=0.25,  # Start above volume chart
                    y1=1,     # Full height of price chart
                    fillcolor=zone['color'],
                    layer="below",
                    line_width=0,
                )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # REGIME METRICS SIDEBAR
    # =========================================================================
    
    with metrics_col:
        st.markdown("### 📈 Market Regime")
        
        if regime_detected:
            # Current Trend Regime Box
            trend_color = "#00ff88" if current_trend == "bullish" else "#ff4444" if current_trend == "bearish" else "#888888"
            trend_emoji = "🟢" if current_trend == "bullish" else "🔴" if current_trend == "bearish" else "⚪"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
                        border-radius: 10px; padding: 1rem; margin-bottom: 1rem;
                        border: 2px solid {trend_color};">
                <div style="color: #a0c4e8; font-size: 0.8rem; margin-bottom: 0.3rem;">TREND</div>
                <div style="color: {trend_color}; font-size: 1.4rem; font-weight: bold;">
                    {trend_emoji} {get_trend_label(current_trend)}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Current Volatility Regime Box
            vol_color = get_volatility_color(current_volatility)
            vol_emoji = "🟢" if current_volatility == "low" else "🟡" if current_volatility == "medium" else "🔴"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
                        border-radius: 10px; padding: 1rem; margin-bottom: 1rem;
                        border: 2px solid {vol_color};">
                <div style="color: #a0c4e8; font-size: 0.8rem; margin-bottom: 0.3rem;">VOLATILITY</div>
                <div style="color: {vol_color}; font-size: 1.4rem; font-weight: bold;">
                    {vol_emoji} {get_volatility_label(current_volatility)}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # =================================================================
            # FORWARD-LOOKING REGIME OUTLOOK
            # =================================================================
            st.markdown("---")
            st.markdown("#### 🔮 Regime Outlook")
            
            try:
                # Get regime outlook
                df_for_outlook = df.copy()
                df_for_outlook['Date'] = pd.to_datetime(df_for_outlook['Date'])
                df_json = df_for_outlook.to_json()
                
                # Reconstruct regime data
                regime_result = detect_regimes(df_for_outlook)
                regime_df = regime_result['df']
                
                # Calculate stress indicators
                regime_df = calculate_regime_stress_indicators(regime_df)
                outlook = get_current_regime_outlook(regime_df)
                
                # Stress level display
                stress = outlook.get('stress_level', 50)
                stress_interp = outlook.get('stress_interpretation', 'Moderate')
                stress_color = "#00ff88" if stress < 25 else "#ffc107" if stress < 50 else "#ff8800" if stress < 75 else "#ff4444"
                
                st.markdown(f"""
                <div style="background: #1a2f4a; border-radius: 8px; padding: 0.8rem; margin-bottom: 0.8rem;
                            border-left: 3px solid {stress_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #a0c4e8; font-size: 0.8rem;">Regime Stress</span>
                        <span style="color: {stress_color}; font-weight: bold;">{stress:.0f}/100</span>
                    </div>
                    <div style="background: #0d1f2d; border-radius: 4px; height: 8px; margin-top: 0.5rem; overflow: hidden;">
                        <div style="background: {stress_color}; width: {stress}%; height: 100%;"></div>
                    </div>
                    <div style="color: #666; font-size: 0.7rem; margin-top: 0.3rem;">{stress_interp} - {outlook.get('outlook', '')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Regime age
                regime_age = outlook.get('regime_age_days', 0)
                avg_duration = outlook.get('avg_regime_duration', 30)
                age_pct = min(100, (regime_age / avg_duration) * 100) if avg_duration > 0 else 50
                
                st.markdown(f"""
                <div style="background: #1a2f4a; border-radius: 8px; padding: 0.8rem; margin-bottom: 0.8rem;">
                    <div style="color: #a0c4e8; font-size: 0.8rem;">Regime Age</div>
                    <div style="color: #fff; font-size: 1.1rem; font-weight: bold;">{regime_age} days</div>
                    <div style="color: #666; font-size: 0.7rem;">Avg duration: ~{avg_duration} days</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Active signals
                signals = outlook.get('signals', [])
                if signals:
                    st.markdown("**Active Signals:**")
                    for signal in signals:
                        st.markdown(f"<div style='color: #a0c4e8; font-size: 0.8rem; padding: 0.2rem 0;'>{signal}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color: #666; font-size: 0.8rem;'>No warning signals active</div>", unsafe_allow_html=True)
                    
            except Exception as e:
                st.markdown(f"<div style='color: #888; font-size: 0.8rem;'>Outlook unavailable</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Regime Legend
            st.markdown("#### 📊 Chart Legend")
            st.markdown("""
            <div style="font-size: 0.85rem;">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <div style="width: 20px; height: 20px; background: rgba(0, 255, 136, 0.3); margin-right: 8px; border-radius: 3px;"></div>
                    <span style="color: #ccc;">Bullish</span>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <div style="width: 20px; height: 20px; background: rgba(128, 128, 128, 0.3); margin-right: 8px; border-radius: 3px;"></div>
                    <span style="color: #ccc;">Neutral</span>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <div style="width: 20px; height: 20px; background: rgba(255, 68, 68, 0.3); margin-right: 8px; border-radius: 3px;"></div>
                    <span style="color: #ccc;">Bearish</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Regime Statistics
            st.markdown("#### 📈 Regime Stats")
            
            for regime in ['bullish', 'neutral', 'bearish']:
                if regime in regime_stats:
                    stats = regime_stats[regime]
                    emoji = "🟢" if regime == "bullish" else "🔴" if regime == "bearish" else "⚪"
                    color = "#00ff88" if regime == "bullish" else "#ff4444" if regime == "bearish" else "#888888"
                    
                    st.markdown(f"""
                    <div style="background: #1a2f4a; border-radius: 8px; padding: 0.8rem; margin-bottom: 0.5rem;
                                border-left: 3px solid {color};">
                        <div style="color: {color}; font-weight: bold; font-size: 0.9rem;">{emoji} {regime.capitalize()}</div>
                        <div style="color: #888; font-size: 0.75rem; margin-top: 0.3rem;">
                            Time: {stats['pct_of_time']:.1f}% | Avg Return: {stats['avg_return']:.2f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # =====================================================================
            # VALIDATION SECTION
            # =====================================================================
            st.markdown("---")
            st.markdown("#### 🧪 Model Validation")
            
            # Validation button
            val_key = f"validation_{st.session_state.selected_asset}"
            if st.button("📋 Validate Against Known Events", key="validate_btn", help="Test if the model correctly identifies historical crashes and rallies"):
                with st.spinner("Running validation..."):
                    try:
                        # Get the full regime DataFrame from cache
                        df_for_hmm = df.copy()
                        df_for_hmm['Date'] = pd.to_datetime(df_for_hmm['Date'])
                        df_json = df_for_hmm.to_json()
                        result = compute_hmm_regimes(selected_asset, df_json)
                        
                        # Reconstruct DataFrame for validation
                        regime_df = df_for_hmm.copy()
                        regime_result = detect_regimes(regime_df)
                        
                        # Run validation
                        validation_results = validate_regime_detection(regime_result['df'])
                        st.session_state[val_key] = validation_results
                    except Exception as e:
                        st.error(f"Validation failed: {str(e)}")
                        st.session_state[val_key] = None
            
            # Display validation results if available
            if val_key in st.session_state and st.session_state[val_key] is not None:
                val = st.session_state[val_key]
                summary = val['summary']
                
                # Grade display
                grade = summary.get('grade', 'N/A')
                grade_color = "#00ff88" if grade in ['A', 'B'] else "#ffc107" if grade == 'C' else "#ff4444"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1a2f4a 0%, #2d4a6a 100%);
                            border-radius: 10px; padding: 1rem; margin: 0.5rem 0;
                            border: 2px solid {grade_color}; text-align: center;">
                    <div style="color: #a0c4e8; font-size: 0.7rem;">VALIDATION GRADE</div>
                    <div style="color: {grade_color}; font-size: 2rem; font-weight: bold;">{grade}</div>
                    <div style="color: #888; font-size: 0.7rem;">
                        {summary.get('pass_rate', 0)}% pass rate | {summary.get('avg_accuracy', 0)}% avg accuracy
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Event results in expander
                with st.expander("📊 Detailed Results"):
                    st.markdown("**Bearish Events:**")
                    for event in val['bearish_events']:
                        status = "✅" if event['status'] == 'PASS' else "❌" if event['status'] == 'FAIL' else "⚠️"
                        st.markdown(f"{status} **{event['event']}**: {event['accuracy']}% ({event['days_correct']}/{event['days_total']} days)")
                    
                    st.markdown("**Bullish Events:**")
                    for event in val['bullish_events']:
                        status = "✅" if event['status'] == 'PASS' else "❌" if event['status'] == 'FAIL' else "⚠️"
                        st.markdown(f"{status} **{event['event']}**: {event['accuracy']}% ({event['days_correct']}/{event['days_total']} days)")
        
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
                        border-radius: 10px; padding: 1.2rem; text-align: center;
                        border: 1px dashed #4a9eff;">
                <div style="color: #4a9eff; font-size: 2rem; margin-bottom: 0.5rem;">🔍</div>
                <div style="color: #a0c4e8; font-size: 0.9rem;">
                    Click <b>"Run Regime Detection"</b> above to analyze market trends using Hidden Markov Model.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Show legend even before running
            st.markdown("#### 📊 Chart Legend")
            st.markdown("""
            <div style="font-size: 0.85rem;">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <div style="width: 20px; height: 20px; background: rgba(0, 255, 136, 0.3); margin-right: 8px; border-radius: 3px;"></div>
                    <span style="color: #ccc;">Bullish</span>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <div style="width: 20px; height: 20px; background: rgba(128, 128, 128, 0.3); margin-right: 8px; border-radius: 3px;"></div>
                    <span style="color: #ccc;">Neutral</span>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <div style="width: 20px; height: 20px; background: rgba(255, 68, 68, 0.3); margin-right: 8px; border-radius: 3px;"></div>
                    <span style="color: #ccc;">Bearish</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Initialize session state
    if 'selected_asset' not in st.session_state:
        st.session_state['selected_asset'] = None
    if 'start_date' not in st.session_state:
        st.session_state['start_date'] = datetime.now().date() - timedelta(days=365)
    if 'end_date' not in st.session_state:
        st.session_state['end_date'] = datetime.now().date()
    if 'active_tab' not in st.session_state:
        st.session_state['active_tab'] = 0
    
    # Render sidebar
    selected_asset, start_date, end_date = render_sidebar()
    
    # Check if we should show asset analysis (asset was selected)
    if st.session_state.get('selected_asset') and st.session_state.get('active_tab') == 1:
        # Show Asset Analysis directly without tabs when asset is selected
        render_asset_analysis()
        
        # Add button to go back to Market Overview
        st.markdown("---")
        if st.button("⬅️ Back to Market Overview", use_container_width=False):
            st.session_state['active_tab'] = 0
            st.rerun()
    else:
        # Show Market Overview
        render_market_overview()


# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()
