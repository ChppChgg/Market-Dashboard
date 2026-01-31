"""
HMMRegime.py - Hidden Markov Model Market Regime Detection
Detects market regimes (Bullish, Neutral, Bearish) and volatility states using HMM.

Key improvements for smooth regime capture:
- Longer rolling windows to capture trends, not noise
- Regime smoothing to prevent rapid switching
- Minimum regime duration filtering
"""

import warnings
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional

# Suppress HMM convergence warnings (model reaches local optimum, which is fine)
warnings.filterwarnings('ignore', message='.*Model is not converging.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)


# =============================================================================
# CONFIGURATION - ADJUST THESE FOR DIFFERENT BEHAVIOR
# =============================================================================

# Rolling window sizes (in trading days)
SHORT_WINDOW = 10      # ~2 weeks - for short-term volatility
LONG_WINDOW = 50       # ~2.5 months - for trend detection

# Regime smoothing: minimum consecutive days before regime change is confirmed
# Higher = fewer regime switches, cleaner zones
MIN_REGIME_DURATION = 15  # ~3 weeks minimum per regime (was 5)

# HMM settings
N_STATES = 3           # Bullish, Neutral, Bearish
N_ITERATIONS = 500     # Max iterations (will stop early if converged)
CONVERGENCE_TOL = 1e-4 # Stop when log-likelihood improvement < this
RANDOM_STATE = 42      # For reproducibility

# Rolling Window Training settings - TUNED FOR RESPONSIVENESS
TRAINING_WINDOW = 504  # 2 years (was 3) - more responsive to recent conditions
STEP_SIZE = 21         # Monthly retraining (was quarterly) - faster adaptation
MIN_TRAINING_SAMPLES = 126  # 6 months minimum (was 1 year) - can react faster

# Reality Check thresholds (override HMM when it disagrees with actual price action)
REALITY_CHECK_WINDOW = 40     # Days to look back (shorter = more responsive)
BULLISH_THRESHOLD = 0.03      # +3% = clearly bullish (was 5%, now more sensitive)
BEARISH_THRESHOLD = -0.03     # -3% = clearly bearish (was 5%)
CRASH_THRESHOLD = -0.08       # -8% in 20 days = crash (was 10%)
RALLY_THRESHOLD = 0.05        # +5% in 20 days = strong rally (was 8%)


# =============================================================================
# REGIME LABELS
# =============================================================================

TREND_LABELS = {
    'bullish': 'Bullish',
    'neutral': 'Neutral',
    'bearish': 'Bearish'
}

VOLATILITY_LABELS = {
    'low': 'Low Volatility',
    'medium': 'Medium Volatility',
    'high': 'High Volatility'
}

REGIME_COLORS = {
    'bullish': 'rgba(0, 255, 136, 0.35)',      # Green
    'neutral': 'rgba(128, 128, 128, 0.25)',   # Grey
    'bearish': 'rgba(255, 68, 68, 0.35)'      # Red
}

VOLATILITY_COLORS = {
    'low': '#00ff88',      # Green
    'medium': '#ffc107',   # Yellow/Orange
    'high': '#ff4444'      # Red
}


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

# Normalization window for rolling z-scores (6 months - more responsive)
NORMALIZATION_WINDOW = 126


def rolling_zscore(series: pd.Series, window: int = NORMALIZATION_WINDOW) -> pd.Series:
    """
    Compute rolling z-score for a series.
    
    This normalizes each value relative to its recent history, making the
    features adaptive to changing market conditions. A -3 sigma event in 2008
    will be measured against 2007-2008 volatility, not 2020 volatility.
    
    Args:
        series: The input series to normalize
        window: Lookback window for computing mean/std
    
    Returns:
        Rolling z-score series
    """
    rolling_mean = series.rolling(window=window, min_periods=window//2).mean()
    rolling_std = series.rolling(window=window, min_periods=window//2).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    
    return (series - rolling_mean) / rolling_std


def compute_features(df: pd.DataFrame, 
                     short_window: int = None, 
                     long_window: int = None,
                     use_rolling_zscore: bool = True) -> pd.DataFrame:
    """
    Compute features for HMM regime detection.
    
    Key improvement: Rolling z-score normalization
    Instead of global standardization, features are normalized within rolling
    windows. This makes the model adaptive to changing market regimes.
    
    Example: A 5% daily drop in 2008 (when vol was 80%) is less extreme
    than a 5% drop in 2019 (when vol was 12%). Rolling z-scores capture this.
    
    Features:
    - Log returns (more stable than simple returns)
    - Cumulative returns over window (captures trend direction)  
    - Rolling volatility (captures market uncertainty)
    - All features converted to rolling z-scores for adaptivity
    """
    if short_window is None:
        short_window = SHORT_WINDOW
    if long_window is None:
        long_window = LONG_WINDOW
        
    df = df.copy()
    
    # Ensure we have Close prices
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")
    
    # ==========================================================================
    # RAW FEATURES
    # ==========================================================================
    
    # Log returns (more normally distributed, better for HMM)
    df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Cumulative return over long window (trend indicator)
    df[f'cum_return_{long_window}'] = df['Return'].rolling(long_window).sum()
    
    # Rolling volatility (annualized)
    df[f'rolling_std_{long_window}'] = df['Return'].rolling(long_window).std() * np.sqrt(252)
    
    # Short-term volatility
    df[f'rolling_std_{short_window}'] = df['Return'].rolling(short_window).std() * np.sqrt(252)
    
    # Price momentum: how far price is from its moving average (%)
    df[f'ma_{long_window}'] = df['Close'].rolling(long_window).mean()
    df['price_momentum'] = (df['Close'] - df[f'ma_{long_window}']) / df[f'ma_{long_window}']
    
    # Trend strength: ratio of cumulative return to volatility (Sharpe-like)
    df['trend_strength'] = df[f'cum_return_{long_window}'] / (df[f'rolling_std_{long_window}'] + 0.001)
    
    # Keep legacy columns for compatibility
    df[f'rolling_mean_{long_window}'] = df['Return'].rolling(long_window).mean()
    
    # ==========================================================================
    # ROLLING Z-SCORE NORMALIZATION (Adaptive to market conditions)
    # ==========================================================================
    
    if use_rolling_zscore:
        # Normalize cumulative returns - a 20% gain means different things
        # in a volatile vs calm market
        df['cum_return_zscore'] = rolling_zscore(
            df[f'cum_return_{long_window}'], 
            NORMALIZATION_WINDOW
        )
        
        # Normalize volatility - helps detect "high vol for THIS era"
        df['volatility_zscore'] = rolling_zscore(
            df[f'rolling_std_{long_window}'], 
            NORMALIZATION_WINDOW
        )
        
        # Normalize momentum
        df['momentum_zscore'] = rolling_zscore(
            df['price_momentum'], 
            NORMALIZATION_WINDOW
        )
        
        # Normalize trend strength
        df['trend_strength_zscore'] = rolling_zscore(
            df['trend_strength'], 
            NORMALIZATION_WINDOW
        )
    
    return df


# =============================================================================
# REGIME SMOOTHING
# =============================================================================

def smooth_regimes(states: np.ndarray, min_duration: int = None) -> np.ndarray:
    """
    Smooth regime predictions to prevent rapid switching.
    
    If a regime lasts fewer than min_duration days, merge it with neighbors.
    This creates cleaner, more stable regime zones.
    """
    if min_duration is None:
        min_duration = MIN_REGIME_DURATION
    
    if min_duration <= 1:
        return states
    
    smoothed = states.copy()
    n = len(smoothed)
    
    # Find regime change points
    i = 0
    while i < n:
        # Find end of current regime
        j = i
        while j < n and smoothed[j] == smoothed[i]:
            j += 1
        
        regime_length = j - i
        
        # If regime is too short, merge with previous
        if regime_length < min_duration and i > 0:
            smoothed[i:j] = smoothed[i-1]
        
        i = j
    
    return smoothed


# =============================================================================
# HMM MODEL FITTING
# =============================================================================

def fit_hmm(df: pd.DataFrame,
            features: list = None,
            n_states: int = None,
            random_state: int = None,
            use_adaptive_features: bool = True) -> Tuple[pd.DataFrame, GaussianHMM, StandardScaler]:
    """
    Fit a Gaussian Hidden Markov Model to detect market regimes.
    
    Key improvement: Uses rolling z-score normalized features by default.
    This makes the model adaptive to changing market conditions across decades.
    
    Optimizations for speed:
    - 'diag' covariance: O(n) instead of O(n²), usually sufficient for finance
    - Early stopping: Stop when log-likelihood converges (tol parameter)
    - K-means initialization: Better starting point = faster convergence
    
    Args:
        df: DataFrame with OHLCV data
        features: List of feature columns (default: z-score normalized features)
        n_states: Number of hidden states
        random_state: Random seed
        use_adaptive_features: If True, use rolling z-score features (recommended)
    """
    if n_states is None:
        n_states = N_STATES
    if random_state is None:
        random_state = RANDOM_STATE
        
    # Use z-score normalized features for adaptive regime detection
    if features is None:
        if use_adaptive_features:
            # Z-score features adapt to local market conditions
            features = ['cum_return_zscore', 'volatility_zscore']
        else:
            # Raw features (less adaptive, may miss historical extremes)
            features = [f'cum_return_{LONG_WINDOW}', f'rolling_std_{LONG_WINDOW}']
    
    # Compute features if not present
    feature_col = features[0] if features else f'cum_return_{LONG_WINDOW}'
    if feature_col not in df.columns:
        df = compute_features(df)
    
    # Prepare data
    df_clean = df.dropna(subset=features).copy()
    X = df_clean[features].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize means using K-means for better starting point (faster convergence)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_states, random_state=random_state, n_init=10)
    kmeans.fit(X_scaled)
    initial_means = kmeans.cluster_centers_
    
    # Fit HMM with optimized settings
    model = GaussianHMM(
        n_components=n_states,
        covariance_type='diag',      # FASTER: O(n) vs O(n²) for 'full'
        n_iter=N_ITERATIONS,
        tol=CONVERGENCE_TOL,         # EARLY STOPPING: Stop when converged
        random_state=random_state,
        init_params='sc',            # Only init start probs and covariances
        params='stmc'                # Learn all parameters
    )
    
    # Set K-means initialized means (better starting point)
    model.means_ = initial_means
    
    # Initialize transition matrix to favor staying in same state (regime persistence)
    init_transmat = np.full((n_states, n_states), 0.05)
    np.fill_diagonal(init_transmat, 0.90)
    init_transmat = init_transmat / init_transmat.sum(axis=1, keepdims=True)
    model.transmat_ = init_transmat
    
    # Fit the model
    model.fit(X_scaled)
    
    # Predict hidden states
    hidden_states = model.predict(X_scaled)
    
    # Apply smoothing to remove short-duration regimes
    hidden_states = smooth_regimes(hidden_states, MIN_REGIME_DURATION)
    
    # Add states to dataframe
    df = df.copy()
    df.loc[df_clean.index, 'HMM_State'] = hidden_states
    
    return df, model, scaler


# =============================================================================
# ROLLING WINDOW TRAINING (Adaptive HMM)
# =============================================================================

def fit_hmm_rolling(df: pd.DataFrame,
                    features: list = None,
                    n_states: int = None,
                    training_window: int = None,
                    step_size: int = None,
                    random_state: int = None) -> pd.DataFrame:
    """
    Fit HMM using rolling window training for adaptive regime detection.
    
    Instead of training one model on all data (which averages across market eras),
    this trains separate models on rolling windows. Each window learns parameters
    relevant to that era, making regime detection adaptive to changing conditions.
    
    Process:
    1. Start at position = training_window
    2. Train HMM on data[position - training_window : position]
    3. Predict regime for data[position : position + step_size]
    4. Move forward by step_size, repeat
    
    This ensures:
    - 2008 predictions use 2005-2008 trained model (knows pre-crisis vol)
    - 2020 predictions use 2017-2020 trained model (knows low-vol era)
    - No lookahead bias - only uses past data for each prediction
    
    Args:
        df: DataFrame with computed features
        features: Feature columns to use
        n_states: Number of hidden states
        training_window: Days of data for each training window
        step_size: Days between model retraining
        random_state: Random seed
    
    Returns:
        DataFrame with HMM_State column populated via rolling predictions
    """
    if n_states is None:
        n_states = N_STATES
    if training_window is None:
        training_window = TRAINING_WINDOW
    if step_size is None:
        step_size = STEP_SIZE
    if random_state is None:
        random_state = RANDOM_STATE
    if features is None:
        features = ['cum_return_zscore', 'volatility_zscore']
    
    df = df.copy()
    
    # Get clean data (no NaN in features)
    valid_mask = df[features].notna().all(axis=1)
    valid_indices = df[valid_mask].index.tolist()
    
    if len(valid_indices) < training_window + step_size:
        raise ValueError(f"Not enough data for rolling window training. "
                        f"Need {training_window + step_size}, have {len(valid_indices)}")
    
    # Initialize HMM_State column
    df['HMM_State'] = np.nan
    
    # Track state mappings to ensure consistency across windows
    # We'll use cumulative return to determine which state is "bullish" etc.
    
    # Rolling window training
    position = training_window
    n_windows = 0
    
    while position < len(valid_indices):
        # Define training window
        train_start_idx = valid_indices[position - training_window]
        train_end_idx = valid_indices[position - 1]
        
        # Define prediction window
        pred_start_idx = valid_indices[position]
        pred_end_pos = min(position + step_size, len(valid_indices))
        pred_end_idx = valid_indices[pred_end_pos - 1]
        
        # Get training data
        train_mask = (df.index >= train_start_idx) & (df.index <= train_end_idx)
        train_df = df.loc[train_mask, features].dropna()
        
        if len(train_df) < MIN_TRAINING_SAMPLES:
            position += step_size
            continue
        
        X_train = train_df.values
        
        # Standardize using training data statistics
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Initialize with K-means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_states, random_state=random_state, n_init=10)
        kmeans.fit(X_train_scaled)
        
        # Fit HMM on training window
        model = GaussianHMM(
            n_components=n_states,
            covariance_type='diag',
            n_iter=N_ITERATIONS,
            tol=CONVERGENCE_TOL,
            random_state=random_state,
            init_params='sc',
            params='stmc'
        )
        model.means_ = kmeans.cluster_centers_
        
        # Biased transition matrix for regime persistence
        init_transmat = np.full((n_states, n_states), 0.05)
        np.fill_diagonal(init_transmat, 0.90)
        init_transmat = init_transmat / init_transmat.sum(axis=1, keepdims=True)
        model.transmat_ = init_transmat
        
        try:
            model.fit(X_train_scaled)
        except Exception:
            # If fitting fails, skip this window
            position += step_size
            continue
        
        # Get prediction data
        pred_mask = (df.index >= pred_start_idx) & (df.index <= pred_end_idx)
        pred_df = df.loc[pred_mask, features].dropna()
        
        if len(pred_df) == 0:
            position += step_size
            continue
        
        # Scale prediction data using training scaler
        X_pred_scaled = scaler.transform(pred_df.values)
        
        # Predict states for prediction window
        predicted_states = model.predict(X_pred_scaled)
        
        # Map states to consistent labels based on this window's statistics
        # Compute mean cumulative return for each state in training data
        train_states = model.predict(X_train_scaled)
        train_df_with_states = train_df.copy()
        train_df_with_states['state'] = train_states
        
        # We need the raw cumulative return column for labeling
        cum_ret_col = f'cum_return_{LONG_WINDOW}'
        if cum_ret_col in df.columns:
            train_df_with_states['cum_return'] = df.loc[train_df.index, cum_ret_col]
            state_means = train_df_with_states.groupby('state')['cum_return'].mean()
        else:
            # Fallback: use z-score feature
            state_means = train_df_with_states.groupby('state')['cum_return_zscore'].mean()
        
        # Sort states by mean return: lowest = 0 (bearish), highest = 2 (bullish)
        sorted_states = state_means.sort_values().index.tolist()
        
        # Create mapping from raw state to ordered state
        state_mapping = {old: new for new, old in enumerate(sorted_states)}
        
        # Apply mapping to predictions
        mapped_predictions = np.array([state_mapping.get(s, s) for s in predicted_states])
        
        # Store predictions
        df.loc[pred_df.index, 'HMM_State'] = mapped_predictions
        
        n_windows += 1
        position += step_size
    
    # Apply smoothing to final predictions
    valid_states = df['HMM_State'].dropna()
    if len(valid_states) > 0:
        smoothed = smooth_regimes(valid_states.values.astype(int), MIN_REGIME_DURATION)
        df.loc[valid_states.index, 'HMM_State'] = smoothed
    
    return df


# =============================================================================
# REGIME LABELING
# =============================================================================

# =============================================================================
# MAIN DETECTION FUNCTION
# =============================================================================

def detect_regimes(df: pd.DataFrame, 
                   n_states: int = None,
                   features: list = None,
                   use_adaptive_features: bool = True,
                   use_rolling_window: bool = True) -> Dict:
    """
    Main function to detect market regimes using HMM.
    
    Two key improvements for adaptive regime detection:
    1. Rolling z-score normalization: Features are normalized relative to recent
       history, so 2008 extremes are measured against 2007-2008 context.
    2. Rolling window training: Instead of one model for all time, trains 
       separate models on rolling windows so each era uses relevant parameters.
    
    Args:
        df: DataFrame with OHLCV data (must have 'Close' column)
        n_states: Number of regimes to detect (default: N_STATES from config)
        features: Features to use for HMM (default: z-score normalized)
        use_adaptive_features: If True, use rolling z-score features
        use_rolling_window: If True, use rolling window training (recommended)
    
    Returns:
        Dictionary containing:
        - df: DataFrame with regime labels
        - model: Fitted HMM model (None if using rolling window)
        - scaler: Fitted StandardScaler (None if using rolling window)
        - current_trend: Current trend regime label
        - current_volatility: Current volatility regime label
        - state_stats: Statistics for each state
        - regime_history: List of regime changes
    """
    if n_states is None:
        n_states = N_STATES
    
    print(f"[detect_regimes] Starting with {len(df)} rows")
    
    # Use z-score normalized features by default for adaptivity
    if features is None:
        if use_adaptive_features:
            features = ['cum_return_zscore', 'volatility_zscore']
        else:
            features = [f'cum_return_{LONG_WINDOW}', f'rolling_std_{LONG_WINDOW}']
    
    # Compute features (including z-score normalization)
    df = compute_features(df, use_rolling_zscore=use_adaptive_features)
    
    # Check if we have enough data for rolling window
    min_required = TRAINING_WINDOW + STEP_SIZE + NORMALIZATION_WINDOW
    
    # Choose training method
    if use_rolling_window and len(df) >= min_required:
        # Rolling window training - adaptive to market eras
        print(f"[detect_regimes] Using rolling window training (data: {len(df)}, required: {min_required})")
        df = fit_hmm_rolling(df, features=features, n_states=n_states)
        model = None  # No single model when using rolling windows
        scaler = None
    else:
        # Single model training - fallback for shorter datasets
        print(f"[detect_regimes] Using single model training (data: {len(df)}, rolling requires: {min_required})")
        df, model, scaler = fit_hmm(df, features=features, n_states=n_states, 
                                     use_adaptive_features=use_adaptive_features)
    
    # Check if we got HMM states
    if 'HMM_State' not in df.columns or df['HMM_State'].dropna().empty:
        print("[detect_regimes] WARNING: No HMM states generated!")
        # Create fallback based on simple returns
        df['Trend_Regime'] = 'neutral'
        df['Volatility_Regime'] = 'medium'
        return {
            'df': df,
            'model': None,
            'scaler': None,
            'current_trend': 'neutral',
            'current_volatility': 'medium',
            'regime_stats': {},
            'regime_history': []
        }
    
    # Label regimes based on state statistics
    df = label_regimes_from_states(df, n_states)
    
    # Check if labeling worked
    trend_count = df['Trend_Regime'].dropna().shape[0]
    print(f"[detect_regimes] Labeled {trend_count} rows with Trend_Regime")
    
    # Get current regime (most recent)
    valid_regimes = df.dropna(subset=['Trend_Regime'])
    if len(valid_regimes) == 0:
        print("[detect_regimes] WARNING: No valid regimes after labeling!")
        return {
            'df': df,
            'model': model if 'model' in dir() else None,
            'scaler': scaler if 'scaler' in dir() else None,
            'current_trend': 'neutral',
            'current_volatility': 'medium',
            'regime_stats': {},
            'regime_history': []
        }
    
    latest = valid_regimes.iloc[-1]
    current_trend = latest['Trend_Regime']
    current_volatility = latest['Volatility_Regime']
    
    # Calculate regime statistics
    regime_stats = calculate_regime_statistics(df)
    
    # Get regime change history
    regime_history = get_regime_changes(df)
    
    print(f"[detect_regimes] Complete - Current: {current_trend}/{current_volatility}")
    
    return {
        'df': df,
        'model': model,
        'scaler': scaler,
        'current_trend': current_trend,
        'current_volatility': current_volatility,
        'regime_stats': regime_stats,
        'regime_history': regime_history
    }


def label_regimes_from_states(df: pd.DataFrame, n_states: int = 3) -> pd.DataFrame:
    """
    Label HMM states as Bullish/Neutral/Bearish based on actual returns in each state.
    Works with both single-model and rolling-window trained states.
    
    INCLUDES PRICE REALITY CHECK:
    The HMM can get confused by volatility. We add a sanity check:
    - If price actually went UP recently, don't call it bearish
    - If price actually went DOWN recently, don't call it bullish
    
    This prevents the model from calling obvious bull markets "bearish"
    just because volatility is elevated.
    """
    df = df.copy()
    
    # Get data with valid states
    df_with_states = df.dropna(subset=['HMM_State']).copy()
    
    if len(df_with_states) == 0:
        df['Trend_Regime'] = np.nan
        df['Volatility_Regime'] = np.nan
        return df
    
    # Find volatility column
    vol_col = f'rolling_std_{LONG_WINDOW}'
    if vol_col not in df.columns:
        for col in df.columns:
            if col.startswith('rolling_std_') and not col.endswith('_zscore'):
                vol_col = col
                break
    
    # Compute mean return and volatility per state
    cum_ret_col = f'cum_return_{LONG_WINDOW}'
    
    state_stats = df_with_states.groupby('HMM_State').agg({
        'Return': 'mean',
        vol_col: 'mean' if vol_col in df.columns else 'count'
    })
    
    # Map states to trend labels (states are pre-ordered by rolling training)
    unique_states = sorted(df_with_states['HMM_State'].dropna().unique())
    
    if len(unique_states) >= 3:
        trend_mapping = {
            unique_states[0]: 'bearish',
            unique_states[1]: 'neutral', 
            unique_states[2]: 'bullish'
        }
    elif len(unique_states) == 2:
        trend_mapping = {
            unique_states[0]: 'bearish',
            unique_states[1]: 'bullish'
        }
    else:
        trend_mapping = {unique_states[0]: 'neutral'}
    
    # Map states to volatility labels based on actual volatility in each state
    if vol_col in state_stats.columns:
        sorted_by_vol = state_stats[vol_col].sort_values()
        vol_states = sorted_by_vol.index.tolist()
        
        if len(vol_states) >= 3:
            vol_mapping = {
                vol_states[0]: 'low',
                vol_states[1]: 'medium',
                vol_states[2]: 'high'
            }
        elif len(vol_states) == 2:
            vol_mapping = {
                vol_states[0]: 'low',
                vol_states[1]: 'high'
            }
        else:
            vol_mapping = {vol_states[0]: 'medium'}
    else:
        vol_mapping = {s: 'medium' for s in unique_states}
    
    # Apply base labels from HMM
    df['Trend_Regime'] = df['HMM_State'].map(trend_mapping)
    df['Volatility_Regime'] = df['HMM_State'].map(vol_mapping)
    
    # =========================================================================
    # PRICE REALITY CHECK - Override when HMM disagrees with actual price action
    # =========================================================================
    
    # Calculate actual rolling returns at multiple timeframes
    df['actual_return_40d'] = df['Close'].pct_change(REALITY_CHECK_WINDOW)
    df['actual_return_20d'] = df['Close'].pct_change(20)
    df['actual_return_60d'] = df['Close'].pct_change(60)  # Longer term confirmation
    
    # REALITY CHECK 1: If price went UP significantly, upgrade from bearish
    # More aggressive: upgrade to BULLISH if strong, NEUTRAL if moderate
    price_strongly_up = (
        (df['actual_return_40d'] > BULLISH_THRESHOLD * 2) &  # +6% = strong bull
        (df['actual_return_60d'] > BULLISH_THRESHOLD)        # Confirmed by 60d
    )
    price_moderately_up = (
        (df['actual_return_40d'] > BULLISH_THRESHOLD) &
        (df['Trend_Regime'] == 'bearish')
    )
    
    # REALITY CHECK 2: If price went DOWN significantly, downgrade from bullish
    price_strongly_down = (
        (df['actual_return_40d'] < BEARISH_THRESHOLD * 2) &  # -6% = strong bear
        (df['actual_return_60d'] < BEARISH_THRESHOLD)        # Confirmed by 60d
    )
    price_moderately_down = (
        (df['actual_return_40d'] < BEARISH_THRESHOLD) &
        (df['Trend_Regime'] == 'bullish')
    )
    
    # CRASH OVERRIDE: Fast crashes get flagged immediately
    crash_detected = (df['actual_return_20d'] < CRASH_THRESHOLD)
    
    # RALLY OVERRIDE: Fast rallies get flagged immediately  
    rally_detected = (df['actual_return_20d'] > RALLY_THRESHOLD)
    
    # Apply reality checks (order matters - most confident overrides last)
    
    # Moderate adjustments
    df.loc[price_moderately_up, 'Trend_Regime'] = 'neutral'     # Bearish → Neutral
    df.loc[price_moderately_down, 'Trend_Regime'] = 'neutral'   # Bullish → Neutral
    
    # Strong adjustments
    df.loc[price_strongly_up, 'Trend_Regime'] = 'bullish'       # Any → Bullish
    df.loc[price_strongly_down, 'Trend_Regime'] = 'bearish'     # Any → Bearish
    
    # Extreme short-term events (highest priority)
    df.loc[rally_detected, 'Trend_Regime'] = 'bullish'
    df.loc[crash_detected, 'Trend_Regime'] = 'bearish'
    df.loc[crash_detected, 'Volatility_Regime'] = 'high'
    
    # Count adjustments for diagnostics
    n_to_bullish = price_strongly_up.sum() + rally_detected.sum()
    n_to_bearish = price_strongly_down.sum() + crash_detected.sum()
    n_to_neutral = price_moderately_up.sum() + price_moderately_down.sum()
    
    if n_to_bullish > 0 or n_to_bearish > 0 or n_to_neutral > 0:
        print(f"[Reality Check] → Bullish: {n_to_bullish}, → Bearish: {n_to_bearish}, → Neutral: {n_to_neutral} days")
    
    # Clean up temporary columns
    df.drop(['actual_return_40d', 'actual_return_20d', 'actual_return_60d'], 
            axis=1, inplace=True, errors='ignore')
    
    return df


def calculate_regime_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate statistics for each regime.
    """
    df_clean = df.dropna(subset=['Trend_Regime'])
    
    stats = {}
    
    for regime in ['bullish', 'neutral', 'bearish']:
        regime_data = df_clean[df_clean['Trend_Regime'] == regime]
        
        if len(regime_data) > 0:
            stats[regime] = {
                'count': len(regime_data),
                'pct_of_time': len(regime_data) / len(df_clean) * 100,
                'avg_return': regime_data['Return'].mean() * 100,
                'avg_volatility': regime_data['rolling_std_50'].mean() * 100 if 'rolling_std_50' in regime_data else 0,
                'total_return': (1 + regime_data['Return']).prod() - 1,
            }
    
    return stats


def get_regime_changes(df: pd.DataFrame) -> list:
    """
    Get list of regime change events.
    """
    df_clean = df.dropna(subset=['Trend_Regime']).copy()
    df_clean['regime_change'] = df_clean['Trend_Regime'] != df_clean['Trend_Regime'].shift(1)
    
    changes = []
    change_points = df_clean[df_clean['regime_change']].index
    
    for idx in change_points[1:]:  # Skip first (not a change)
        changes.append({
            'date': idx,
            'new_regime': df_clean.loc[idx, 'Trend_Regime'],
            'price': df_clean.loc[idx, 'Close']
        })
    
    return changes[-10:]  # Return last 10 changes


# =============================================================================
# FORWARD-LOOKING REGIME SIGNALS
# =============================================================================

def calculate_regime_stress_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate forward-looking stress indicators that may signal regime changes
    BEFORE they happen. These are leading indicators, not lagging.
    
    Returns DataFrame with additional columns:
    - regime_stress: 0-100 score (higher = more likely regime is changing)
    - trend_weakening: True if current trend is losing momentum
    - volatility_expanding: True if vol is increasing (often precedes crashes)
    - breadth_divergence: Price vs momentum divergence
    
    IMPORTANT: These are probabilistic signals, not guarantees!
    """
    df = df.copy()
    
    # =========================================================================
    # 1. VOLATILITY EXPANSION (Leading indicator for crashes)
    # When volatility starts expanding from low levels, trouble often follows
    # =========================================================================
    
    vol_col = f'rolling_std_{LONG_WINDOW}'
    if vol_col in df.columns:
        # Compare current vol to recent average
        df['vol_sma_20'] = df[vol_col].rolling(20).mean()
        df['vol_expanding'] = df[vol_col] > df['vol_sma_20'] * 1.2  # Vol 20% above average
        
        # Rate of change in volatility
        df['vol_acceleration'] = df[vol_col].pct_change(5)  # 1-week vol change
        
    # =========================================================================
    # 2. MOMENTUM DIVERGENCE (Price going up but momentum slowing)
    # Classic warning sign of trend exhaustion
    # =========================================================================
    
    if 'Close' in df.columns:
        # Price making new highs?
        df['price_vs_20d_high'] = df['Close'] / df['Close'].rolling(20).max()
        
        # But is momentum (rate of change) slowing?
        df['momentum_10d'] = df['Close'].pct_change(10)
        df['momentum_20d'] = df['Close'].pct_change(20)
        
        # Divergence: price near highs but momentum declining
        df['momentum_weakening'] = (
            (df['price_vs_20d_high'] > 0.95) &  # Price near recent high
            (df['momentum_10d'] < df['momentum_20d'] * 0.5)  # Short momentum < long
        )
        
        # Inverse for bottoming signals
        df['price_vs_20d_low'] = df['Close'] / df['Close'].rolling(20).min()
        df['momentum_strengthening'] = (
            (df['price_vs_20d_low'] < 1.05) &  # Price near recent low
            (df['momentum_10d'] > df['momentum_20d'])  # Short momentum improving
        )
    
    # =========================================================================
    # 3. TREND STRENGTH (Is the current regime weakening?)
    # =========================================================================
    
    if 'Trend_Regime' in df.columns and 'Return' in df.columns:
        # Rolling returns over different windows
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_10d'] = df['Close'].pct_change(10)
        df['return_20d'] = df['Close'].pct_change(20)
        
        # In a bullish regime, are short-term returns turning negative?
        df['bull_weakening'] = (
            (df['Trend_Regime'] == 'bullish') &
            (df['return_5d'] < 0) &
            (df['return_10d'] < df['return_20d'])
        )
        
        # In a bearish regime, are short-term returns turning positive?
        df['bear_weakening'] = (
            (df['Trend_Regime'] == 'bearish') &
            (df['return_5d'] > 0) &
            (df['return_10d'] > df['return_20d'])
        )
        
        df['trend_weakening'] = df['bull_weakening'] | df['bear_weakening']
    
    # =========================================================================
    # 4. COMPOSITE REGIME STRESS SCORE (0-100)
    # Higher score = higher probability of regime change coming
    # =========================================================================
    
    stress_components = []
    
    if 'vol_expanding' in df.columns:
        stress_components.append(df['vol_expanding'].astype(float) * 25)
    
    if 'vol_acceleration' in df.columns:
        # High vol acceleration = stress
        vol_stress = (df['vol_acceleration'].clip(-0.5, 0.5) + 0.5) * 25
        stress_components.append(vol_stress)
    
    if 'momentum_weakening' in df.columns:
        stress_components.append(df['momentum_weakening'].astype(float) * 25)
    
    if 'trend_weakening' in df.columns:
        stress_components.append(df['trend_weakening'].astype(float) * 25)
    
    if stress_components:
        df['regime_stress'] = sum(stress_components).clip(0, 100)
    else:
        df['regime_stress'] = 50  # Neutral if we can't calculate
    
    # Clean up intermediate columns
    cleanup_cols = ['vol_sma_20', 'vol_acceleration', 'price_vs_20d_high', 
                    'price_vs_20d_low', 'return_5d', 'return_10d', 'return_20d',
                    'bull_weakening', 'bear_weakening']
    df.drop(cleanup_cols, axis=1, inplace=True, errors='ignore')
    
    return df


def get_current_regime_outlook(df: pd.DataFrame) -> dict:
    """
    Get a forward-looking assessment of the current regime.
    This is the key function for "what does TODAY's regime tell us?"
    
    Returns:
        Dictionary with:
        - current_regime: bullish/neutral/bearish
        - regime_age: how many days in current regime
        - stress_level: 0-100 (higher = regime more likely to change)
        - outlook: text description
        - signals: list of active warning/confirmation signals
    """
    df = df.dropna(subset=['Trend_Regime'])
    
    if len(df) == 0:
        return {'error': 'No regime data available'}
    
    # Get most recent data
    latest = df.iloc[-1]
    current_regime = latest['Trend_Regime']
    
    # Calculate regime age (days in current regime)
    regime_changes = df['Trend_Regime'] != df['Trend_Regime'].shift(1)
    last_change_idx = regime_changes[::-1].idxmax() if regime_changes.any() else df.index[0]
    regime_age = len(df.loc[last_change_idx:])
    
    # Get stress level
    stress = latest.get('regime_stress', 50)
    
    # Compile active signals
    signals = []
    
    if latest.get('vol_expanding', False):
        signals.append("⚠️ Volatility expanding - risk increasing")
    
    if latest.get('momentum_weakening', False):
        signals.append("⚠️ Momentum divergence - trend may be exhausting")
    
    if latest.get('momentum_strengthening', False):
        signals.append("📈 Momentum improving - potential bottom forming")
    
    if latest.get('trend_weakening', False):
        signals.append("⚠️ Current trend showing weakness")
    
    # Historical context: how long do regimes typically last?
    avg_regime_duration = df.groupby(
        (df['Trend_Regime'] != df['Trend_Regime'].shift()).cumsum()
    ).size().mean()
    
    # Generate outlook text
    if stress < 25:
        outlook = f"✅ {current_regime.title()} regime appears stable"
    elif stress < 50:
        outlook = f"⚡ {current_regime.title()} regime showing some stress signals"
    elif stress < 75:
        outlook = f"⚠️ {current_regime.title()} regime under pressure - watch for changes"
    else:
        outlook = f"🚨 High probability of regime change from {current_regime}"
    
    # Add regime age context
    if regime_age > avg_regime_duration * 1.5:
        signals.append(f"📊 Regime is {regime_age} days old (above average of {avg_regime_duration:.0f})")
    
    return {
        'current_regime': current_regime,
        'regime_age_days': regime_age,
        'avg_regime_duration': round(avg_regime_duration),
        'stress_level': round(stress, 1),
        'stress_interpretation': 'Low' if stress < 25 else 'Moderate' if stress < 50 else 'High' if stress < 75 else 'Critical',
        'outlook': outlook,
        'signals': signals,
        'vol_expanding': bool(latest.get('vol_expanding', False)),
        'trend_weakening': bool(latest.get('trend_weakening', False)),
    }


# =============================================================================
# REGIME ZONES FOR PLOTTING
# =============================================================================

def get_regime_zones(df: pd.DataFrame) -> list:
    """
    Get regime zones for chart overlay.
    Returns list of dictionaries with start, end dates, and regime type.
    Dates are stored as ISO strings for JSON serialization compatibility.
    """
    # Check if Trend_Regime exists
    if 'Trend_Regime' not in df.columns:
        print("[get_regime_zones] ERROR: No 'Trend_Regime' column found")
        return []
    
    df_clean = df.dropna(subset=['Trend_Regime']).copy()
    
    if len(df_clean) == 0:
        print("[get_regime_zones] WARNING: All Trend_Regime values are NaN")
        return []
    
    # Ensure we have a Date column
    if 'Date' not in df_clean.columns:
        # Try to use index if it's datetime
        if isinstance(df_clean.index, pd.DatetimeIndex):
            df_clean['Date'] = df_clean.index
        else:
            print("[get_regime_zones] ERROR: No 'Date' column and index is not datetime")
            return []
    
    # Ensure Date is datetime
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    
    zones = []
    current_regime = None
    zone_start_date = None
    
    for idx, row in df_clean.iterrows():
        regime = row['Trend_Regime']
        
        # Skip if regime is NaN or None
        if pd.isna(regime):
            continue
            
        if regime != current_regime:
            # Close previous zone
            if current_regime is not None and zone_start_date is not None:
                zones.append({
                    'start': zone_start_date.isoformat(),  # Store as ISO string
                    'end': row['Date'].isoformat(),        # Store as ISO string
                    'regime': current_regime,
                    'color': REGIME_COLORS.get(current_regime, 'rgba(128, 128, 128, 0.2)')
                })
            
            # Start new zone
            current_regime = regime
            zone_start_date = row['Date']
    
    # Close final zone
    if current_regime is not None and zone_start_date is not None:
        zones.append({
            'start': zone_start_date.isoformat(),              # Store as ISO string
            'end': df_clean.iloc[-1]['Date'].isoformat(),      # Store as ISO string
            'regime': current_regime,
            'color': REGIME_COLORS.get(current_regime, 'rgba(128, 128, 128, 0.2)')
        })
    
    print(f"[get_regime_zones] Generated {len(zones)} zones from {len(df_clean)} data points")
    
    return zones


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_regime_color(regime: str) -> str:
    """Get color for a regime."""
    return REGIME_COLORS.get(regime, 'rgba(128, 128, 128, 0.2)')


def get_volatility_color(volatility: str) -> str:
    """Get color for volatility level."""
    return VOLATILITY_COLORS.get(volatility, '#ffc107')


def get_trend_label(regime: str) -> str:
    """Get display label for trend regime."""
    return TREND_LABELS.get(regime, 'Unknown')


def get_volatility_label(volatility: str) -> str:
    """Get display label for volatility regime."""
    return VOLATILITY_LABELS.get(volatility, 'Unknown')
