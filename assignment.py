
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Fixed URL - use raw GitHub URL instead of blob URL
BASE_URL = "https://raw.githubusercontent.com/siddharthyashkukreja-cloud/QuantInvesting/35528ac1dde5d2d873e1adfd0864381326d3b768"

# Load the data
df = pd.read_csv(f"{BASE_URL}/predictor_data.csv")

print("Data loaded successfully!")
print(f"Data shape: {df.shape}")
print("Columns:", df.columns.tolist())

# =============================================================================
# DATA PREPARATION
# =============================================================================

# Map column names to assignment variable names
predictor_mapping = {
    'logDP': 'D/P',
    'logDY': 'D/Y', 
    'logEP': 'E/P',
    'logDE': 'D/E',
    'svar': 'SVAR',
    'b/m': 'B/M',
    'ntis': 'NTIS',
    'tbl': 'TBL',
    'lty': 'LTY',
    'ltr': 'LTR',
    'tms': 'TMS',
    'dfy': 'DFY',
    'dfr': 'DFR',
    'lagINFL': 'INFL'
}

# Create working dataset
data = df.copy()

# Apply negative transformation for specified variables (as per assignment)
variables_to_negate = ['ntis', 'tbl', 'lty', 'lagINFL']
for var in variables_to_negate:
    data[var] = -data[var]

print("Applied negative transformation to:", variables_to_negate)

predictor_vars = list(predictor_mapping.keys())
dependent_var = 'r'

# =============================================================================
# PART 1: IN-SAMPLE ESTIMATION
# =============================================================================

print("\n" + "="*60)
print("PART 1: IN-SAMPLE ESTIMATION")
print("="*60)

Y = data['r'].values
results_dict = {}
in_sample_results = []

for var in predictor_vars:
    X = data[var].values
    X = sm.add_constant(X)

    # OLS regression with HAC standard errors
    model = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

    # Extract results
    alpha = model.params[0]
    beta = model.params[1]
    t_stat = model.tvalues[1]
    p_value = model.pvalues[1]

    # One-sided p-value (H0: β = 0 vs HA: β > 0)
    one_sided_p = p_value / 2 if t_stat > 0 else 1 - (p_value / 2)

    adj_r2 = model.rsquared_adj

    result = {
        'Variable': predictor_mapping[var],
        'Code': var,
        'Alpha': alpha,
        'Beta': beta,
        'T-stat': t_stat,
        'P-value (two-sided)': p_value,
        'P-value (one-sided)': one_sided_p,
        'Adj R²': adj_r2
    }

    in_sample_results.append(result)
    results_dict[var] = model

in_sample_df = pd.DataFrame(in_sample_results)

print("In-sample regression results:")
print(in_sample_df[['Variable', 'Beta', 'T-stat', 'P-value (one-sided)', 'Adj R²']].to_string(index=False))


# =============================================================================
# PART 2: OUT-OF-SAMPLE ESTIMATION (MODIFIED TO STORE HOLDOUT FORECASTS)
# =============================================================================

print("\n" + "="*60)
print("PART 2: OUT-OF-SAMPLE ESTIMATION (Storing Holdout Forecasts for DMSPE)")
print("="*60)

# Sample splits
total_obs = len(data)
m = 80  # 20 years * 4 quarters
p = 40  # 10 years * 4 quarters
q = total_obs - m - p

print(f"Sample split: m={m}, p={p}, q={q}, total={total_obs}")

# Generate out-of-sample forecasts for each predictor (including holdout period)
Y_full = data['r'].values
forecasts_dict = {}
# Store forecasts for holdout period (m+1 to m+p) and out-of-sample period (m+p+1 to T)
holdout_forecasts_dict = {} # To store forecasts for periods m+1 to m+p (used for DMSPE weights)
oos_forecasts_dict = {}     # To store forecasts for periods m+p+1 to T (used for final R2)

for var in predictor_vars:
    X_full = data[var].values
    holdout_individual_forecasts = []
    oos_individual_forecasts = []

    # Generate forecasts for holdout period (m+1 to m+p, i.e., index m to m+p-1)
    for t in range(m, m + p):
        Y_train = Y_full[1:t] # Data up to t-1
        X_train = X_full[:t-1] # Predictor lagged by 1, so up to t-1
        X_train = sm.add_constant(X_train)

        if len(Y_train) == len(X_train[:, 1]): # Ensure lengths match
            model = sm.OLS(Y_train, X_train).fit()
            X_forecast = np.array([1, X_full[t-1]]) # Predictor at time t-1 predicts r_t
            forecast = model.predict(X_forecast)[0]
            holdout_individual_forecasts.append(forecast)
        else:
            print(f"Length mismatch for {var} at t={t}: Y_train={len(Y_train)}, X_train={len(X_train[:, 1])}")
            # Handle potential mismatch, maybe append NaN or skip, but need consistent indexing
            # For now, append a placeholder if lengths don't match, though this indicates an issue
            holdout_individual_forecasts.append(np.nan) 

    # Generate forecasts for out-of-sample period (m+p+1 to T, i.e., index m+p to T-1)
    for t in range(m + p, total_obs - 1): # Forecast for t+1, so loop until T-2 (index T-1)
        Y_train = Y_full[1:t+1] # Data up to t
        X_train = X_full[:t] # Predictor up to t-1 (lags by 1)
        X_train = sm.add_constant(X_train)

        if len(Y_train) == len(X_train[:, 1]): # Ensure lengths match
            model = sm.OLS(Y_train, X_train).fit()
            X_forecast = np.array([1, X_full[t]]) # Predictor at time t predicts r_t+1
            forecast = model.predict(X_forecast)[0]
            oos_individual_forecasts.append(forecast)
        else:
            print(f"Length mismatch for {var} at t={t}: Y_train={len(Y_train)}, X_train={len(X_train[:, 1])}")
            oos_individual_forecasts.append(np.nan)

    holdout_forecasts_dict[var] = holdout_individual_forecasts
    oos_forecasts_dict[var] = oos_individual_forecasts

# Historical average benchmark for Out-of-Sample Period
benchmark_forecasts_oos = []
for t in range(m + p, total_obs - 1): # t from m+p to T-2, forecast for t+1 (m+p+1 to T-1)
    hist_avg = np.mean(Y_full[1:t+1]) # Average from start to time t
    benchmark_forecasts_oos.append(hist_avg)

# Actual returns for Out-of-Sample Period
actual_returns_oos = Y_full[m + p + 1:total_obs] # r from m+p+1 to T
mspe_benchmark_oos = np.mean((actual_returns_oos - benchmark_forecasts_oos)**2)

# Calculate out-of-sample R² (for Part 2)
oos_r2_results = []
for var in predictor_vars:
    forecasts = oos_forecasts_dict[var] # Use OOS forecasts
    # Check for NaNs and handle if necessary
    if np.isnan(forecasts).any():
         print(f"Warning: NaN found in OOS forecasts for {var}")
    mspe_model = np.mean((actual_returns_oos - np.array(forecasts))**2)
    oos_r2 = 1 - (mspe_model / mspe_benchmark_oos)

    result = {
        'Variable': predictor_mapping[var],
        'Code': var,
        'OOS R²': oos_r2,
        'MSPE Model': mspe_model,
        'Outperforms': oos_r2 > 0
    }
    oos_r2_results.append(result)

oos_df = pd.DataFrame(oos_r2_results)
print("Out-of-sample R² results:")
print(oos_df[['Variable', 'OOS R²', 'Outperforms']].sort_values('OOS R²', ascending=False).to_string(index=False))


# =============================================================================
# PART 3: KITCHEN SINK REGRESSION
# =============================================================================

print("\n" + "="*60)
print("PART 3: KITCHEN SINK REGRESSION")
print("="*60)

kitchen_sink_forecasts = []
for t in range(m + p, total_obs - 1): # Forecast for t+1, loop until T-2
    Y_train = Y_full[1:t+1] # Use data up to t
    X_train = data[predictor_vars].iloc[:t].values # Use predictor data up to t-1
    X_train = sm.add_constant(X_train)

    model = sm.OLS(Y_train, X_train).fit()
    X_forecast = np.concatenate([[1], data[predictor_vars].iloc[t].values]) # Use predictor data at t
    forecast = model.predict(X_forecast)[0]
    kitchen_sink_forecasts.append(forecast)

mspe_kitchen_sink = np.mean((actual_returns_oos - kitchen_sink_forecasts)**2) # Use OOS actuals
oos_r2_kitchen_sink = 1 - (mspe_kitchen_sink / mspe_benchmark_oos) # Use OOS benchmark MSPE

print(f"Kitchen Sink OOS R²: {oos_r2_kitchen_sink:.4f}")
print(f"Outperforms benchmark: {oos_r2_kitchen_sink > 0}")


# =============================================================================
# PART 4: FORECAST COMBINATION
# =============================================================================

print("\n" + "="*60)
print("PART 4: FORECAST COMBINATION")
print("="*60)

# Mean combination (OOS forecasts)
mean_combination_forecasts = []
for t_idx in range(len(actual_returns_oos)): # Iterate over OOS period length
    individual_forecasts_t = [oos_forecasts_dict[var][t_idx] for var in predictor_vars]
    # Check for NaNs in individual forecasts
    individual_forecasts_t = [f for f in individual_forecasts_t if not np.isnan(f)]
    if len(individual_forecasts_t) == len(predictor_vars): # All forecasts available
        mean_forecast = np.mean(individual_forecasts_t)
    else:
        print(f"Warning: NaN found in individual forecasts at OOS index {t_idx}")
        # Decide how to handle: skip, use available, etc. For now, assume all present or handle gracefully if not.
        mean_forecast = np.nanmean(individual_forecasts_t) # Use nanmean if some are NaN
    mean_combination_forecasts.append(mean_forecast)

mspe_mean_combo = np.mean((actual_returns_oos - np.array(mean_combination_forecasts))**2)
oos_r2_mean_combo = 1 - (mspe_mean_combo / mspe_benchmark_oos)

# Median combination (OOS forecasts)
median_combination_forecasts = []
for t_idx in range(len(actual_returns_oos)): # Iterate over OOS period length
    individual_forecasts_t = [oos_forecasts_dict[var][t_idx] for var in predictor_vars]
    individual_forecasts_t = [f for f in individual_forecasts_t if not np.isnan(f)]
    if len(individual_forecasts_t) == len(predictor_vars):
        median_forecast = np.median(individual_forecasts_t)
    else:
        print(f"Warning: NaN found in individual forecasts at OOS index {t_idx}")
        median_forecast = np.nanmedian(individual_forecasts_t)
    median_combination_forecasts.append(median_forecast)

mspe_median_combo = np.mean((actual_returns_oos - np.array(median_combination_forecasts))**2)
oos_r2_median_combo = 1 - (mspe_median_combo / mspe_benchmark_oos)

# DMSPE combination
def calculate_dmspe_weights_fixed(holdout_forecasts_dict, actual_returns_holdout, oos_forecasts_dict, actual_returns_oos, theta):
    """
    Calculate DMSPE weights based on holdout period performance and apply to OOS period.
    """
    n_vars = len(predictor_vars)
    n_oos = len(actual_returns_oos)

    # Calculate phi (cumulative discounted MSPE) for each variable using HOLDOUT data
    phi_values_holdout_end = []
    for var in predictor_vars:
        phi = 0.0
        # Iterate through the holdout period forecasts (index 0 to p-1)
        for s_idx in range(len(actual_returns_holdout)):
            actual_ret_holdout = actual_returns_holdout[s_idx]
            forecast_ret_holdout = holdout_forecasts_dict[var][s_idx]
            # Use the holdout period index s_idx, where s_idx=0 corresponds to time m+1, etc.
            # The discount factor should apply relative to the end of the holdout period.
            # Stock & Watson often calculate phi at the *end* of the estimation/holdout window.
            # For weights used *after* the holdout (for OOS), phi is calculated using all holdout errors.
            # phi_i = sum_{s=m+1}^{m+p} theta^{(m+p)-s} * (r_s - r_hat_i,s)^2
            # phi_i = sum_{k=0}^{p-1} theta^{p-1-k} * (r_{m+1+k} - r_hat_i,m+1+k)^2
            # Using s_idx = k (0 to p-1), time is m+1+s_idx
            # phi_i = sum_{s_idx=0}^{p-1} theta^{p-1-s_idx} * (r_{m+1+s_idx} - r_hat_i,m+1+s_idx)^2
            discount_power = len(actual_returns_holdout) - 1 - s_idx # p - 1 - s_idx
            phi += (theta ** discount_power) * (actual_ret_holdout - forecast_ret_holdout) ** 2
        phi_values_holdout_end.append(phi + 1e-8) # Add small value to avoid division by zero

    phi_values_holdout_end = np.array(phi_values_holdout_end)
    # Calculate initial weights based on holdout performance (used for first OOS forecast and potentially updated)
    inv_phi = 1.0 / phi_values_holdout_end
    initial_weights = inv_phi / np.sum(inv_phi)

    # Apply the initial weights (calculated from holdout) to the entire OOS period
    # This is the standard interpretation: weights are fixed based on holdout performance.
    dmspe_forecasts = []
    for t_idx in range(n_oos):
        individual_forecasts_t = [oos_forecasts_dict[var][t_idx] for var in predictor_vars]
        combined_forecast = np.sum(initial_weights * np.array(individual_forecasts_t))
        dmspe_forecasts.append(combined_forecast)

    return dmspe_forecasts, initial_weights # Return weights for potential inspection


# Actual returns for Holdout Period (used for calculating DMSPE weights)
actual_returns_holdout = Y_full[m + 1 : m + p + 1] # r from m+1 to m+p

# DMSPE for θ = 0.9 and θ = 1.0
dmspe_results = []
for theta in [0.9, 1.0]:
    dmspe_forecasts, weights = calculate_dmspe_weights_fixed(holdout_forecasts_dict, actual_returns_holdout, oos_forecasts_dict, actual_returns_oos, theta)
    mspe_dmspe = np.mean((actual_returns_oos - np.array(dmspe_forecasts))**2)
    oos_r2_dmspe = 1 - (mspe_dmspe / mspe_benchmark_oos)
    dmspe_results.append({
        'Method': f'DMSPE (θ={theta})',
        'OOS R²': oos_r2_dmspe,
        'Weights': weights # Optional: store weights
    })
    print(f"DMSPE θ={theta}: OOS R² = {oos_r2_dmspe:.4f}")

print(f"Mean combination: OOS R² = {oos_r2_mean_combo:.4f}")
print(f"Median combination: OOS R² = {oos_r2_median_combo:.4f}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

best_individual_idx = oos_df['OOS R²'].idxmax()
print(f"Best individual predictor: {oos_df.loc[best_individual_idx, 'Variable']} ({oos_df.loc[best_individual_idx, 'OOS R²']:.4f})")
print(f"Kitchen sink: {oos_r2_kitchen_sink:.4f}")
print(f"Mean combination: {oos_r2_mean_combo:.4f}")
print(f"Median combination: {oos_r2_median_combo:.4f}")
# Print DMSPE results
for res in dmspe_results:
    print(f"{res['Method']}: {res['OOS R²']:.4f}")

print("\nCompleted successfully!")
