import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Fixed URL - use raw GitHub URL instead of blob URL
# Note: Ensure there are no trailing spaces in the URL
BASE_URL = "https://raw.githubusercontent.com/siddharthyashkukreja-cloud/QuantInvesting/35528ac1dde5d2d873e1adfd0864381326d3b768/"

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
# Note: Assignment mentions NTIOS, but likely means NTIS based on list and common usage
variables_to_negate = ['ntis', 'tbl', 'lty', 'lagINFL']
for var in variables_to_negate:
    data[var] = -data[var]

print("Applied negative transformation to:", variables_to_negate)

predictor_vars = list(predictor_mapping.keys())
dependent_var = 'r'

# =============================================================================
# PART 0: PACF TEST TO DECIDE LAG (Optional Visualization)
# =============================================================================

print("\n" + "="*60)
print("PART 0: PACF TEST TO DECIDE LAG")
print("="*60)

lags = 20  # Number of lags to consider
plt.figure(figsize=(10, 6))
plot_pacf(data['r'], lags=lags, method='ywm', alpha=0.05)
plt.title("Partial Autocorrelation Function (PACF) for 'r'")
plt.xlabel("Lags")
plt.ylabel("PACF")
plt.grid()
plt.show()
print("PACF plot displayed. Use this to decide the appropriate lag if needed for other analyses.")

# =============================================================================
# PART 1: IN-SAMPLE ESTIMATION (CORRECTED: rt+1 on xi,t)
# =============================================================================

print("\n" + "="*60)
print("PART 1: IN-SAMPLE ESTIMATION (CORRECTED: rt+1 on xi,t)")
print("="*60)

Y_full = data['r'].values # Store full series for later use
results_dict = {}
in_sample_results = []

for var in predictor_vars:
    X_full = data[var].values # Store full predictor series

    # --- CORRECT ALIGNMENT FOR REGRESSION ---
    # To regress r_{t+1} on xi,t, align Y[t+1] with X[t].
    # This means Y_for_regression should be r[1:] (r_2 to r_T) and
    # X_for_regression should be x[:-1] (x_1 to x_{T-1}).
    Y_for_regression = Y_full[1:]  # r_{t+1} values (from period 2 to T)
    X_for_regression = X_full[:-1] # x_i,t values (from period 1 to T-1)

    # Add constant term
    X_with_const = sm.add_constant(X_for_regression)

    # OLS regression with HAC standard errors
    # Now regressing r_{t+1} on const and x_i,t
    model = sm.OLS(Y_for_regression, X_with_const).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

    # Extract results (index 0 is const, index 1 is slope beta)
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
    results_dict[var] = model # Store the correctly fitted model if needed later

in_sample_df = pd.DataFrame(in_sample_results)

print("In-sample regression results (Corrected: rt+1 on xi,t):")
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
Y_full = data['r'].values # Re-assign Y_full here if it was modified earlier
forecasts_dict = {}
# Store forecasts for holdout period (m+1 to m+p) and out-of-sample period (m+p+1 to T)
holdout_forecasts_dict = {} # To store forecasts for periods m+1 to m+p (used for DMSPE weights)
oos_forecasts_dict = {}     # To store forecasts for periods m+p+1 to T (used for final R2)

for var in predictor_vars:
    X_full = data[var].values
    holdout_individual_forecasts = []
    oos_individual_forecasts = []

    # Generate forecasts for holdout period (m+1 to m+p, i.e., forecast r_{m+1} to r_{m+p})
    # Use data up to t-1 to forecast r_t (where t ranges from m+1 to m+p)
    # So, regress r_2 to r_{m+p} on const and x_1 to x_{m+p-1}
    for t in range(m + 1, m + p + 1): # t from m+1 to m+p (forecast r_{m+1} to r_{m+p})
        Y_train = Y_full[1:t] # Data r_2 to r_t (predictors start from r_2)
        X_train = X_full[:t-1] # Predictor x_1 to x_{t-1} (predicts r_2 to r_t)
        if len(Y_train) == len(X_train): # Ensure lengths match
            X_train_with_const = sm.add_constant(X_train)
            model = sm.OLS(Y_train, X_train_with_const).fit()
            # Forecast r_{t} using x_{t-1} (X_full[t-1] is x_{t-1})
            X_forecast = np.array([1, X_full[t-1]]) # Predictor x_{t-1} predicts r_t
            forecast = model.predict(X_forecast)[0]
            holdout_individual_forecasts.append(forecast)
        else:
            print(f"Length mismatch for {var} at t={t}: Y_train={len(Y_train)}, X_train={len(X_train)}")
            holdout_individual_forecasts.append(np.nan)

    # Generate forecasts for out-of-sample period (m+p+1 to T, i.e., forecast r_{m+p+1} to r_T)
    # Use data up to t-1 to forecast r_t (where t ranges from m+p+1 to T)
    for t in range(m + p + 1, total_obs): # t from m+p+1 to T (forecast r_{m+p+1} to r_T)
        Y_train = Y_full[1:t] # Data r_2 to r_t (predictors start from r_2)
        X_train = X_full[:t-1] # Predictor x_1 to x_{t-1} (predicts r_2 to r_t)
        if len(Y_train) == len(X_train): # Ensure lengths match
            X_train_with_const = sm.add_constant(X_train)
            model = sm.OLS(Y_train, X_train_with_const).fit()
            # Forecast r_{t} using x_{t-1} (X_full[t-1] is x_{t-1})
            X_forecast = np.array([1, X_full[t-1]]) # Predictor x_{t-1} predicts r_t
            forecast = model.predict(X_forecast)[0]
            oos_individual_forecasts.append(forecast)
        else:
            print(f"Length mismatch for {var} at t={t}: Y_train={len(Y_train)}, X_train={len(X_train)}")
            oos_individual_forecasts.append(np.nan)

    holdout_forecasts_dict[var] = holdout_individual_forecasts
    oos_forecasts_dict[var] = oos_individual_forecasts

# Historical average benchmark for Out-of-Sample Period (m+p+1 to T)
benchmark_forecasts_oos = []
for t in range(m + p + 1, total_obs): # t from m+p+1 to T, forecast for r_t (using data up to t-1)
    # Average from start to time t-1 (r_1 to r_{t-1})
    hist_avg = np.mean(Y_full[0:t]) # Average from r_1 to r_{t-1}
    if t == 1: # Handle first case where t=1, mean of [] is undefined, use r_1 if forecasting r_2 with no data makes sense
        # The first forecast in OOS period is for t = m+p+1, which is likely > 1.
        # The benchmark r_bar_{t} = (1/(t-1)) * sum(r_j, j=1 to t-1)
        # If t = m+p+1, the benchmark uses r_1 to r_{m+p}, which requires t-1 >= 1, so t >= 2.
        # Our loop starts at t = m+p+1 = 121, so t >= 2 is satisfied.
        # The benchmark r_bar_{t} is the forecast for r_t, based on data up to t-1.
        hist_avg = np.mean(Y_full[0:t]) # This handles t=2 correctly (mean of r_1)
    benchmark_forecasts_oos.append(hist_avg)

# Actual returns for Out-of-Sample Period (m+p+1 to T)
actual_returns_oos = Y_full[m + p + 1:total_obs] # r from r_{m+p+1} to r_T (indices m+p+1 to T-1)
mspe_benchmark_oos = np.mean((actual_returns_oos - benchmark_forecasts_oos)**2)

# Calculate out-of-sample R² (for Part 2)
oos_r2_results = []
for var in predictor_vars:
    forecasts = oos_forecasts_dict[var] # Use OOS forecasts
    # Check for NaNs and handle if necessary - ideally, there should be none with corrected loop
    forecasts_array = np.array(forecasts)
    if np.isnan(forecasts_array).any():
         print(f"Warning: NaN found in OOS forecasts for {var}")
    # Assuming no NaNs based on corrected logic, calculate MSPE
    mspe_model = np.mean((actual_returns_oos - forecasts_array)**2)
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
for t in range(m + p + 1, total_obs): # t from m+p+1 to T (forecast r_t using data up to t-1)
    Y_train = Y_full[1:t] # Use data r_2 to r_t (predictors start from r_2)
    X_train = data[predictor_vars].iloc[:t-1].values # Use predictor data x_1 to x_{t-1}
    if len(Y_train) == X_train.shape[0]: # Ensure lengths match
        X_train_with_const = sm.add_constant(X_train)
        model = sm.OLS(Y_train, X_train_with_const).fit()
        # Forecast r_t using x_{t-1} (data[predictor_vars].iloc[t-1].values)
        X_forecast = np.concatenate([[1], data[predictor_vars].iloc[t-1].values]) # Use predictor data x_{t-1}
        forecast = model.predict(X_forecast)[0]
        kitchen_sink_forecasts.append(forecast)
    else:
        print(f"Length mismatch in Kitchen Sink at t={t}: Y_train={len(Y_train)}, X_train={X_train.shape[0]}")
        kitchen_sink_forecasts.append(np.nan)

# Remove potential NaNs before calculating MSPE if any occurred (they shouldn't with corrected loop)
kitchen_sink_forecasts_clean = [f for f in kitchen_sink_forecasts if not np.isnan(f)]
if len(kitchen_sink_forecasts_clean) == len(actual_returns_oos):
    mspe_kitchen_sink = np.mean((actual_returns_oos - np.array(kitchen_sink_forecasts_clean))**2)
    oos_r2_kitchen_sink = 1 - (mspe_kitchen_sink / mspe_benchmark_oos)
    print(f"Kitchen Sink OOS R²: {oos_r2_kitchen_sink:.4f}")
    print(f"Outperforms benchmark: {oos_r2_kitchen_sink > 0}")
else:
    print("Error: Kitchen Sink forecasts length mismatch with actual returns.")
    print(f"Expected: {len(actual_returns_oos)}, Got (clean): {len(kitchen_sink_forecasts_clean)}")


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
    # Assuming no NaNs from corrected OOS loop
    mean_forecast = np.mean(individual_forecasts_t)
    mean_combination_forecasts.append(mean_forecast)

mspe_mean_combo = np.mean((actual_returns_oos - np.array(mean_combination_forecasts))**2)
oos_r2_mean_combo = 1 - (mspe_mean_combo / mspe_benchmark_oos)

# Median combination (OOS forecasts)
median_combination_forecasts = []
for t_idx in range(len(actual_returns_oos)): # Iterate over OOS period length
    individual_forecasts_t = [oos_forecasts_dict[var][t_idx] for var in predictor_vars]
    # Assuming no NaNs from corrected OOS loop
    median_forecast = np.median(individual_forecasts_t)
    median_combination_forecasts.append(median_forecast)

mspe_median_combo = np.mean((actual_returns_oos - np.array(median_combination_forecasts))**2)
oos_r2_median_combo = 1 - (mspe_median_combo / mspe_benchmark_oos)

# DMSPE combination
def calculate_dmspe_weights_fixed(holdout_forecasts_dict, actual_returns_holdout, oos_forecasts_dict, actual_returns_oos, theta):
    """
    Calculate DMSPE weights based on holdout period performance and apply to OOS period.
    """
    n_vars = len(predictor_vars)
    n_oos = len(actual_returns_oos) # Number of OOS forecasts

    # Calculate phi (cumulative discounted MSPE) for each variable using HOLDOUT data
    phi_values_holdout_end = []
    for var in predictor_vars:
        phi = 0.0
        # Iterate through the holdout period forecasts (index 0 to p-1)
        # The holdout forecasts are for r_{m+1} to r_{m+p}, based on data up to m, m+1, ..., m+p-1
        # So the actuals are Y_full[m+1 : m+p+1], forecasts are holdout_forecasts_dict[var]
        for s_idx in range(len(actual_returns_holdout)): # s_idx goes from 0 to p-1
            # actual_ret_holdout corresponds to r_{m+1+s_idx}
            # forecast_ret_holdout corresponds to r_hat_{m+1+s_idx}
            actual_ret_holdout = actual_returns_holdout[s_idx] # r_{m+1+s_idx}
            forecast_ret_holdout = holdout_forecasts_dict[var][s_idx] # r_hat_{m+1+s_idx}
            # Discount factor: theta^{(m+p) - (m+1+s_idx)} = theta^{p - 1 - s_idx}
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
actual_returns_holdout = Y_full[m + 1 : m + p + 1] # r from r_{m+1} to r_{m+p} (indices m+1 to m+p)

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
