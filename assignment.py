
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
# PART 2: OUT-OF-SAMPLE ESTIMATION
# =============================================================================

print("\n" + "="*60)
print("PART 2: OUT-OF-SAMPLE ESTIMATION")
print("="*60)

# Sample splits
total_obs = len(data)
m = 80  # 20 years * 4 quarters
p = 40  # 10 years * 4 quarters
q = total_obs - m - p

print(f"Sample split: m={m}, p={p}, q={q}, total={total_obs}")

# Generate out-of-sample forecasts for each predictor
Y_full = data['r'].values
forecasts_dict = {}

for var in predictor_vars:
    X_full = data[var].values
    individual_forecasts = []

    for t in range(m + p, total_obs - 1):
        # Training data
        Y_train = Y_full[1:t+1]
        X_train = X_full[:t]
        X_train = sm.add_constant(X_train)

        # Fit model and forecast
        model = sm.OLS(Y_train, X_train).fit()
        X_forecast = np.array([1, X_full[t]])
        forecast = model.predict(X_forecast)[0]
        individual_forecasts.append(forecast)

    forecasts_dict[var] = individual_forecasts

# Historical average benchmark
benchmark_forecasts = []
for t in range(m + p, total_obs - 1):
    hist_avg = np.mean(Y_full[1:t+1])
    benchmark_forecasts.append(hist_avg)

# Actual returns
actual_returns = Y_full[m + p + 1:total_obs]
mspe_benchmark = np.mean((actual_returns - benchmark_forecasts)**2)

# Calculate out-of-sample R²
oos_r2_results = []
for var in predictor_vars:
    forecasts = forecasts_dict[var]
    mspe_model = np.mean((actual_returns - forecasts)**2)
    oos_r2 = 1 - (mspe_model / mspe_benchmark)

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
for t in range(m + p, total_obs - 1):
    Y_train = Y_full[1:t+1]
    X_train = data[predictor_vars].iloc[:t].values
    X_train = sm.add_constant(X_train)

    model = sm.OLS(Y_train, X_train).fit()
    X_forecast = np.concatenate([[1], data[predictor_vars].iloc[t].values])
    forecast = model.predict(X_forecast)[0]
    kitchen_sink_forecasts.append(forecast)

mspe_kitchen_sink = np.mean((actual_returns - kitchen_sink_forecasts)**2)
oos_r2_kitchen_sink = 1 - (mspe_kitchen_sink / mspe_benchmark)

print(f"Kitchen Sink OOS R²: {oos_r2_kitchen_sink:.4f}")
print(f"Outperforms benchmark: {oos_r2_kitchen_sink > 0}")

# =============================================================================
# PART 4: FORECAST COMBINATION
# =============================================================================

print("\n" + "="*60)
print("PART 4: FORECAST COMBINATION")
print("="*60)

# Mean combination
mean_combination_forecasts = []
for t_idx in range(len(actual_returns)):
    individual_forecasts_t = [forecasts_dict[var][t_idx] for var in predictor_vars]
    mean_forecast = np.mean(individual_forecasts_t)
    mean_combination_forecasts.append(mean_forecast)

mspe_mean_combo = np.mean((actual_returns - mean_combination_forecasts)**2)
oos_r2_mean_combo = 1 - (mspe_mean_combo / mspe_benchmark)

# Median combination
median_combination_forecasts = []
for t_idx in range(len(actual_returns)):
    individual_forecasts_t = [forecasts_dict[var][t_idx] for var in predictor_vars]
    median_forecast = np.median(individual_forecasts_t)
    median_combination_forecasts.append(median_forecast)

mspe_median_combo = np.mean((actual_returns - median_combination_forecasts)**2)
oos_r2_median_combo = 1 - (mspe_median_combo / mspe_benchmark)

# DMSPE combination
def calculate_dmspe_weights(forecasts_dict, actual_returns, theta, m, p):
    n_vars = len(predictor_vars)
    n_oos = len(actual_returns)
    dmspe_forecasts = []

    for t_idx in range(n_oos):
        if t_idx == 0:
            weights = np.ones(n_vars) / n_vars
        else:
            phi_values = []
            for var in predictor_vars:
                phi = 0
                for s in range(min(t_idx, p)):
                    actual_ret = actual_returns[t_idx - 1 - s]
                    forecast_ret = forecasts_dict[var][t_idx - 1 - s]
                    phi += (theta ** s) * (actual_ret - forecast_ret) ** 2
                phi_values.append(phi + 1e-8)

            phi_values = np.array(phi_values)
            inv_phi = 1.0 / phi_values
            weights = inv_phi / np.sum(inv_phi)

        individual_forecasts_t = [forecasts_dict[var][t_idx] for var in predictor_vars]
        combined_forecast = np.sum(weights * individual_forecasts_t)
        dmspe_forecasts.append(combined_forecast)

    return dmspe_forecasts

# DMSPE for θ = 0.9 and θ = 1.0
dmspe_results = []
for theta in [0.9, 1.0]:
    dmspe_forecasts = calculate_dmspe_weights(forecasts_dict, actual_returns, theta, m, p)
    mspe_dmspe = np.mean((actual_returns - dmspe_forecasts)**2)
    oos_r2_dmspe = 1 - (mspe_dmspe / mspe_benchmark)
    dmspe_results.append({
        'Method': f'DMSPE (θ={theta})',
        'OOS R²': oos_r2_dmspe
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

print(f"Best individual predictor: {oos_df.loc[oos_df['OOS R²'].idxmax(), 'Variable']} ({oos_df['OOS R²'].max():.4f})")
print(f"Kitchen sink: {oos_r2_kitchen_sink:.4f}")
print(f"Mean combination: {oos_r2_mean_combo:.4f}")
print(f"Median combination: {oos_r2_median_combo:.4f}")

print("\nCompleted successfully!")
