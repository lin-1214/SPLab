import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from pykalman import KalmanFilter
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from arch import arch_model
import warnings
from statsmodels.tsa.arima.model import ARIMA
warnings.filterwarnings('ignore')


def evaluate_thresholds(entry_threshold, exit_threshold, z_scores, prices):
    positions = np.zeros(len(z_scores))
    in_position = False
    
    # Generate trading signals
    for i in range(1, len(z_scores)):
        if not in_position and z_scores[i-1] <= entry_threshold:  # Enter long
            positions[i] = 1
            in_position = True
        elif in_position and z_scores[i-1] >= exit_threshold:  # Exit long
            positions[i] = 0
            in_position = False
        elif in_position:  # Stay in position
            positions[i] = 1
            
    # Calculate returns
    price_returns = np.diff(prices) / prices[:-1]
    strategy_returns = positions[1:] * price_returns
    return np.sum(strategy_returns)  # Total return


df = pd.read_csv('./stock_price_6669_202401_202502.csv')
df['Date'] = pd.to_datetime(df['Date'].apply(lambda x: f"{int(x.split('/')[0])+1911}/{x.split('/')[1]}/{x.split('/')[2]}"))

kf = KalmanFilter(
    transition_matrices=[1],
    observation_matrices=[1],
    initial_state_mean= np.mean(df['Closing Price']),
    initial_state_covariance=np.var(df['Closing Price']),
    observation_covariance=np.var(df['Closing Price']),  # Use data variance
    transition_covariance=0.1*np.var(df['Closing Price'])) 

mean, cov = kf.filter(df['Closing Price'].values)

z_scores = [(df['Closing Price'].values[i] - mean[i]) / np.sqrt(cov[i]) for i in range(len(df['Closing Price'].values))]
z_scores = np.array(z_scores).squeeze()

# Create grid search for thresholds
entry_thresholds = np.linspace(-1, 0, 100)  # Entry thresholds between -1 and 0
exit_thresholds = np.linspace(-0.5, 1, 150)  # Exit thresholds between -0.5 and 1
best_return = float('-inf')
best_entry_threshold = 0
best_exit_threshold = 0

# Find optimal thresholds
for entry_t in entry_thresholds:
    for exit_t in exit_thresholds:
        if exit_t > entry_t:  # Only test when exit threshold is higher than entry
            total_return = evaluate_thresholds(entry_t, exit_t, z_scores, df['Closing Price'].values)
            if total_return > best_return:
                best_return = total_return
                best_entry_threshold = entry_t
                best_exit_threshold = exit_t

print(f"Best long entry threshold: {best_entry_threshold:.3f}")
print(f"Best long exit threshold: {best_exit_threshold:.3f}")
print(f"Strategy return: {best_return:.3f}")

# Calculate positions using best thresholds
positions = np.zeros(len(z_scores))
in_position = False
returns = np.zeros(len(z_scores))  # Add array to store returns
cumulative_returns = np.zeros(len(z_scores))  # Add array for cumulative returns

for i in range(1, len(z_scores)):
    if not in_position and z_scores[i-1] <= best_entry_threshold:
        positions[i] = 1
        in_position = True
    elif in_position and z_scores[i-1] >= best_exit_threshold:
        positions[i] = 0
        in_position = False
    elif in_position:
        positions[i] = 1
    
    # Calculate returns when in position
    if positions[i] == 1:
        returns[i] = (df['Closing Price'].values[i] - df['Closing Price'].values[i-1]) / df['Closing Price'].values[i-1]
    
    # Calculate cumulative returns
    cumulative_returns[i] = cumulative_returns[i-1] + returns[i]

# First figure - Price with Kalman Filter
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Closing Price'], 'b-', label='Closing Price')
plt.plot(df['Date'], mean, 'r-', label='Kalman Filter Average')

# Add green background for holding periods
plt.fill_between(df['Date'], plt.gca().get_ylim()[0], plt.gca().get_ylim()[1],
                where=positions==1, color='green', alpha=0.2, label='Holding Period')

plt.title('Stock Closing Price with Kalman Filter')
plt.xlabel('Date')
plt.ylabel('Price')

plt.gca().xaxis.set_major_locator(MonthLocator())
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))

plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('./img/stock_price_kalman_6669_202401_202412.png')

# Plot z-scores with optimal thresholds
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], z_scores, 'g-', label='Z-Score')
plt.axhline(y=best_entry_threshold, color='b', linestyle='--', label='Entry Threshold')
plt.axhline(y=best_exit_threshold, color='r', linestyle='--', label='Exit Threshold')

# Add green background for holding periods
plt.fill_between(df['Date'], plt.gca().get_ylim()[0], plt.gca().get_ylim()[1],
                where=positions==1, color='green', alpha=0.2, label='Holding Period')

plt.title('Z-Scores Over Time with Optimal Long Entry/Exit Thresholds')
plt.xlabel('Date')
plt.ylabel('Z-Score')

plt.gca().xaxis.set_major_locator(MonthLocator())
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))

plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('./img/z_scores_thresholds_6669_202401_202412.png')

# Create a third figure for returns analysis
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], cumulative_returns, 'b-', label='Cumulative Returns')

# Add green background for holding periods
plt.fill_between(df['Date'], plt.gca().get_ylim()[0], plt.gca().get_ylim()[1],
                where=positions==1, color='green', alpha=0.2, label='Holding Period')

plt.title('Cumulative Strategy Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')

plt.gca().xaxis.set_major_locator(MonthLocator())
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))

plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('./img/cumulative_returns_6669_202401_202412.png')

# ==========================================================================

# Test for stationarity
def test_stationarity(data):
    result = adfuller(data)
    return result[1] < 0.05  # Returns True if stationary (p-value < 0.05)

# Find best ARIMA parameters
def find_best_arima(data):
    model = auto_arima(data,
                      start_p=0, max_p=3,
                      start_q=0, max_q=3,
                      m=1,
                      start_P=0, max_P=2,
                      start_Q=0, max_Q=2,
                      seasonal=False,
                      d=None, max_d=2,
                      trace=False,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)
    return model.order

# Find best GARCH parameters
def find_best_garch(residuals):
    best_aic = np.inf
    best_order = (1, 1)
    
    for p in range(1, 4):
        for q in range(1, 4):
            try:
                model = arch_model(residuals, vol='Garch', p=p, q=q)
                results = model.fit(disp='off')
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, q)
            except:
                continue
    return best_order

# Fit models with optimal parameters
prices = mean
best_arima_order = find_best_arima(prices)
print(f"Best ARIMA order: {best_arima_order}")

arima_model = ARIMA(prices, order=best_arima_order)
arima_fit = arima_model.fit()
residuals = arima_fit.resid

best_garch_order = find_best_garch(residuals)
print(f"Best GARCH order: {best_garch_order}")

garch_model = arch_model(residuals, vol="Garch", p=best_garch_order[0], q=best_garch_order[1])
garch_fit = garch_model.fit(disp="off")

# Forecast next 30 days
forecast_days = 30
arima_forecast = arima_fit.forecast(steps=forecast_days)
garch_forecast = garch_fit.forecast(horizon=forecast_days)
forecast_variance = garch_forecast.variance.values[-1, :]  # Get the last row of variance forecasts

# Calculate confidence intervals (95%)
confidence_multiplier = 1.96  # 95% confidence interval
forecast_std = np.sqrt(forecast_variance)
lower_bound = arima_forecast - confidence_multiplier * forecast_std
upper_bound = arima_forecast + confidence_multiplier * forecast_std

# Plot with confidence intervals
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], prices, 'b-', label='Historical Prices')
forecast_dates = pd.date_range(df['Date'].iloc[-1], periods=forecast_days, freq='D')
plt.plot(forecast_dates, arima_forecast, 'r-', label='ARIMA-GARCH Forecast')
plt.fill_between(forecast_dates, 
                lower_bound, 
                upper_bound, 
                color='r', 
                alpha=0.2, 
                label='95% Confidence Interval')

plt.title('Stock Price Forecast with Optimal ARIMA-GARCH Model')
plt.xlabel('Date')
plt.ylabel('Price')

plt.gca().xaxis.set_major_locator(MonthLocator())
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))

plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('./img/arima_garch_forecast_6669_202401_202412.png')









