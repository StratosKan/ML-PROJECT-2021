import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def trend(time, slope=0):
    return slope*time


def seasonal_pattern(season_time):
    return np.where(season_time < 0.4, np.cos(season_time*2*np.pi), 1/np.exp(3*season_time))


def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time+phase) % period) / period
    return amplitude * seasonal_pattern(season_time)  # repeats the same pattern at each period


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, .05)
amplitude = 15
slope = 0.09
noise_level = 6

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update the noise
series += noise(time, noise_level, seed=42)

plt.figure("TimeSeriesChart", figsize=(10, 6))
plot_series(time, series)
plt.show()

# Split the series to start forecasting
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

# plt.figure(figsize=(10, 6))
# plot_series(time_train, x_train)
# plt.show()

# plt.figure(figsize=(10, 6))
# plot_series(time_valid, x_valid)
# plt.show()

# NAIVE FORECAST

naive_forecast = series[split_time-1:-1]

# Computing the mean square error and mean absolute error between forecasts/predictions
print("Naive Forecast MSE:")
print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
print("Naive Forecast MAE:")
print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())

# MOVING AVERAGE FORECAST


def moving_average_forecast(series, window_size):
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    return np.array(forecast)


moving_avg = moving_average_forecast(series, 30)[split_time-30:]
plt.figure(num="MovingAverage_Forecast1", figsize=(10,6))
plot_series(time_valid, x_valid)
plot_series(time_valid, moving_avg)
plt.show()

# Moving average doesnt anticipate trend or seasonality. Removing them using differencing.

diff_series = (series[365:] - series[:-365])
diff_time = time[365:]

plt.figure(num="TrendSeasonalityDifferencing", figsize=(10, 6))
plot_series(diff_time, diff_series)
plt.show()

diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - 365 - 50:]
plt.figure(num="MovingAverage_Forecast2", figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - 365:])
plot_series(time_valid, diff_moving_avg)
plt.show()

diff_moving_avg_plus_past = series[split_time-365:-365] + diff_moving_avg
plt.figure(num="FixedMovingAverage_Forecast", figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_past)
plt.show()

print("Moving average MSE:")
print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_past).numpy())
print("Moving average MAE:")
print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past).numpy())

# Removing noise from forecast
diff_moving_avg_plus_smooth_past = moving_average_forecast(
    series[split_time - 370:-360], 10) + diff_moving_avg
plt.figure(num="MovingAverage_ForecastSmooth", figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_smooth_past)
plt.show()

print("Smooth Moving average MSE:")
print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
print("Smooth Moving average MAE:")
print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
