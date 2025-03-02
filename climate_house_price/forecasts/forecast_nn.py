import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM
from keras.models import Sequential
from scalecast.Forecaster import Forecaster
from scalecast.SeriesTransformer import SeriesTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, r2_score as r2e
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from statsmodels.tsa.arima.model import ARIMA

# region constants
MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

CLIMATE_DATA_DIRECTORY = "NOAA Climate Data/"
# endregion

# region read NOAA data
cdd_data = pd.read_csv(
    os.path.join(CLIMATE_DATA_DIRECTORY, "cdd.tsv"),
    index_col=False,
    sep="\t",
)
hdd_data = pd.read_csv(
    os.path.join(CLIMATE_DATA_DIRECTORY, "hdd.tsv"),
    index_col=False,
    sep="\t",
)
precipitation_data = pd.read_csv(
    os.path.join(CLIMATE_DATA_DIRECTORY, "pcpn.tsv"),
    index_col=False,
    sep="\t",
)
avg_temp_data = pd.read_csv(
    os.path.join(CLIMATE_DATA_DIRECTORY, "tavg.tsv"),
    index_col=False,
    sep="\t",
)
min_temp_data = pd.read_csv(
    os.path.join(CLIMATE_DATA_DIRECTORY, "tmin.tsv"),
    index_col=False,
    sep="\t",
)
max_temp_data = pd.read_csv(
    os.path.join(CLIMATE_DATA_DIRECTORY, "tmax.tsv"),
    index_col=False,
    sep="\t",
)
# endregion

for df in (
    cdd_data,
    hdd_data,
    precipitation_data,
    avg_temp_data,
    min_temp_data,
    max_temp_data,
):
    df["FIPS"] = df["Code"].apply(lambda x: int(str(x)[:-6]))
    df["Year"] = df["Code"].apply(lambda x: int(str(x)[-4:]))

precipitation_data_autauga = precipitation_data[precipitation_data["FIPS"] == 1001]
with open("log.txt", "w") as f:
    f.write(precipitation_data_autauga.to_string())

precipitation_data_autauga_jan = precipitation_data_autauga.drop(
    MONTHS[1:] + ["Code", "FIPS", "Year"], axis=1
)
with open("log.txt", "w") as f:
    f.write(precipitation_data_autauga_jan.to_string())

f = Forecaster(
    precipitation_data_autauga_jan["Jan"],
    (precipitation_data_autauga_jan["Jan"].index - 77) * 31_536_000_000_000_000,
)

f.plot_pacf(lags=60)  # NOTE: this showed that the first lag was significant

# f.seasonal_decompose().plot()
plt.show()

f.set_test_length(12)  # 1. 12 observations to test the results
f.generate_future_dates(12)  # 2. 12 future points to forecast

# f.set_estimator("lstm")  # 3. LSTM neural network
# f.manual_forecast(
#     call_me='lstm_best',
#     lags=36,
#     batch_size=32,
#     epochs=15,
#     validation_split=.2,
#     shuffle=True,
#     activation='tanh',
#     optimizer='Adam',
#     learning_rate=0.001,
#     lstm_layer_sizes=(72,)*4,
#     dropout=(0,)*4,
# )
# f.plot_test_set()
# plt.show()

# transformer = SeriesTransformer(f)
# f = transformer.DiffTransform()

# f.add_ar_terms(1)
# f.add_seasonal_regressors('month', dummy=True)
# f.add_seasonal_regressors('year')
# f.add_time_trend()
# f.set_estimator('mlr')
# f.manual_forecast()

# f.plot_test_set()
# f.plot()
# plt.show()
