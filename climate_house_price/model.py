import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, r2_score as r2e
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

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

house_pricing = pd.read_csv("house_pricing.csv", index_col=False, sep="\t")
house_inflation = pd.read_csv("house_inflation.csv")

# normalized based on 2021 inflation
normalized_house_pricing = house_pricing.copy().rename(
    columns={
        "ABS-diff price Q2 2021": "Normalized Growth 2021",
        "Geography": "County",
    }
)

# drop NaN, cast % string to float, and normalize
normalized_house_pricing["Normalized Growth 2021"] = normalized_house_pricing[
    "Normalized Growth 2021"
].dropna().str.rstrip("%").astype("float") / (
    100.0 * house_inflation["Inflation Rate"][2021 - 1968]
)

# remove price ranges, growth Q42021, latlong, and full county number
normalized_house_pricing = normalized_house_pricing.drop(
    columns=[
        "Price Range-Q3 2022",
        "growth Q42021",
        "Latitude (generated)",
        "Full County Number",
        "Price Range-Q3 2022.1",
        "Longitude (generated)",
    ]
)

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


fips = pd.read_csv(
    os.path.join(CLIMATE_DATA_DIRECTORY, "fips.csv"), index_col=False
)

# the NOAA uses different state codes at the beginning of the fips code
with open("noaa_state_codes.txt", "r") as f:
    noaa_state_codes = {
        line.split(",")[0].strip(): line.split(",")[1].strip()
        for line in f.readlines()
    }

# remove first 2 digits of each fips and replace with NOAA code for that state
fips["noaa_state_code"] = fips["state_name"].map(noaa_state_codes)
fips["fips"] = fips["fips"].astype("str").apply(lambda x: x[-3:])
fips["fips"] = (fips["noaa_state_code"] + fips["fips"]).dropna().astype("int")

fips_simple = fips[["fips", "long_name"]].rename(
    columns={"long_name": "County"}
)

# with open("log.txt", "w") as f:
#     f.write(fips_simple.to_string())

# convert state name at end of county name to state abbreviation in
# normalized_house_pricing
state_abbreviations = pd.read_csv("state_abbreviations.csv", index_col=False)

# convert normalized_house_pricing['State'] to state abbreviation
normalized_house_pricing["County"] = (
    normalized_house_pricing["County"].str.split(",").str[0]
    + " "
    + (
        normalized_house_pricing["County"]
        .str.split(",")
        .str[1]
        .str.strip()
        .map(state_abbreviations.set_index("state")["code"])
    )
)

# add fips column to normalized_house_pricing
normalized_house_pricing = normalized_house_pricing.merge(
    fips_simple, on="County"
)

# dropna for Washington DC
normalized_house_pricing["fips"] = (
    normalized_house_pricing["fips"].dropna().astype("int")
)

# print(normalized_house_pricing)

full_data = normalized_house_pricing.copy()

cdd_data_q3_2021 = cdd_data[cdd_data["Year"] == 2021]
hdd_data_q3_2021 = hdd_data[hdd_data["Year"] == 2021]
precipitation_data_q3_2021 = precipitation_data[
    precipitation_data["Year"] == 2021
]
avg_temp_data_q3_2021 = avg_temp_data[avg_temp_data["Year"] == 2021]
min_temp_data_q3_2021 = min_temp_data[min_temp_data["Year"] == 2021]
max_temp_data_q3_2021 = max_temp_data[max_temp_data["Year"] == 2021]


for (df, colname) in [
    (cdd_data_q3_2021, "cooling degree days"),
    (hdd_data_q3_2021, "heating degree days"),
    (precipitation_data_q3_2021, "precipitation"),
    (avg_temp_data_q3_2021, "average temperature"),
    (min_temp_data_q3_2021, "minimum temperature"),
    (max_temp_data_q3_2021, "maximum temperature"),
]:
    # print(df)
    full_data = full_data.merge(
        df.rename(
            {month: f"{month} {colname}" for month in MONTHS},
            axis=1,
        ),
        left_on=["fips"],
        right_on=["FIPS"],
    ).drop(
        columns=["FIPS", "Code", "Year"],
        axis=1,
    )

# print(full_data)

# remove the commas in between the prices in the Price Range-Q3 2022, Q1 2022,
# and Q4 2021 columns
for col in ("Price Q3 2022", "Q1 2022", "Q4 2021"):
    full_data[col] = (
        full_data[col]
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .astype(float)
    )
full_data = full_data.rename(
    columns={
        "Price Q3 2022": "Q3 2022 ($)",
        "Q1 2022": "Q1 2022 ($)",
        "Q4 2021": "Q4 2021 ($)",
    }
)

with open("log.txt", "w") as f:
    # sort by county name
    f.write(full_data.sort_values(by="County").to_string())

# split the data into training and testing sets
X = full_data.drop(
    columns=[
        "County",
        "Normalized Growth 2021",
        "Q3 2022 ($)",
        "Q1 2022 ($)",
        "Q4 2021 ($)",
        "fips"
    ]
)

# Create sequences for LSTM (time series forecasting)
def create_sequence_data(df, sequence_length):
    sequences = []
    labels = []
    for i in range(len(df) - sequence_length):
        sequences.append(df.iloc[i:i + sequence_length].values)
        labels.append(df.iloc[i + sequence_length]["Normalized Growth 2021"])
    return np.array(sequences), np.array(labels)

sequence_length = 12 
X_seq, y_seq = create_sequence_data(pd.DataFrame(scaled_features), sequence_length)

# s[;ot] into train and test sets
train_size = int(len(X_seq) * 0.75)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

# reshapaing data
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    tf.keras.layers.LSTM(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# model training
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# model predictoin
y_pred = model.predict(X_test)

# evaluate the model
mean_squared_error = mse(y_test, y_pred)
r2_score = r2e(y_test, y_pred)

print("Mean squared error: %.6f" % mean_squared_error)
print("Sqrt of mean squared error: %.6f" % np.sqrt(mean_squared_error))
print("Variance score: %.2f" % r2_score)

# plot the predicted normalized growth vs the actual normalized growth
plt.scatter(y_test, y_pred, color="black")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=4)

# line of best fit
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), color="red", linewidth=3, label="Line of Best Fit")
plt.legend()
plt.title("Predicted vs Actual Normalized Growth")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# region ridgereg
ridgemodel = Ridge(alpha=7)  # NOTE: could tune this
ridgemodel.fit(X_train, y_train)
y_pred = ridgemodel.predict(X_test)
# endregion

# region linreg
# linmodel = LinearRegression()
# linmodel.fit(X_train, y_train)
# y_pred = linmodel.predict(X_test)
# endregion

# evaluate the model
mean_squared_error = mse(y_test, y_pred)
r2_score = r2e(y_test, y_pred)

print("Mean squared error: %.6f" % mean_squared_error)
print("Sqrt of mean squared error: %.6f" % (mean_squared_error)**.5)
print("Variance score: %.2f" % r2_score)

# plot the predicted normalized growth vs the actual normalized growth
plt.scatter(y_test, y_pred, color="black")
plt.plot(
    [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=4
)
# line of best fit
plt.plot(
    np.unique(y_test),
    np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)),
    color="red",
    linewidth=3,
    label="Line of Best Fit",
)
plt.legend()
plt.title("Predicted vs Actual Normalized Growth")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()