import pandas as pd
from pathlib import Path
data_path = Path.cwd() / "data_raw" / "seoul_bike.csv"
out_data_path = Path.cwd() / "data_raw" / "seoul_bike_with_splits.csv"
df = pd.read_csv(str(data_path))
print(df.head())
print(df.columns)

df.columns = ['date', 'rented_bike_count', 'hour', 'temperature', 'humidity',
       'wind_speed', 'visibility', 'dew_point_temperature',
       'solar_radiation', 'rainfall', 'snowfall', 'seasons',
       'holiday', 'functioning_day']

print(len(df))

train_size = 0.7
val_size = 0.15

# Get index cutoffs
n = len(df)
train_end = int(n * train_size)
val_end = int(n * (train_size + val_size))

# Create splits
train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]
pass

df["forecast_split"] = ""
df.loc[:train_end,"forecast_split"] = "train"
df.loc[train_end:val_end,"forecast_split"] = "validation"
df.loc[val_end:,"forecast_split"] = "test"

df_random = df.sample(frac=1, random_state=42).reset_index(drop=True)

df_random["regression_split"] = ""
df_random.loc[:train_end,"regression_split"] = "train"
df_random.loc[train_end:val_end,"regression_split"] = "validation"
df_random.loc[val_end:,"regression_split"] = "test"
df_random["date"] = pd.to_datetime(df_random["date"],dayfirst=True)
df_random["date_and_hour" ] = df_random["date"] + pd.to_timedelta(df_random['hour'], unit='h')

df_random = df_random.drop(columns=["date","hour"])

df_random.to_csv(str(out_data_path),index=True)