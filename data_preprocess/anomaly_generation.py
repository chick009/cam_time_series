
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import numpy as np

def preprocess_data(csv_file, seq_length, ratio, k):
  # Import subset of CSV  
  df = pd.read_csv(csv_file, index_col=0, nrows=int(ratio*pd.read_csv(csv_file).shape[0]))

  # Get shape
  print(df.shape)

  # Normalize data 
  scaler = MinMaxScaler()
  df_scaled = scaler.fit_transform(df)
  df_scaled = pd.DataFrame(df_scaled)
  df_scaled = df_scaled.interpolate()
  # Construct DataFrames
  # Filter outliers & interpolate
  df1 = df_scaled.copy()
  db = DBSCAN(eps=0.5, min_samples=5).fit(df1) 
  df1 = df1.interpolate()[db.labels_ != -1]

  # Randomly add noise on some timesteps 
  num_noisy = int(k*len(df_scaled))
  noise_idx = np.random.choice(len(df_scaled), num_noisy, replace=False)
  df2 = df_scaled.copy()
  df2.loc[noise_idx] += np.random.normal(loc=0, scale=0.1, size=(num_noisy, len(df_scaled.columns)))

  # Convert to sliding windows
  df1_windows = [df1.iloc[i:i+seq_length].values for i in range(len(df1)-seq_length+1)]
  df2_windows = [df2.iloc[i:i+seq_length].values for i in range(len(df2)-seq_length+1)]
  original_windows = [df_scaled.iloc[i:i+seq_length].values for i in range(len(df2)-seq_length+1)]

  df1_windows = np.array(df1_windows)
  df2_windows = np.array(df2_windows)
  original_windows = np.array(original_windows)

  # Return preprocessed arrays
  return df1_windows, df2_windows, original_windows

# Example usage
df1, df2, df3 = preprocess_data('C:/Users/johnn/cam_time_series_new/inputs/train.csv', 300, 0.3, 0.1)
print(df1.shape)
print(df2.shape)
print(df3.shape)