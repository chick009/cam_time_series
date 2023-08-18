
import pandas as pd
from .utils import TSDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from torch.utils.data import Dataset, DataLoader
import numpy as np

def dataloader_conversion(df1_windows, df2_windows):

  train_ratio = 0.7

  data_window = int(len(df2_windows) * train_ratio)

  # Got a train dataloader
  train_df1 = df1_windows[0: data_window, :, :]
  train_df2 = df2_windows[0: data_window, :, :]

  all_class = np.concatenate((train_df1, train_df2), axis=0)
  all_class = np.transpose(all_class, (0, 2, 1))
  label = [1] * len(train_df1) + [0] * len(train_df2)

  dataset = TSDataset(all_class, label)
  train_loader = DataLoader(dataset, batch_size = 32, shuffle = True)
  

  # Create a test dataloader:
  test_df1 = df1_windows[0: data_window, :, :]
  test_df2 = df2_windows[0: data_window, :, :]
  all_class = np.concatenate((test_df1, test_df2), axis=0)
  all_class = np.transpose(all_class, (0, 2, 1))
  label = [1] * len(test_df1) + [0] * len(test_df2)
  dataset = TSDataset(all_class, label)
  test_loader = DataLoader(dataset, batch_size = 32, shuffle = True)

  return train_loader, test_loader

def preprocess_data(input_path, test_path, test_path_label, seq_length, ratio, k):
  # Import 
  df = pd.read_csv(input_path, index_col=0, nrows=int(ratio*pd.read_csv(input_path).shape[0]))

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
  

  df1_windows = np.array(df1_windows)
  df2_windows = np.array(df2_windows)
  train_loader, test_loader = dataloader_conversion(df1_windows, df2_windows)

  # -------------------- Testing DataSet --------------------------# 
  df = pd.read_csv(test_path, index_col=0, nrows=int(ratio*pd.read_csv(test_path).shape[0]))
  # df_scaled = scaler.fit_transform(df)
  test_windows = [df_scaled.iloc[i:i+seq_length].values for i in range(len(df_scaled)-seq_length+1)]
  test_windows = np.array(test_windows)
  labels = pd.read_csv(test_path_label, index_col=0, nrows=int(ratio*pd.read_csv(test_path_label).shape[0]))
  
  
  # Return preprocessed arrays  
  return train_loader, test_loader, test_windows, labels.values

# Example usage
# df1, df2, df3 = preprocess_data('C:/Users/johnn/cam_time_series_new/inputs/train.csv', 'C:/Users/johnn/cam_time_series_new/inputs/test.csv', 'C:/Users/johnn/cam_time_series_new/inputs/test_label.csv', 300, 0.3, 0.1)
# print(df1.shape)
# print(df2.shape)
# print(df3.shape)