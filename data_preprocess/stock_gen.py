import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

def prepare_stock_dataloader(stock_symbol, start_date, end_date, sequence_length=30):
    """
    Prepares a PyTorch dataloader for stock prediction.

    Args:
        stock_symbol (str): The symbol of the stock to fetch data for (e.g., 'AAPL' for Apple Inc.).
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.
        sequence_length (int, optional): The length of each sequence/window. Defaults to 30.

    Returns:
        torch.utils.data.DataLoader: A PyTorch dataloader containing the stock data.
    """

    # Step 1: Extract daily stock data from Yahoo Finance and convert it to a pandas dataframe
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

    

    # Step 2.2: Min-max scale the stock data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close','Volume']])
    stock_data = pd.DataFrame(scaled_data, columns = ['Open', 'High', 'Low', 'Close', 'Adj Close','Volume'])
    print(stock_data)
    # Step 2.1: Add a label column to indicate if the stock price increased over the previous day
    stock_data['Label'] = np.where(stock_data['Close'].shift(-5) > stock_data['Close'], 1, 0)

    # Step 3: Create an array with shape (sliding window, sequence length, dimension)
    dimension = 6
    sliding_windows = []

    labels = []
    for i in range(len(stock_data) - sequence_length + 1):
        window = stock_data.iloc[i:i+sequence_length, :dimension].values
        labels.append(np.mean(stock_data.iloc[i:i+sequence_length, dimension]))
        sliding_windows.append(window)

    # Convert the sliding windows to a numpy array
    sliding_windows = np.array(sliding_windows)

    # Step 4: Create an array labels with shape (Sliding Window, )
    # labels = stock_data['Label'].values[sequence_length-1:]
    # print("labels", labels)
    # Step 5: Check each sliding window and assign labels accordingly
    final_labels = []

    for window_labels in labels:
        result = -1
        if window_labels >= 0.6:
            result = 1
            final_labels.append(result)
        elif window_labels < 0.4:
            result = 0
            final_labels.append(result)
        else:
            final_labels.append(result)


    # print("final_labels", final_labels)
    # Step 6: Remove sliding windows with label -1
    final_sliding_windows = sliding_windows[np.array(final_labels) != -1]
    final_labels = np.array(final_labels)[np.array(final_labels) != -1]
    print("Final Sliding Windows", final_sliding_windows.shape)
    # Split sliding windows into two classes based on labels
    class1_windows = final_sliding_windows[final_labels == 1]
    class1_labels = final_labels[final_labels == 1]
    class0_windows = final_sliding_windows[final_labels == 0]
    class0_labels = final_labels[final_labels == 0]
    print("Class 1 Window", class1_windows.shape)
    print("Class 2 Window", class0_windows.shape)

    np.save('stock_data_class0_seq10_delay5.npy', class0_windows)
    np.save('stock_data_class1_seq10_delay5.npy', class1_windows)
    

    # Define a custom PyTorch dataset
    class StockDataset(Dataset):
        def __init__(self, sliding_windows, labels):
            self.sliding_windows = sliding_windows
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            window = self.sliding_windows[idx]
            label = self.labels[idx]
            return window, label

    # Create an instance of the StockDataset
    dataset = StockDataset(final_sliding_windows, final_labels)

    # Create a PyTorch dataloader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

print(prepare_stock_dataloader('AAPL', '2019-01-01', '2023-01-01', sequence_length=10))