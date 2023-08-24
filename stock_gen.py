import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
# from utils import TSDataset
import yfinance as yf

class TSDataset(Dataset):
		def __init__(self,x_train,labels):
				self.samples = x_train
				self.labels = labels

		def __len__(self):
				return len(self.samples)

		def __getitem__(self,idx):
				return self.samples[idx],self.labels[idx]

class EmptyDataset(Dataset):
    def __init__(self):
        super(EmptyDataset, self).__init__()

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError("Empty dataset does not contain any items.")

def gen_cube(instance):
    result = []
    for i in range(len(instance)):
        result.append([instance[(i+j)%len(instance)] for j in range(len(instance))])
    return result

def prepare_stock_dataloader(stock_symbols, start_date, end_date, sequence_length=30):
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


    # Create an instance of the empty dataset
    current_train_dataset = EmptyDataset()
    current_test_dataset = EmptyDataset()
    # Step 1: Extract daily stock data from Yahoo Finance and convert it to a pandas dataframe
    stock_data = pd.DataFrame()
    for stock in stock_symbols:
        stock_data = yf.download(stock, start=start_date, end=end_date)

    

        # Step 2.2: Min-max scale the stock data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close','Volume']])
        stock_data = pd.DataFrame(scaled_data, columns = ['Open', 'High', 'Low', 'Close', 'Adj Close','Volume'])
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

        windows, seq_len, dim = final_sliding_windows.shape
        final_sliding_windows = final_sliding_windows.reshape(windows, dim , seq_len)
        final_sliding_windows = np.array([gen_cube(acl) for acl in final_sliding_windows])
        # print("Final Sliding Windows", final_sliding_windows.shape)
        
        # Split sliding windows into two classes based on labels
        class1_windows = final_sliding_windows[final_labels == 1]
        class1_labels = final_labels[final_labels == 1]
        class0_windows = final_sliding_windows[final_labels == 0]

        class0_labels = final_labels[final_labels == 0]
        print("Class 1 Window", class1_windows.shape)
        print("Class 2 Window", class0_windows.shape)

        np.save('stock_data_class0_seq10_delay5.npy', class0_windows)
        np.save('stock_data_class1_seq10_delay5.npy', class1_windows)
        

        # Create an instance of the StockDataset
        train_dataset = TSDataset(final_sliding_windows[0: 150], final_labels)
        test_dataset = TSDataset(final_sliding_windows[150:], final_labels)
        
        current_train_dataset = ConcatDataset([current_train_dataset, train_dataset])
        current_test_dataset = ConcatDataset([current_test_dataset, test_dataset])
    # Create a PyTorch dataloader
    batch_size = 32
    train_dataloader = DataLoader(current_train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(current_test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader

train_loader, test_loader = prepare_stock_dataloader(['AAPL'], '2017-01-01', '2023-01-01', sequence_length=30)