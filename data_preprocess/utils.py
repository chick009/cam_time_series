from torch.utils.data import Dataset, DataLoader
# Define a custom PyTorch dataset
class TSDataset(Dataset):
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
# dataset = StockDataset(final_sliding_windows, final_labels)

# Create a PyTorch dataloader
# batch_size = 32
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)