## Abstract 

Data Series Classification

## Steps 
1. ./data_preprocess: Preprocess time series and return train_dataloader, test_window, and labels on testing data
2. ./ts_model & ./train: Initiate Time Series Classification Model and Train using train_dataloader
3. ./explanation: Using the trained model to generate dCAM on the testing dataset.
4. ./anomaly_detection: detect the anomaly and generate a predicted dataset
5. ./metrics: Provide visualisation and metrics calculation


## Data

The data used in the project is prepared using three kind of datasets
- The S&P 500 Dataset
- Anomaly Detection Dataset (PSM, SMAP)
- Clustered result

## Code

The code is divided as follows:
- The data preprocessing procedure (Extract Data and Turn it into DataLoader):
    - anomaly_generation.py: Generate Normal and Anomaly Dataset
    - stock_gen.py: Generate Stock DataLoader for predicting Dynamic
    - cluster_result.py: Generate cluster result for different steps

- The explanation/ folder that contains:
    - CAM.py: Normal Class Activation Map Generation
    - DCAM.py: Dimensional-Wise Class Activation Map Generation

- The ts_model/ folder that contains: 
    - FullConv.py
    - InceptionTime.py
    - ResNet.py

- 