## Abstract 

Data Series Classification

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