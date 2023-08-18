import argparse
import numpy as np
from train import *
from data_preprocess import anomaly_generation, stock_gen
from classtrainer import InceptionTime
from metrics.score_calculation import evaluate_metrics
from detection import anomaly
from sklearn.model_selection import train_test_split
'''
    # ------------- The logic is like that ----------- #
    After we declare the dcam class with model, we just find anomaly score
    result array is of shape (Total Length - Sequence Length x Total Length x D)
    1. loop through the testing dataset, for each we 
        i)  Create the DCAM Matrix 
        ii) Append to the result array
    2. Based on the result array, take average for each of sliding window
    3. Given the first S and last S scaler is only has single value, we scale it larger
    4. On the adjusted DCAM matrix, we make anomaly detection applied on both single sequence
'''




def main(config):
    
    
    # Step 1: Preprocess the data and generate necessary data for further computation
    file_path = config['input_path'] + config['input_file']
    seq_len = config['seq_len']
    noise_pct = config['noise_pct']

    dataloader, test_windows, labels = anomaly_generation.preprocess_data(file_path, seq_len, 0.3, noise_pct)

    # Step 2: Initialize a model and train the model.

    _, seq_len, dim = test_windows.shape

    model = InceptionTime(config)
    model = classifcation.train(model, dataloader)
    last_conv_layer = model._modules['layer3']
    fc_layer = model._modules['fc1']
    
    
    # Step 3: Generate aggragated dcam and predicted anomaly labels using STD
    dcam = anomaly.dcam_result(model, test_windows, last_conv_layer, fc_layer)
    pred_label = anomaly.anamoly_detection_one(dcam)

    # Step 4: Evaluate the predicted labels and test labels
    test_label = np.array(labels)
    metrics = evaluate_metrics(pred_label, test_label)

    return metrics
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # --------------- Arguments for the Data Preprocessing Process ---------------- # 
    parser.add_argument('--input_path', type= str, default = 'inputs/')
    parser.add_argument('--input_file', type = str, default= 'train.csv')
    parser.add_argument('--train_ratio', type = float, default = 0.7)
    parser.add_argument('--seq_len', type = float, default = 200)
    parser.add_argument('--noise_pct', type = float, default = 0.1)

    # --------------- Arguments for training the time series classification model ---------------- # 
    parser.add_argument('--num_epochs', type= int, default = 200)
    parser.add_argument('--out_channel', type = int, default = 32)
    parser.add_argument('--bottleneck_channel', type = int, default = 32)
    parser.add_argument('--nb_class', default = int, default = 2)

    '''
        in_channel = args['in_channel']
        out_channel = args['out_channel']
        bottleneck_channel = args['bottleneck_channel']
        nb_class = args['nb_class']
    '''
    config = parser.parse_args
    args = vars(config)
    
    print('--------- Options ---------')
    for k, v in sorted(args.items()):
        print(f'{k}:{v}')

    print('--------- End ---------')

    main(config)