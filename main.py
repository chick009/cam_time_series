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
    
    task_to_solve = config['task']
    input_path = config['path']

    # Step 1: Preprocess the data and generate necessary data for further computation

    dataloader, test_windows, labels = anomaly_generation.preprocess_data('C:/Users/johnn/cam_time_series_new/inputs/train.csv', 300, 0.3, 0.1)
    
    # Step 2: Initialize a model and train the model.
    model = InceptionTime()
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
    
    parser.add_argument('--path', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type= int, default = 20)

    config = parser.parse_args
    args = vars(config)
    
    print('--------- Options ---------')
    for k, v in sorted(args.items()):
        print(f'{k}:{v}')

    print('--------- End ---------')

    main(config)