import argparse
import numpy as np
from train import *
from data_preprocess import anomaly_generation, stock_gen
from classtrainer import InceptionTime
from metrics.score_calculation import evaluate_metrics
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

def dcam_result(model, inputs, last_conv_layer, fc_layer, device = 'cpu'):
    '''
    
    args:
        inputs - shape (B x S x D)

    '''
    
    window, seq_len, dim = inputs.shape
    DCAM_m = DCAM(model, 'cpu', last_conv_layer = last_conv_layer, fc_layer = fc_layer)

    total_len = window + seq_len 
    result = np.zeros((total_len - seq_len, total_len, dim))

    # Create DCAM Matrix for each instance and append to result array
    for i in range(window):
        dcam, label = DCAM_m.run(
            instance = inputs[i, :, :],
            nb_permutation = 200,
        )
        result[i, :, i: i + seq_len] = dcam

    # 2. Based on the result array, take average for each of sliding window
    final_result = np.mean(result, axis = 0)      

    # 3. Given the first S and last S scaler is only has single value, we scale it larger
    for num in reversed(range(1, seq_len + 1)):
        final_result[:][num] = final_result[:][num] * num
        final_result[:][-num] = final_result[:][-num] * num

    return final_result

def anamoly_detection_one(final_result):
    '''
        Using Ideas from TAD-GAN in self-supervised learning 
        args:
            final_result: shape (S x D)

    '''
    # For each timestep, combine all dimension into one to determine anomaly
    seq_value = np.mean(final_result, axis = 1)

    # Sequence length for ones - or may use max in future
    avg_seq_value = np.mean(final_result, axis = 1)
    std_seq_value = np.std(final_result, axis = 1)

    # Set the thresholds
    threshold = avg_seq_value + 3 * std_seq_value

    index = [1 for idx, value in enumerate(final_result) if value > threshold else 0]
    
    return index


def main(config):
    
    task_to_solve = config['task']
    input_path = config['path']
    
    # Need to split input data into testing and training dataset
    class1, class2, original_window = preprocess_data('C:/Users/johnn/cam_time_series_new/inputs/train.csv', 300, 0.3, 0.1)
    
  
    
    # Set up dataloader for training data

    label = [1]* total_length + [0] * total_length
    label = np.array(label)

    all_class = class1_train + class2_train
  
    from torch.utils.data import TensorDataset, DataLoader
    data_train = torch.from_numpy(all_class) 
    label_train = torch.from_numpy(label)
    train_ds = TensorDataset(data_train, label_train)
    train_loader = DataLoader(train_ds, batch_size= 32)


    # --------------------------# 

    dataloader, test_windows, labels = anomaly_generation(...)
    # Train the data with anomaly class and normal class
    model = InceptionTime()
    model = classifcation.train(model, dataloader)

    # Randomly choose an instance
    last_conv_layer = model._modules['layer3']
    fc_layer = model._modules['fc1']
    
    dcam = dcam_result(model, test_windows, last_conv_layer, fc_layer)
    
    # Create labels for comparing evaluating metrics
    pred_label = anamoly_detection_one(dcam)
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