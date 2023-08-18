import numpy as np
from explanation.DCAM import DCAM
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

    labels = [1 if value > threshold else 0 for value in final_result]
    
    return labels