import random
import numpy as np
from torch.autograd import Variable
import torch
from torch.nn import functional as F
from torch import topk
from tqdm import tqdm

'''
1. Compute Permutation
- 
'''
class DCAM():
    def __init__(self, model, device, last_conv_layer, fc_layer):
        self.device = device
        self.last_conv_layer = last_conv_layer
        self.fc_layer = fc_layer
        self.model = model

    def create_DCAM(self, instance, nb_permutation, label):
        all_permut, success_rate = self.__compute_permutations(instance, label)
        dcam = self.__extract_dcam(self.__merge_permutation(all_permut))

    def __gen_random_cube(self, instance):
        result = []
        result_comb = []

        # Extract the number of dimensions, turn it into list and shuffle index
        initial_comb = list(range(len(instance)))
        random.shuffle(initial_comb)

        # Generate C(T) where result is the C(T) matrix
        # Result_comb is only storing the permutation index in the matrix
        for i in range(len(instance)):
            # Row Initialization Operations
            result.append([instance[initial_comb[(i + j) % len(instance)]]] for j in range(len(instance)))
            result_comb.append([initial_comb[(i+j)%len(instance)] for j in range(len(instance))])

        return result, result_comb
    
    def __getCAM(self,feature_conv, weight_fc, class_idx):
        _ , nch, nc, length = feature_conv.shape
		feature_conv_new = feature_conv
		cam = weight_fc[class_idx].dot(feature_conv_new.reshape((nch,nc*length)))
		cam = cam.reshape(nc,length)
		cam = (cam - np.min(cam))/(np.max(cam) - np.min(cam))
		return cam


	def __get_CAM_class(self, instance):
        original_dim = len(instance)
		original_length = len(instance[0][0])
		instance_to_try = Variable(
			torch.tensor(
				instance.reshape(
					(1,original_dim,original_dim,original_length))).float().to(self.device),
			requires_grad=True)
		final_layer  = self.last_conv_layer
		activated_features = SaveFeatures(final_layer)
		prediction = self.model(instance_to_try)
		pred_probabilities = F.softmax(prediction).data.squeeze()
		activated_features.remove()
		weight_softmax_params = list(self.fc_layer_name.parameters())
		weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
		
		class_idx = topk(pred_probabilities,1)[1].int()
		overlay = self.__getCAM(activated_features.features, weight_softmax, class_idx )
		
		return overlay,class_idx.item()


    def __compute_multidim_cam(self, instance, nb_dim):
        cube, cube_comb = self.__gen_random_cube(instance) # Initialize different permutation
        overlay, label = self.__get_CAM_class(np.array(cube)) # Generate CAM for a specific permutation
        full_mat = np.zeros((nb_dim, nb_dim, len(overlay[0]))) # Shape (D x D x S)
        
        for i in range(nb_dim):
            for j in range(nb_dim):
                full_mat[cube_comb[i][j]][i] = overlay[j]
        
        return full_mat, label
    
    def __compute_permutations(self, instance, nb_permutation, label):
        all_pred_class = []
        all_matfull_list = [] 

        nb_dim, S = instance.shape # Dimension, Sequence Length
        final_mat = np.zeros((nb_dim, nb_dim))

        for k in tqdm(range(0, nb_permutation)):

            _, cam, label = self.__compute_multidim_cam(instance, nb_dim)
            
            all_matfull_list.append(cam)
            all_pred_class(label)

        return all_matfull_list, float(all_pred_class.count(label))
    
    def __merge_permutation(self, all_permut):
        nb_permut, nb_dim, nb_dim, nb_seq = all_permut.shape
        full_mat_avg = np.zero((nb_dim, nb_dim, nb_seq))

        for i in range(nb_dim):
            for j in range(nb_dim):
                mean_line = [] # CAM at pos i, dim j, at different time index 
                for n in range(nb_seq):
                    mean_line.append(np.mean([all_permut[k][i][j][n] for k in range(nb_permut)]))
                full_mat_avg[i, j] = mean_line
        return full_mat_avg
    
    def __extract_dcam(self, full_mat_avg):
        # full_mat_avg contains shape (D x D x N)

        # Mean is to filter out irrelevant window (dimension)
        means = np.mean(np.mean(full_mat_avg, 1), 0) # First averageout for each 

        variance = ((full_mat_avg - np.mean(full_mat_avg, 1)) ** 2, 1)

        return means * variance
             
    