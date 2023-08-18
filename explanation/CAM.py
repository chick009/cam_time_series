import random
import numpy as np
from torch.autograd import Variable
import torch
from torch.nn import functional as F
from torch import topk


class SaveFeatures():
	features=None
	def __init__(self, m): 
		self.hook = m.register_forward_hook(self.hook_fn)
	def hook_fn(self, module, input, output): 
		self.features = ((output.cpu()).data).numpy()
	def remove(self): 
		self.hook.remove()

class CAM():
    
    def __init__(self, model, device, last_conv_layer, fc_layer):
        
        self.device = device
        self.last_conv_layer = last_conv_layer
        self.fc_layer = fc_layer
        self.model = model

    def predict_class(self, reshaped_instance):
        prediction = self.model(reshaped_instance) # FC Layer For Class
        pred_proba = F.softmax(prediction).data.squeeze() # Remove single dimension
        class_idx = topk(pred_proba, 1)[1].int()
        return class_idx
    
    def create_CAM(self, feature_maps, weight_softmax, label):
        
        _, nc, length = feature_maps.shape
        
        cam = weight_softmax[label].dot(feature_maps.reshape(nc, length))
        cam = cam.reshape(length)

        return cam

    def predict(self, instance):
        
        # Assuming a time series for shape (B x S x D)
        original_dim, original_length = instance.shape
        
        reshaped_instance = Variable(
            torch.tensor(
                instance.reshape(
                    (1, original_dim, original_length)
                )).float().to('cpu'),
                requires_grad = True
        )

        prediction = self.predict_class(reshaped_instance)
        
        # Generate CAM
        final_layer = self.last_conv_layer
        feature_maps = SaveFeatures(final_layer) # Return B x S x D
        feature_maps.remove() # Save the space

        # Turning m feature maps into c classes.
        # We get m x c weight parameters 
        weight_softmax_params = list(self.fc_layer.parameters())
        # Extract the weights parameters and turn (m, ) in 0th index
        weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
        # hook.features only contains conv output without weights and bias
        cam = self.create_CAM(feature_maps.feature, weight_softmax, prediction)

        return cam, prediction
     
    

