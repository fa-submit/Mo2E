import sys, os
import numpy as np
import torch
from . import bit_models
from torchvision.models import mobilenet_v2
from utils.model_saving_loading import load_model_tail_training

def get_arch(model_name, in_c=3, n_classes=1):

    if model_name == 'mobilenetV2':
        model = mobilenet_v2(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier = torch.nn.Linear(num_ftrs, n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
              
    elif model_name == 'efficientnet_b4':
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
        model.classifier.fc = torch.nn.Linear(model.classifier.fc.in_features, n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        
    elif model_name == 'bit_resnext50_1':
        bit_variant = 'BiT-M-R50x1'
        model = bit_models.KNOWN_MODELS[bit_variant](head_size=n_classes, zero_head=True)
        if not os.path.isfile('models/BiT-M-R50x1.npz'):
            print('downloading bit_resnext50_1 weights:')
            os.system('wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz -P models/')
        model.load_from(np.load('models/BiT-M-R50x1.npz'))
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        
    else:
        sys.exit('not a valid model_name, check models.get_model.py')
    setattr(model, 'n_classes', n_classes)
    return model, mean, std

def get_trained_arch(model_name, weights_path, n_classes=1):

    if model_name == 'mobilenetV2':
        model = mobilenet_v2()
        num_ftrs = model.classifier[1].in_features
        model.classifier = torch.nn.Linear(num_ftrs, 17)
        model, stats = load_model_tail_training(model, weights_path, device='cpu', with_opt=False)
        model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
              
    elif model_name == 'efficientnet_b4':
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
        model.classifier.fc = torch.nn.Linear(model.classifier.fc.in_features, 17)
        model, stats = load_model_tail_training(model, weights_path, device='cpu', with_opt=False)
        model.classifier.fc = torch.nn.Linear(model.classifier.fc.in_features, n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        
    if model_name == 'bit_resnext50_1':
        bit_variant = 'BiT-M-R50x1'
        model = bit_models.KNOWN_MODELS[bit_variant](head_size=17, zero_head=True)
        model, stats = load_model_tail_training(model, weights_path, device='cpu', with_opt=False)
        model.head.conv = torch.nn.Conv2d(2048, 7, 1, 1)
        # if not os.path.isfile('models/BiT-M-R50x1.npz'):
        #     print('downloading bit_resnext50_1 weights:')
        #     os.system('wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz -P models/')
        # model.load_from(np.load('models/BiT-M-R50x1.npz'))
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        
    else:
        sys.exit('not a valid model_name, check models.get_model.py')
    setattr(model, 'n_classes', n_classes)
    return model, mean, std

