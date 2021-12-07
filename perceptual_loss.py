from torchvision.models import vgg16_bn
import torchvision.transforms as T
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import math
import os

class FeatureHook:
    def __init__(self, layer):
        self.feature = None
        self.hook = layer.register_forward_hook(self._my_hook)

    def _my_hook(self, module, inputs, output):
        self.feature = output
    
    def close(self):
        self.hook.remove()

class VGG_perceptual_loss:
    def __init__(self, block_ids, block_ws= None, device = 'cpu'):
        if block_ws == None:
            block_ws = [1 for block in block_ids]
        vgg = vgg16_bn(pretrained = True)
        features = vgg.features
        features.eval()
        # for param in features.parameters():
        #     param.require_grad = False

        #Ids of conv layers before pooling
        feature_ids = [i-2 for i, layer in enumerate(features) if isinstance(layer, nn.MaxPool2d)]
        #remove the useless block to save compute time
        valid_feature_ids = [feature_ids[i] for i in block_ids]
        self.feature_net = features[:valid_feature_ids[-1]+1]
        self.device = device
        self.feature_net = self.feature_net.to(self.device)
        self.block_ws = torch.tensor(block_ws).to(self.device)
        self.hooks = [FeatureHook(self.feature_net[i]) for i in valid_feature_ids]

    def compute_loss(self, input, target, loss_function):
        #normalize the pixels to ImageNet statistics pre-trained VGG
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        normalizer = T.Normalize(mean=mean, std=std)
        input = input.to(self.device)
        input = normalizer(input)
        target = target.to(self.device)
        target = normalizer(target)
        
        self.feature_net(input)
        input_features = [hook.feature.clone() for hook in self.hooks]

        self.feature_net(target)
        target_features = [hook.feature.clone() for hook in self.hooks]
       
        loss = 0
        num_features = len(input_features)
        for i in range(num_features):
            loss += loss_function(input_features[i],target_features[i])*self.block_ws[i]
            # print(input_hooks[i].feature)
            # print(output_hooks[i].feature)

        return input_features, target_features, loss


def test():
    transformer = T.Compose([
        T.CenterCrop(512),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    transformer2 = torch.nn.Sequential(
        T.CenterCrop(256),
        # T.PILToTensor(), not subclass of nn.Module, so fail
        T.ConvertImageDtype(torch.float),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )
    dog1 = Image.open('../dog1.jpeg')
    dog1 = transformer(dog1)[None,:]

    dog2 = Image.open('../dog2.jpeg')
    dog2 = transformer(dog2)[None,:]

    human1 = Image.open('../human1.jpeg')
    human1 = transformer(human1)[None,:]
    
    # print(image1.size())
    # plt.imshow(image1)
    # plt.show()

    block_ids = [0,1,2,3,4]
    # block_ws = [0.5,0.5,0.5,0.5,0.5]
    device = (torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
    losser = VGG_perceptual_loss(block_ids, device = device)
    dog1_features, dog2_features, dogs_loss = losser.compute_loss(dog1, dog2, F.l1_loss)
    print(dogs_loss)
    # human1_features, dog1_features, human_dog_loss = losser.compute_loss(human1, dog1, F.l1_loss)
    # print(human_dog_loss)
    for layer in range(len(dog1_features)):
        feature = dog1_features[layer][0]
        channel = feature.shape[0]
        result_dir = '../dog1_{}'.format(layer)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        for i in range(channel):
            result_path = os.path.join(result_dir, 'dog1_layer{}_{}.png'.format(layer, i))
            # plt.imsave(result_path,feature[i].detach().numpy(), cmap = 'gray')
            result = Image.fromarray((feature[i].detach().numpy()*255).astype(np.uint8))
            result.save(result_path)

        #subplot image
        # rows = math.ceil(temp_c/10)
        # fig, ax = plt.subplots(nrows = rows, ncols = 10)
        # for i,row in enumerate(ax):
        #     for j,col in enumerate(row):
        #         img_idx = i*10+j
        #         if img_idx < temp_c:
        #             col.imshow(feature[img_idx].detach().numpy())
        #             col.axis('off')
        # plt.axis('off')
        # plt.savefig('../dog1_layer_{}.png'.format(layer), bbox_inches='tight')
        # plt.show()



if __name__ == '__main__':
    test()