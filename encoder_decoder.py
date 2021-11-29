import torch.nn as nn
import torch

class Conv_layer(nn.Module):
    def __init__(self,inputc, outputc):
        super().__init__()
        self.conv = nn.Conv2d(inputc, outputc, 3, 1, padding =1)
        self.bn = nn.BatchNorm2d(outputc)
        self.act = nn.ReLU()
    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class Encoder(nn.Module):
    def __init__(self,start_channels,n_layers):
        super().__init__()
        layers = []
        cur_channel = start_channels
        for i in range(n_layers):
            layers.append(Conv_layer(cur_channel,cur_channel*2))
            cur_channel *= 2
        self.layers = nn.Sequential(*layers)

    def forward(self,input):
        output = self.layers(input)
        return output

class Decoder(nn.Module):
    def __init__(self,start_channels,n_layers):
        super().__init__()
        layers = []
        cur_channel = (int)(start_channels * (2**n_layers))
        for i in range(n_layers):
            # print(cur_channel//2)
            layers.append(Conv_layer(cur_channel, cur_channel // 2))
            cur_channel //= 2
        self.layers = nn.Sequential(*layers)

    def forward(self,input):
        output = self.layers(input)
        return output

class E_D_Net(nn.Module):
    def __init__(self, start_channels, layers):
        super().__init__()
        self.encoder = Encoder(start_channels,layers)
        self.decoder = Decoder(start_channels,layers)
    def forward(self,input):
        output = self.encoder(input)
        output = self.decoder(output)
        return output


    