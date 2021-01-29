import time
import torch
from torch import nn, optim
import my_dl as dl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vgg_block(num_convs,in_channels,out_channels):
    blk = []
    for i in range(num_convs):
        if i==0: # first block
            blk.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        else:
            blk.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*blk)


def vgg(conv_arch,fc_features,fc_hidden_units=50):
    net = nn.Sequential()
    # conv layers
    for i, (num_convs,in_channels,out_channels) in enumerate(conv_arch):
        net.add_module('vgg_block_'+str(i),vgg_block(num_convs,in_channels,out_channels))
    
    net.add_module('fc',nn.Sequential(dl.FlattenLayer(),nn.Linear(512,fc_features),nn.ReLU(),nn.Dropout(0.5),nn.Linear(fc_features,fc_hidden_units),nn.ReLU(),nn.Dropout(0.5),nn.Linear(fc_hidden_units,10)))
    return net


if __name__=='__main__':
    conv_arch = ((1,1,64),(1,64,128),(2,128,256),(2,256,512))
    net = vgg(conv_arch,200,100)
    
    batch_size = 64
    train_iter, test_iter = dl.load_data_fashion_mnist(batch_size, resize=None)
    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    dl.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)