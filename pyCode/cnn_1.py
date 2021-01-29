import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchviz import make_dot
import my_dl as dl
import argparse

# resudidula model
# net = dl.get_resnet()
net = dl.LeNet()
batch_size = 256
train_iter, test_iter = dl.load_data_fashion_mnist(batch_size, resize=None)

args = argparse.ArgumentParser()
args.add_argument('--mode',default='test')
args.add_argument('--path',default='../model_result/lenet.pt')
args.add_argument('--num_epochs',default=1,type=int)
args.add_argument('--lr',default=0.001,type=float)
config = args.parse_args()

mode = config.mode
PATH = config.path
num_epochs = config.num_epochs
lr = config.lr

print(config)
if mode=='display':
    print(net.state_dict())
elif mode=='train':
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    dl.train_ch5(net, train_iter, test_iter, batch_size, optimizer, 'cuda', num_epochs)
    torch.save(net.state_dict(),PATH)
else:
    net.load_state_dict(torch.load(PATH))
    train_acc = dl.evaluate_accuracy(train_iter, net, device='cuda')
    test_acc = dl.evaluate_accuracy(test_iter, net, device='cuda')
    print(train_acc,test_acc)