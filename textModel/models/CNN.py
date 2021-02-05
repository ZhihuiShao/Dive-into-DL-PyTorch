import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class CNN(nn.Module):
    def __init__(self,batch_size,output_size,in_channels,out_channels,kernel_heights,stride,padding,keep_probab,vocab_size,embedding_length,weights):
        super(CNN,self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_heights = kernel_heights
        self.stride = stride
        self.padding = padding
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        print(vocab_size)
        
        self.word_embeddings = nn.Embedding(vocab_size,embedding_length)


    def forward(self, input_sentences,batch_size=None):
        input = self.word_embeddings(input_sentences)
        return input

if __name__=='__main__':
    # learning_rate = 2e-5
    # batch_size = 32
    # output_size = 2
    # hidden_size = 256
    # embedding_length = 300
    # net = CNN(batch_size,output_size,1,1,1,1,1,1,32,embedd