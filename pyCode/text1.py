import torch
import random
import zipfile
import my_dl as dl


if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    corpus_chars, idx_to_char, char_to_idx, corpus_ints = dl.load_text_from_zip('../data/jaychou_lyrics.txt.zip',cap_len = 10000)
    print(len(corpus_ints))
    print(corpus_ints[:10])
    corpus_ints = list(range(30))
    random_iter = dl.data_iter_random(corpus_ints,1,4,'cuda')
    for X,y in random_iter:
        print(X.size(),y.size())
        print(X,y)
        break
    other_iter = dl.data_iter(corpus_ints,1,4,'cuda')
    for X,y in other_iter:
        print(X.size(),y.size())
        print(X,y)
        break