import torch
import numpy as np

from tools.vqa import VQA

def get_dataloaders(batch_size, trainset, testset):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)  
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def get_data(glove_word2idx, top_n_answers):
    trainset = VQA(glove_word2idx, top_n_answers, split=0)
    testset  = VQA(glove_word2idx, top_n_answers, split=1, id2answer=trainset.id2answer)

    return trainset, testset

def load_GloVe(path='data/glove.6B.300d.txt'):
    print('Loading {} embedding.'.format(path))
    words = []
    idx = 0
    word2idx = {}
    vectors = []

    with open(path, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0].lower()
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    return words, word2idx, vectors
