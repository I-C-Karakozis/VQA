import math
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.autograd import Variable

def load_to_device(net):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    if device == 'cuda':
        cudnn.benchmark = True
    return net, device

def initLinear(linear, val=None):
    if val is None:
        fan = linear.in_features + linear.out_features
        spread = math.sqrt(2.0) * math.sqrt(2.0/fan)
    else:
        spread = val
    linear.weight.data.uniform_(-spread, spread)
    linear.bias.data.uniform_(-spread, spread)

def create_emb_layer(weights_matrix):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    emb_layer.weight.requires_grad = False

    return emb_layer, embedding_dim

class LSTM(nn.Module):
    def __init__(self, weights_matrix, img_feat_dim, hidden_dim, n_classes, batch_size, num_layers):
        super(LSTM, self).__init__()
        print('Initializing LSTM: {} hidden_dim | {} num_layers.'.format(hidden_dim, num_layers))
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size

        emb_layer, embedding_dim = create_emb_layer(weights_matrix)
        self.word_embedding = emb_layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)

        self.img_linear = nn.Linear(img_feat_dim, hidden_dim)
        initLinear(self.img_linear)

        self.cls = nn.Linear(hidden_dim, n_classes)
        initLinear(self.cls)

    def set_device(self, device):
        self.device = device

    def init_hidden_state(self):
        hidden_a = torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
        hidden_a = torch.FloatTensor(hidden_a).to(self.device)
        hidden_b = torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
        hidden_b = torch.FloatTensor(hidden_b).to(self.device)
        return (Variable(hidden_a), Variable(hidden_b))

    def forward(self, sentence, img_features, only_img=False, only_q=False):
        # get question features
        embeds = self.word_embedding(sentence)
        init_hidden = self.init_hidden_state()
        hidden_states, _ = self.lstm(embeds, init_hidden)
        last_hidden_state = hidden_states[:, -1, :]

        # get image features
        img_features = self.img_linear(img_features)

        # feature fusion
        if only_img: features = img_features
        elif only_q: features = last_hidden_state
        else:        features = last_hidden_state + img_features

        # get network outputs
        cls_scores = self.cls(features.view(len(sentence), -1))
        return cls_scores
