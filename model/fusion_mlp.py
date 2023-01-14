import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from utils import freeze_layer
from torch.autograd import Variable
from .fc import GroupMLP


class MLP(nn.Module):
    #args, self.train_loader.dataset, self.question_word2vec
    # def __init__(self, args, dataset, question_word2vec):
    def __init__(self, args, num_tokens, embedding_weights=None, rnn_bidirectional=True):
        super(MLP, self).__init__()
        embedding_requires_grad = not args.freeze_w2v
        question_features = 16
        vision_features = args.visual_feature_dim
        assert args.hidden_size % 60 == 0 or args.hidden_size % 64 == 0
        self.groups = 60 if args.hidden_size % 60 == 0 else 64

        # self.text = BagOfWordsMLPProcessor(
        self.text = BagOfWordsProcessor(
            embedding_tokens=embedding_weights.size(
                0) if embedding_weights is not None else num_tokens,
            embedding_weights=embedding_weights,
            embedding_features=question_features,
            embedding_requires_grad=embedding_requires_grad,
        )
        self.mlp = GroupMLP(
            in_features=vision_features + question_features,
            mid_features=args.hidden_size,
            out_features=args.embedding_size,
            drop=0.5,
            groups=self.groups,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, q, q_len):
        a = self.text(q, list(q_len.data))
        q = F.normalize(self.text(q, list(q_len.data)), p=2, dim=1)
        v = F.normalize(v, p=2, dim=1)

        combined = torch.cat([v, q], dim=1)
        embedding = self.mlp(combined)
        # embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class BagOfWordsProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features,
                 embedding_weights, embedding_requires_grad):
        super(BagOfWordsProcessor, self).__init__()
        self.embedding = nn.Embedding(
            embedding_tokens, embedding_features, padding_idx=0)
        if embedding_weights is not None:
            self.embedding.weight.data = embedding_weights
        else:
            self.embedding.weight.data.normal_(mean=0.0, std=0.2)
            # nn.init.eye(self.embedding.weight.data)
        self.embedding.weight.requires_grad = embedding_requires_grad

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        q_len = Variable(torch.Tensor(q_len).view(-1, 1) +
                         1e-12, requires_grad=False).cuda()

        return torch.div(torch.sum(embedded, 1), q_len)
