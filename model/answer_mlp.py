import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import model.fc as FC
from .fc import GroupMLP, GroupMLP_2lay, GroupMLP_1lay


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        ans_net_list = ["GroupMLP", "GroupMLP_1lay", "GroupMLP_2lay"]
        ans_net = ans_net_list[args.ans_net_lay]

        assert args.hidden_size % 60 == 0 or args.hidden_size % 64 == 0
        self.groups = 60 if args.hidden_size % 60 == 0 else 64

        self.mlp = getattr(FC, ans_net)(
            in_features=args.KG_feature_dim,
            mid_features=args.hidden_size,
            out_features=args.embedding_size,
            drop=0.0,
            groups=self.groups,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, a, a_len=None):
        # pdb.set_trace()
        return self.mlp(F.normalize(a, p=2))
