import torch
from torch import Tensor
import torch.nn as nn

from auto_feature import AutoFeature
from models.utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F
import models
class Attention(nn.Module):

    def __init__(self, args):
        self.L = args.dim_insVec+1
        self.D = args.dim_attenLatent
        self.K = 1
        self.batch_size = args.batch_size
        super(Attention, self).__init__()

        self.autofeature = AutoFeature(args)
        self.fc = nn.Sequential(
            nn.Linear(self.L, 2)
        )
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        # 门注意力机制
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)


    # 注意力机制
    def forward(self, x: Tensor, label, train) -> Tensor:
        y = self.autofeature(x, label, train)
        size = y.shape[0]
        y = y.permute(0, 2, 1).contiguous()
        res = []
        for i in range(size):
            data = y[i, :, :]
            A = self.attention(data)        # torch.Size([5, 1])           NxK
            A = torch.transpose(A, 1, 0)    # torch.Size([1, 5])           KxN  交换维度
            A = F.softmax(A, dim=1)         # softmax over N

            M = torch.mm(A, data)  # torch.Size([1, 2048])  KxL

            res.append(M)

        M = torch.cat(res, dim=0)

        # 新增attention
        # A = self.attention(y)         # torch.Size([5, 1])           NxK
        # A = torch.transpose(A, 1, 0)  # torch.Size([1, 5])           KxN  交换维度
        # A = F.softmax(A, dim=1)       # softmax over N

        # M = torch.mm(A, y)  # torch.Size([1, 2048])  KxL

        y = self.fc(M)                # torch.Size([1, 2])

        return y


    # 门注意力机制
    # def forward(self, x: Tensor) -> Tensor:
    #     y = self.feature(x)
    #
    #     A_V = self.attention_V(y)  # NxD
    #     A_U = self.attention_U(y)  # NxD
    #     A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
    #     A = torch.transpose(A, 1, 0)  # KxN
    #     A = F.softmax(A, dim=1)  # softmax over N
    #
    #     M = torch.mm(A, y)  # torch.Size([1, 2048])  KxL
    #
    #     y = self.feature.fc(M)  # torch.Size([1, 2])
    #
    #     return y

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Attention()
    model = model.to(device)
    input = torch.randn(5, 3, 224, 224).to(device)
    out = model(input)
    print(out.shape)
