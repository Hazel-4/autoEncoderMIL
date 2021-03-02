from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from dataloader import dataLoader
from model import Attention, GatedAttention
import util_data
import os
import Attention
import torch.nn as nn
from dataloader import UtilData
from torchvision import models, transforms


os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-4, metavar='R',
                    help='weight decay')
parser.add_argument('--batch_size', type=int, default=16, help="batch size for training")
parser.add_argument('--dim_insVec', type=int, default=2048, help="dimension for instance vector")
parser.add_argument('--dim_attenLatent', type=int, default=32, help="latent dimension for Attention")
parser.add_argument('--dim_autoLatten', type=int, default=50, help="latent dimension for autoencoder")
parser.add_argument('--is_bn', type=bool, default=True, help="if using bn layer in CAE")
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}



# model.feature.eval()

# optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = 0.09)

# print('Init Model')
# if args.model=='attention':
#     model = Attention()
# elif args.model=='gated_attention':
#     model = GatedAttention()






def train(epoch, train_loader):
    model.train()
    train_loss = 0.
    train_error = 0.

    TP = 0
    FN = 0
    TN = 0
    FP = 0

    for batch_idx, (data, label) in enumerate(train_loader):
        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)
        # reset gradients
        optimizer.zero_grad()                                      # 先将网络中的所有梯度置0
        # calculate loss and metrics
        output = model(data, label, True)
        pred = output.max(1, keepdim=True)[1]

        loss = criteration(output, label)
        loss = 0.9 * loss + 0.1 * torch.mean(model.autofeature.loss)
        loss.backward()
        optimizer.step()




        # 计算TPR,TNR
        # bag_level = (label.cpu().data.numpy(), pred.cpu().data.numpy().squeeze(1))

        label = label.cpu().data.numpy()
        pred = pred.cpu().data.numpy().squeeze(1)
        for i in range(args.batch_size):
            if label[i] == 1 and pred[i]  == 1:
                TP += 1
            elif label[i] == 1 and pred[i] == 0:
                FN += 1
            elif label[i] == 0 and pred[i] == 0:
                TN += 1
            elif label[i] == 0 and pred[i] == 1:
                FP += 1
        print("Step {} Loss {:.4f}".format(batch_idx, loss))


    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    correct = TP + TN
    train_num = TP + FN + TN + FP
    print("Epoch {} Accuracy {}/{} ({:.2f}%) TPR={:.2f}% TNR={:.2f}%".format(epoch, correct, train_num,
                                                                     100 * correct / train_num, 100*TPR, 100*TNR))
        # loss, _ = model.calculate_objective(data, bag_label)
        # train_loss += loss.data[0]
        # error, _ = model.calculate_classification_error(data, bag_label)
        # train_error += error
        # # backward pass
        # loss.backward()
        # # step
        # optimizer.step()

    # calculate loss and error for epoch
    # train_loss /= len(train_loader)
    # train_error /= len(train_loader)

    # print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy()[0], train_error))


def test(test_loader):
    model.eval()

    TP = 0
    FN = 0
    TN = 0
    FP = 0

    res = []

    for batch_idx, (data, label) in enumerate(test_loader):
        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)
        # data = data.squeeze(0)

        output = model(data, label, False)

        pred = output.max(1, keepdim=True)[1]

        # 计算准确率
        label = label.cpu().data.numpy()
        pred = pred.cpu().data.numpy().squeeze(1)
        for i in range(len(label)):
            if label[i] == 1 and pred[i] == 1:
                TP += 1
            elif label[i] == 1 and pred[i] == 0:
                FN += 1
            elif label[i] == 0 and pred[i] == 0:
                TN += 1
            elif label[i] == 0 and pred[i] == 1:
                FP += 1


    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    correct = TP + TN
    test_num = TP + FN + TN + FP

    res.append(correct/test_num)
    res.append(TPR)
    res.append(TNR)


    print("Accuracy {}/{} ({:.2f}%), TPR {:.2f}%, TNR {:.2f}%".format(correct, test_num,
                                                                correct / test_num * 100, 100*TPR, 100*TNR))
    return res

if __name__ == "__main__":
    utilData = UtilData("./data/image.txt")
    train_names = utilData.train_names
    test_names = utilData.test_names
    img_info = utilData.img_info

    resList = []


    for i in range(5):
        model = Attention.Attention(args)
        if args.cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
        criteration = nn.CrossEntropyLoss()

        print('cross-validation: {}'.format(i))

        train_loader = data_utils.DataLoader(dataLoader(train_name=train_names[i],
                                                       test_name=test_names[i],
                                                       img_info=img_info,
                                                       train=True,
                                                       data_transforms=transforms.Compose([
                                                           transforms.RandomResizedCrop(224),
                                                           transforms.RandomHorizontalFlip(),
                                                           transforms.ToTensor()
                                                           # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                       ])
                                                       ),
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             drop_last=True,
                                             **loader_kwargs)

        test_loader = data_utils.DataLoader(dataLoader(train_name=train_names[i],
                                                       test_name=test_names[i],
                                                       img_info=img_info,
                                                       train=False,
                                                      data_transforms=transforms.Compose([
                                                          transforms.RandomResizedCrop(224),
                                                          transforms.RandomHorizontalFlip(),
                                                          transforms.ToTensor()
                                                          # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                      ])
                                                      ),
                                            batch_size=1,
                                            shuffle=False,
                                            **loader_kwargs)
        print('Start Training')

        for epoch in range(1, args.epochs + 1):
            train(epoch, train_loader)
        print('Start Testing')
        res = test(test_loader)
        resList.append(res)

    ACC_mean = .0
    TPR_mean = .0
    TNR_mean = .0

    for i in range(5):
        ACC_mean += resList[i][0]
        TPR_mean += resList[i][1]
        TNR_mean += resList[i][2]
        print("fold_{}: Accuracy {:.2f}% TPR {:.2f}% TNR {:.2f}%".format(i, 100*resList[i][0], 100*resList[i][1], 100*resList[i][2]))


    print("ACC_mean {:.2f}%, TPR_mean {:.2f}%, TNR_mean {:.2f}%".format(100*ACC_mean/5, 100*TPR_mean/5, 100*TNR_mean/5))
