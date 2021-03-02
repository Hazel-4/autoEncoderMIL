from __future__ import print_function, division
from torch.utils.data import Dataset

from PIL import Image
import numpy as np
import torch
from torchvision import transforms



class UtilData():
    def __init__(self, txt_path):
        self.txt_path = txt_path
        self.name, self.path, self.label = self._get_bag_name()
        self.train_names, self.test_names = self._split_train_test()
        self.img_info = self._get_info()

    def _get_bag_name(self):
        '''读取txt中存储的图片信息
        :return: 图片名称img_name, 图片路径img_path, 图片标签img_label
        '''
        # 获取Instance
        with open(self.txt_path) as input_file:
            lines = input_file.readlines()
            name = [line.strip().split(',')[0] for line in lines]
            path = [line.strip().split(',')[1] for line in lines]
            label = [line.strip().split(',')[-1] for line in lines]


        return name, path, label

    def _get_info(self):
        '''
        {name:[path][label]}
        :return: key为name的字典
        '''
        img_info = {}
        for i in range(len(self.name)):
            path_label = (self.path[i], self.label[i])
            img_info[self.name[i]] = path_label

        return img_info




    def _split_train_test(self):
        funi_names = []
        no_funi_names = []

        for i in range(len(self.name)):
            if self.label[i] == "funi":
                funi_names.append(self.name[i])
                self.label[i] = 1
            elif self.label[i] == "no_funi":
                no_funi_names.append(self.name[i])
                self.label[i] = 0
        print('腐腻舌的数量={},非腐腻舌的数量={}'.format(len(funi_names), len(no_funi_names)))

        # 打乱顺序
        np.random.shuffle(funi_names)
        np.random.shuffle(no_funi_names)

        funi_names = list(funi_names)
        no_funi_names = list(no_funi_names)


        split_size = 5
        # 每一份的大小
        each_funi_size = int(len(funi_names) / split_size)
        # 平均分成5份(每份10个)
        split_funi_names = [funi_names[i:i + each_funi_size] for i in range(0, len(funi_names), each_funi_size)]
        # for i in range(3):
        #     split_funi_names[i].append(split_funi_names[5][i])
        # # 删除最后一个元素
        # split_funi_names.pop()
        print('腐腻切分：%d,%d,%d,%d,%d' % (
        len(split_funi_names[0]), len(split_funi_names[1]), len(split_funi_names[2]), len(split_funi_names[3]),
        len(split_funi_names[4])))


        each_no_funi_size = int(len(no_funi_names) / split_size)
        # 平均分成5份(每份28个，还剩3个，加在第1,2,3上面)
        split_no_funi_names = [no_funi_names[i:i + each_no_funi_size] for i in
                               range(0, len(no_funi_names), each_no_funi_size)]
        for i in range(3):
            split_no_funi_names[i].append(split_no_funi_names[5][i])
        # 删除最后一个元素
        split_no_funi_names.pop()

        print('非腐腻切分：%d,%d,%d,%d,%d' % (
        len(split_no_funi_names[0]), len(split_no_funi_names[1]), len(split_no_funi_names[2]),
        len(split_no_funi_names[3]), len(split_no_funi_names[4])))

        '''
            将切分后的腐腻数据和非腐腻数据整合，对应位置相加
        '''
        split_names = []
        for i in range(5):
            dataset = split_funi_names[i] + split_no_funi_names[i]
            split_names.append(dataset)
        print('腐腻和非腐腻对应位置相加后：%d,%d,%d,%d,%d' % (
        len(split_names[0]), len(split_names[1]), len(split_names[2]), len(split_names[3]), len(split_names[4])))


        '''
            四份整合成训练集，剩余一份作为测试集
        '''
        # 训练集
        train_names = []
        # 测试集
        test_names = []
        for i in range(5):
            train_data = []
            for j in range(5):
                if j != i:
                    data1 = split_names[j]
                    train_data.extend(data1)
            train_names.append(train_data)
            data2 = split_names[i]
            test_names.append(data2)
        print('整合后训练集：%d,%d,%d,%d,%d' % (
            len(train_names[0]), len(train_names[1]), len(train_names[2]), len(train_names[3]), len(train_names[4])))
        print('整合后测试集：%d,%d,%d,%d,%d' % (
            len(test_names[0]), len(test_names[1]), len(test_names[2]), len(test_names[3]), len(test_names[4])))

        for i in range(5):
            np.random.shuffle(train_names[i])
            np.random.shuffle(test_names[i])

        return train_names, test_names





"""
    加载图片文件
"""
class dataLoader(Dataset):
    def __init__(self, train_name, test_name, img_info, train=True, data_transforms=None):
        self.train_name = train_name
        self.test_name = test_name
        self.img_info = img_info
        self.train = train
        self.data_transforms = data_transforms

    def __len__(self):
        if self.train:
            return len(self.train_name)
        else:
            return len(self.test_name)

    def __getitem__(self, index):
        if self.train:
            name = self.train_name[index]
            path = self.img_info[name][0]
            label = self.img_info[name][1]
            img = Image.open(path)
            img = self.data_transforms(img.convert('RGB'))
        else:
            name = self.test_name[index]
            path = self.img_info[name][0]
            label = self.img_info[name][1]
            img = Image.open(path)
            img = self.data_transforms(img.convert('RGB'))
        return img, label


if __name__ == "__main__":
    utilData = UtilData("./data/image.txt")
    train_names = utilData.train_names
    test_names = utilData.test_names
    img_info = utilData.img_info

    loader = dataLoader(train_name=train_names[0],
                        test_name=test_names[0],
                        img_info=img_info,
                        train=True,
               data_transforms=transforms.Compose([
                   transforms.RandomResizedCrop(224),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor()
                   # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
               ]))
    dataloaders = torch.utils.data.DataLoader(loader, batch_size=16, shuffle=True)

