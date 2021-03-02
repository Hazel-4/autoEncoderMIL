# coding=utf-8
import os
import random
import numpy as np


def get_img_infos(img_info_txt):
    '''读取txt中存储的图片信息
    :return: 图片名称img_name, 图片路径img_path, 图片标签img_label
    '''
    with open(img_info_txt) as input_file:
        lines = input_file.readlines()
        img_name = [line.strip().split(',')[0] for line in lines]
        img_path = [line.strip().split(',')[1] for line in lines]
        img_label = [line.strip().split(',')[-1] for line in lines]

    return img_name, img_path, img_label


def split_train_test(img_name, img_label):
    """
    函数功能: 先获得所有包名，根据包名划分数据集
    """
    # 保存腐腻名
    funi_names = []
    # 保存非腐腻名
    no_funi_names = []

    size = len(img_name)
    for i in range(size):
        label = img_label[i]
        bag_name = img_name[i]
        if label == "funi":
            funi_names.append(bag_name)
        elif label == "no_funi":
            no_funi_names.append(bag_name)
    funi_names = np.unique(funi_names)
    no_funi_names = np.unique(no_funi_names)
    # 打乱顺序
    np.random.shuffle(funi_names)
    np.random.shuffle(no_funi_names)

    funi_names = list(funi_names)
    no_funi_names = list(no_funi_names)

    print('腐腻包的数量={},非腐腻包的数量={}'.format(len(funi_names), len(no_funi_names)))

    # funi_train_len = int(len(funi_names)*0.5)
    # nofuni_train_len = int(len(no_funi_names)*0.5)
    #
    # train = funi_names[0: funi_train_len]
    # train.extend(no_funi_names[0: nofuni_train_len])
    #
    # test = funi_names[funi_train_len: ]
    # test.extend(no_funi_names[nofuni_train_len: ])

    '''
        将腐腻和非腐腻分别均分成5份
    '''


    split_size = 5
    # 每一份的大小
    each_funi_size = int(len(funi_names) / split_size)
    print(len(funi_names))
    # 平均分成5份(每份17个，还剩一个，分别加在第0上面)
    split_funi_names = [funi_names[i:i + each_funi_size] for i in range(0, len(funi_names), each_funi_size)]
    for i in range(0):
        split_funi_names[i].append(split_funi_names[5][i])
    # 删除最后一个元素
    split_funi_names.pop()
    print('腐腻切分：%d,%d,%d,%d,%d' % (len(split_funi_names[0]), len(split_funi_names[1]), len(split_funi_names[2]), len(split_funi_names[3]),len(split_funi_names[4])))

    each_no_funi_size = int(len(no_funi_names) / split_size)
    # 平均分成5份(每份37个，还剩3个，加在第0，1，2上面)
    split_no_funi_names = [no_funi_names[i:i + each_no_funi_size] for i in range(0, len(no_funi_names), each_no_funi_size)]
    for i in range(3):
        split_no_funi_names[i].append(split_no_funi_names[5][i])
    split_no_funi_names.pop()
    print('非腐腻切分：%d,%d,%d,%d,%d' % (len(split_no_funi_names[0]), len(split_no_funi_names[1]), len(split_no_funi_names[2]), len(split_no_funi_names[3]), len(split_no_funi_names[4])))


    '''
        将切分后的腐腻数据和非腐腻数据整合，对应位置相加
    '''
    split_names = []
    for i in range(5):
        dataset = split_funi_names[i] + split_no_funi_names[i]
        split_names.append(dataset)
    print('腐腻和非腐腻对应位置相加后：%d,%d,%d,%d,%d' % (len(split_names[0]), len(split_names[1]), len(split_names[2]), len(split_names[3]), len(split_names[4])))


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

    # # 整合后的数据集
    # cross_names = []
    # for i in range(5):
    #     data2 = []
    #     data1 = train_names[i]
    #     data2.append(data1)
    #     data2.append(test_names[i])
    #
    #     cross_names.append(data2)
    # print(len(cross_names[0][0]))




    return train_names, test_names


def generate_train_test_txt(save_txt_path, train, test, img_name, img_path, img_label):
    train_ins_num = 0
    test_ins_num = 0

    # 训练集 instance 数量
    with open(save_txt_path + "/train.txt", mode="w", encoding="utf-8") as f_wirter:
        for i in range(len(img_name)):
            bag_name = img_name[i]
            if bag_name in train:
                train_ins_num += 1
                f_wirter.write("%s,%s,%s\n" % (img_name[i], img_path[i], img_label[i]))
        f_wirter.close()

    with open(save_txt_path + "/test.txt", mode="w", encoding="utf-8") as f_wirter:
        for i in range(len(img_name)):
            bag_name = img_name[i]
            if bag_name in test:
                test_ins_num += 1
                f_wirter.write("%s,%s,%s\n" % (img_name[i], img_path[i], img_label[i]))
        f_wirter.close()

    print('训练集path数量={},测试集path数量={}'.format(train_ins_num, test_ins_num))

    return train_ins_num, test_ins_num


if __name__ == "__main__":
    img_name, img_path, img_label = get_img_infos("./data/image.txt")
    train_names, test_names = split_train_test(img_name, img_label)
    generate_train_test_txt("./data", train_names[0], test_names[0], img_name, img_path, img_label)




