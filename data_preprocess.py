#-*-coding:utf-8-*- 
"""
  author: lw
  email: hnu-lw@foxmail.com
  data: 2019/7/1 下午2:20 
  description:对标签数据和图像数据进行预处理
"""
from scipy.io import loadmat
import numpy as np
import os
import glob
import shutil
import tqdm
import collections
from utils.utils import *


train_data_path = '../../data/custom/ps2.0/training'
test_data_path = '../../data/custom/ps2.0/testing/outdoor-slanted'
train_imgs_path = '../../data/custom/train/images'
test_imgs_path = '../../data/custom/test/images'
train_label_path='../../data/custom/train/labels'
test_label_path='../../data/custom/test/labels'
ground_truth_path = '../output/ground-truth'
train_label_points_path='../../data/custom/train/labels'
val_label_points_path='../../data/custom/train/labels'
gt_parired_marking_points_path = '../output/paired_marking_points/ground-truth'
gt_parking_slot = '../output/parking_slot/outdoor-slanted/ground-truth'
os.makedirs(train_label_path, exist_ok=True)
os.makedirs(gt_parired_marking_points_path, exist_ok=True)
os.makedirs(gt_parking_slot, exist_ok=True)
train_imgs_list = glob.glob(train_data_path+'/*.jpg')
print('Total training images are {}'.format(len(train_imgs_list)))
test_imgs_list = glob.glob(test_data_path+'/*.jpg')
print('Total testing images are {}'.format(len(test_imgs_list)))
train_m_list = glob.glob(train_data_path+'/*.mat')
test_m_list = glob.glob(test_data_path+'/*.mat')
train_per=0.9
valid_per=0.1
classes = load_classes('./classes-4.names')
#  将图像数据拷贝到另一个文件夹中
# for img_path in tqdm.tqdm(train_imgs_list):
#     shutil.copy(img_path, imgs_path)

# 将xywh转换为x1y1x2y2
def wh_to_xy(x, y, w, h):
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2= y + h/2
    return x1, y1, x2, y2
# 生成ground truth of paired marking points
def save_gt_paired_marking_points(m_list):
    for m_path in tqdm.tqdm(m_list):
        m=loadmat(m_path)
        num_boxes = m['slots'].shape[0]
        m_name = m_path.split('/')[-1]
        file_name = m_name.split('.')[0] + '.txt'
        # 保存生成的标签
        if num_boxes != 0:  # 排除没有parking slot的图片
            file = open(os.path.join(gt_parired_marking_points_path, file_name), 'w')
            for i in range(num_boxes):
                point_1 = m['marks'][m['slots'][i, 0] - 1, :]
                point_2 = m['marks'][m['slots'][i, 1] - 1, :]
                if point_1[1] < point_2[1]:  # 确保最小的y为第一个点
                    x1 = point_1[0]
                    y1 = point_1[1]
                    x2 = point_2[0]
                    y2 = point_2[1]
                else:
                    x1 = point_2[0]
                    y1 = point_2[1]
                    x2 = point_1[0]
                    y2 = point_1[1]
                box_1 = list((x1, y1, x2, y2))
                for j in box_1:  # 如果一张图片中有多个parking slots, 则每一个parking slot为1行
                    file.write(str(j))
                    file.write(' ')
                file.write('\n')
            file.close()

# 生成ground truth of parking slot
def save_gt_parking_slot(m_list):
    for m_path in tqdm.tqdm(m_list):
        m=loadmat(m_path)
        num_boxes = m['slots'].shape[0]
        m_name = m_path.split('/')[-1]
        file_name = m_name.split('.')[0] + '.txt'
        # 保存生成的标签
        if num_boxes != 0:  # 排除没有parking slot的图片
            file = open(os.path.join(gt_parking_slot, file_name), 'w')
            for i in range(num_boxes):
                label = int(m['slots'][i, 2] - 1)  # 标签
                point_1 = m['marks'][m['slots'][i, 0] - 1, :]
                point_2 = m['marks'][m['slots'][i, 1] - 1, :]
                if point_1[1] < point_2[1]:  # 确保最小的y为第一个点
                    x1 = point_1[0]
                    y1 = point_1[1]
                    x2 = point_2[0]
                    y2 = point_2[1]
                else:
                    x1 = point_2[0]
                    y1 = point_2[1]
                    x2 = point_1[0]
                    y2 = point_1[1]
                box_1 = list((label, x1, y1, x2, y2))
                for j in box_1:  # 如果一张图片中有多个parking slots, 则每一个parking slot为1行
                    file.write(str(j))
                    file.write(' ')
                file.write('\n')
            file.close()
# 生成用于检测器性能评测的ground truth
def save_ground_truth(m_list):
    for m_path in tqdm.tqdm(m_list):
        m=loadmat(m_path)
        num_boxes = m['slots'].shape[0]
        num_points = m['marks'].shape[0]
        m_name = m_path.split('/')[-1]
        file_name = m_name.split('.')[0] + '.txt'
        # 保存生成的标签
        file = open(os.path.join(ground_truth_path, file_name), 'w')
        if num_boxes != 0:  # 排除没有parking slot的图片
            for i in range(num_boxes):
                boxs = []
                label = classes[int(m['slots'][i,2]-1)]  # 标签
                point_1 = m['marks'][m['slots'][i, 0] - 1, :]
                point_2 = m['marks'][m['slots'][i, 1] - 1, :]
                w = abs(point_2[0] - point_1[0])+48
                h = abs(point_2[1] - point_1[1])+44
                l_x = point_1[0] + (point_2[0] - point_1[0])/2
                l_y = point_1[1] + (point_2[1] - point_1[1])/2
                # box_1 = list((label, l_x, l_y, w, h))  # 归一化后存入列表里
                x1, y1, x2, y2 = wh_to_xy(l_x, l_y, w, h)
                box_1 = list((label, x1, y1, x2, y2))
                for j in box_1:  # 如果一张图片中有多个parking slots, 则每一个parking slot为1行
                    file.write(str(j))
                    file.write(' ')
                file.write('\n')
            # file.close()

        if num_points != 0:
            for i in range(num_points):
                point = m['marks'][i, :]
                x1, y1, x2, y2 = wh_to_xy(point[0], point[1], 40, 60)
                box = list(('point', x1, y1, x2, y2))
                for j in box:
                    file.write(str(j))
                    file.write(' ')
                file.write('\n')
            file.close()

# 数据的部分可视化
def show_data(m_list):
    num_emperty=0
    angles=[]
    for m_path in m_list:
        m = loadmat(m_path)
        num_boxes = m['slots'].shape[0]
        num_points = m['marks'].shape[0]
        if num_points == 0:
            # print('There is not parking slot in {} image'.format(m_path.split('/')[-1]))
            num_emperty += 1
        elif num_boxes !=0:
            for i in range(num_boxes):
                angle = m['slots'][i, 3]  # 角度
                # if i==0:
                angles.append(angle)
                label = m['slots'][i, 2] - 1
    print('Total {} images have not marking points'.format(num_emperty))
    return angles, num_emperty

# 保存生成的标签,只生成所有marking points的标签,用于跟踪,统一为40*40
def save_label_bbx_points(m_list, train_point):
    num_img = 0
    for m_path in tqdm.tqdm(m_list):
        m=loadmat(m_path)
        num_points = m['marks'].shape[0]
        m_name = m_path.split('/')[-1]
        file_name = m_name.split('.')[0] + '.txt'
        # 保存要训练集和验证集中图片的路径至train.txt和valid.txt
        if num_img < train_point:
            file = open(os.path.join(train_label_points_path, file_name), 'w')
        else:
            file = open(os.path.join(val_label_points_path, file_name), 'w')
        num_img +=1
        if num_points != 0:
            for i in range(num_points):
                point = m['marks'][i, :]
                box = list((0, point[0] / 600, point[1] / 600, 40 / 600, 40 / 600))  # 归一化后存入列表里
                for j in box:  # 如果一张图片中有多个parking slots, 则每一个parking slot为1行
                    file.write(str(j))
                    file.write(' ')
                file.write('\n')
        file.close()

# 保存生成的标签-v2.0:以车位的两个顶点的中点,宽和高为一组(即一个bounding box),分别为:类别(6类) 中心点x 中心点y 宽:w 高:H,同时生成训练问价和验证文件
def save_label_bbx(m_list, train_point, train=True):
    if train:
        train_file = open('train.txt', 'w')
        valid_file = open('valid.txt', 'w')
        num_img = 0
    else:
        test_file = open('test.txt', 'w')
    for m_path in tqdm.tqdm(m_list):
        m=loadmat(m_path)
        num_boxes = m['slots'].shape[0]
        if num_boxes != 0:  # 排除没有parking slot的图片
            m_name = m_path.split('/')[-1]
            file_name = m_name.split('.')[0] + '.txt'
            img_name = m_name.split('.')[0] + '.jpg'
            img_path = os.path.join(test_data_path,img_name)
            if train:
                save_path = os.path.join('data/custom/train/images', img_name)
                file = open(os.path.join(train_label_path, file_name), 'w')
                # 保存要训练集和验证集中图片的路径至train.txt和valid.txt
                if num_img < train_point:
                    train_file.write(save_path)
                    train_file.write('\n')
                else:
                    valid_file.write(save_path)
                    valid_file.write('\n')
                #shutil.copy(img_path,imgs_path)
                num_img +=1
                # print('-------------------------{} images to train and valid------------------------'.format(num_img))
            else:
                save_path = os.path.join('../data/custom/test/images', img_name)
                file = open(os.path.join(test_label_path, file_name), 'w')
                test_file.write(save_path)
                test_file.write('\n')
                # shutil.copy(img_path, test_imgs_path)
            # 保存生成的标签
            for i in range(num_boxes):
                label = m['slots'][i,2]-1  # 标签
                point_1 = m['marks'][m['slots'][i, 0] - 1, :]
                point_2 = m['marks'][m['slots'][i, 1] - 1, :]
                # 确定标签类型,一共6类,分为直角左上顶点 0,锐角左上顶点1,钝角左上顶点2,直角左下顶点3,锐角左下顶点4,钝角左下顶点5
                if point_2[0] > point_1[0]:
                    if point_2[1] > point_1[1]:
                        label=label
                    else:
                        label = label+3
                else:
                    if point_1[1] > point_2[1]:
                        label=label
                    else:
                        label= label +3
                w = abs(point_2[0] - point_1[0])+48
                h = abs(point_2[1] - point_1[1])+44
                l_x = point_1[0] + (point_2[0] - point_1[0])/2
                l_y = point_1[1] + (point_2[1] - point_1[1])/2
                box = list((label, l_x/600, l_y/600, w/600, h/600))  # 归一化后存入列表里
                for j in box:  # 如果一张图片中有多个parking slots, 则每一个parking slot为1行
                    file.write(str(j))
                    file.write(' ')
                file.write('\n')
        file.close()
    if train:
        train_file.close()
        valid_file.close()
    else:
        test_file.close()
# 保存生成的标签-v3.0:以车位的两个顶点的中点,宽和高为一组(即一个bounding box),分别为:类别(4类) 中心点x 中心点y 宽:w 高:H,同时生成训练文件和验证文件
def save_label_bbx_4(m_list, train_point, train=True):
    if train:
        train_file = open('train.txt', 'w')
        valid_file = open('valid.txt', 'w')
        num_img = 0
    else:
        test_file = open('test.txt', 'w')
    for m_path in tqdm.tqdm(m_list):
        m=loadmat(m_path)
        num_boxes = m['slots'].shape[0]
        num_points = m['marks'].shape[0]
        m_name = m_path.split('/')[-1]
        file_name = m_name.split('.')[0] + '.txt'
        img_name = m_name.split('.')[0] + '.jpg'
        if num_points != 0:
            if train:
                save_path = os.path.join('data/custom/train/images', img_name)
                file = open(os.path.join(train_label_path, file_name), 'w')
                # 保存要训练集和验证集中图片的路径至train.txt和valid.txt
                if num_img < train_point:
                    train_file.write(save_path)
                    train_file.write('\n')
                else:
                    valid_file.write(save_path)
                    valid_file.write('\n')
                #shutil.copy(img_path,imgs_path)
                num_img +=1
                # print('-------------------------{} images to train and valid------------------------'.format(num_img))
            else:
                save_path = os.path.join('../data/custom/test/images', img_name)
                file = open(os.path.join(test_label_path, file_name), 'w')
                test_file.write(save_path)
                test_file.write('\n')
            for i in range(num_points):
                point = m['marks'][i, :]
                box = list((3, point[0] / 600, point[1] / 600, 40 / 600, 60 / 600))  # 归一化后存入列表里
                for j in box:  # 如果一张图片中有多个parking slots, 则每一个parking slot为1行
                    file.write(str(j))
                    file.write(' ')
                file.write('\n')

        if num_boxes != 0:  # 排除没有parking slot的图片
            # 保存生成的标签
            for i in range(num_boxes):
                boxs = []
                label = m['slots'][i,2]-1  # 标签
                point_1 = m['marks'][m['slots'][i, 0] - 1, :]
                point_2 = m['marks'][m['slots'][i, 1] - 1, :]
                w = abs(point_2[0] - point_1[0])+48
                h = abs(point_2[1] - point_1[1])+44
                l_x = point_1[0] + (point_2[0] - point_1[0])/2
                l_y = point_1[1] + (point_2[1] - point_1[1])/2
                box_1 = list((label, l_x/600, l_y/600, w/600, h/600))  # 归一化后存入列表里
                for j in box_1:  # 如果一张图片中有多个parking slots, 则每一个parking slot为1行
                    file.write(str(j))
                    file.write(' ')
                file.write('\n')
        file.close()
    if train:
        train_file.close()
        valid_file.close()
    else:
        test_file.close()
# 生成测试集的parking slot标签
# 生成测试集的parking slot标签
def save_test_points(m_list):
    for m_path in tqdm.tqdm(m_list):
        m=loadmat(m_path)
        num_boxes = m['slots'].shape[0]
        if num_boxes != 0:  # 排除没有parking slot的图片
            m_name = m_path.split('/')[-1]
            file_name = m_name.split('.')[0] + '.txt'
            file = open(os.path.join(test_label_path, file_name), 'w')
            for i in range(num_boxes):
                point_1 = m['marks'][m['slots'][i, 0] - 1, :]
                point_2 = m['marks'][m['slots'][i, 1] - 1, :]
                # 保证具有较小y值的点在前面,便于和预测值做对比
                if point_2[1] > point_1[1]:
                    points = np.concatenate((point_1, point_2), axis=0)
                else:
                    points = np.concatenate((point_2, point_1), axis=0)
                label = m['slots'][i, 2] - 1  # 标签
                file.write(str(label))
                file.write(' ')
                for j in points:  # 如果一张图片中有多个parking slots, 则每一个parking slot为1行
                    file.write(str(j))
                    file.write(' ')
                file.write('\n')
            file.close()
# 统计平行车位的宽高比
def compute_average_width(m_list):
    width = []
    for m_path in tqdm.tqdm(m_list):
        m=loadmat(m_path)
        num_boxes = m['slots'].shape[0]
        if num_boxes != 0:  # 排除没有parking slot的图片
            for i in range(num_boxes):
                point1 = m['marks'][m['slots'][i, 0] - 1, :]
                point2 = m['marks'][m['slots'][i, 1] - 1, :]
                # 计算两点之间的距离
                p1_p2 = point2 - point1
                p1_p2_norm = np.sqrt(p1_p2[0] ** 2 + p1_p2[1] ** 2)
                if p1_p2_norm > 190:
                    width.append(p1_p2_norm)
    return width

# 一些数据的可视化分析
#训练数据集
train_angles, num_emperty = show_data(test_m_list)
# train_count_angles = collections.Counter(train_angles)
# print('In the train dataset:')
# for angle in train_count_angles:
#     print('angle={}:{}'.format(angle, train_count_angles[angle]))
num_train = len(train_imgs_list)-num_emperty
train_point = int(num_train*0.95)
# save_label_bbx_4(train_m_list, train_point,train=True)
# save_label_bbx(train_m_list, train_point,train=True)
# 统计测试集中各个角度出现的次数
# test_angles, _ = show_data(test_m_list)
# test_count_angles = collections.Counter(test_angles)
# print('In the train dataset')
# for angle in test_count_angles:
#     print('angle={}:{}'.format(angle, test_count_angles[angle]))
# save_label_bbx(test_m_list, 0,train=False)
# save_label_bbx_4(test_m_list,train_point,train=False)
# 生成测试集需要的点的坐标
# save_test_points(test_m_list)
# 生成ground_truth
# save_ground_truth(test_m_list)
# save_gt_paired_marking_points(test_m_list)
save_gt_parking_slot(test_m_list)
#统计宽度
# width = np.array(compute_average_width(train_m_list))
# print(width.mean())


