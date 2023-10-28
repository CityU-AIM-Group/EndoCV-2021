
import os
import torch
import os.path as osp
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .transform import *
from PIL import Image, ImageEnhance, ImageOps, ImageFile, ImageFilter

def calculateCompact(image):
    #image = Image.open(image).convert('L')
    edge = image.filter(ImageFilter.FIND_EDGES)
    #edge.save('/home/cyang/SFDA/edge.png')
    edge = np.asarray(edge, np.float32)
    image = np.asarray(image, np.float32)
    image = image / 255
    edge = edge / 255
    S = np.sum(image)
    C = np.sum(edge)
    #print(C, S, edge[0])
    return np.asarray(100 * S / C ** 2, np.float32)

def randomRotation(image, label):
    """
    对图像进行随机任意角度(0~360度)旋转
    :param mode 邻近插值,双线性插值,双三次B样条插值(default)
    :param image PIL的图像image
    :return: 旋转转之后的图像
    """
    random_angle = np.random.randint(1, 60)
    return image.rotate(random_angle, Image.BICUBIC), label.rotate(random_angle, Image.NEAREST)

def randomColor(image):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

def randomGaussian(image, mean=0.2, sigma=0.3):
    """
    对图像进行高斯噪声处理
    :param image:
    :return:
    """

    def gaussianNoisy(im, mean=0.2, sigma=0.3):
        """
        对图像做高斯噪音处理
        :param im: 单通道图像
        :param mean: 偏移量
        :param sigma: 标准差
        :return:
        """
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    # 将图像转化成数组
    img = np.asarray(image)
    #img.flags.writeable = True  # 将数组改为读写模式
    width, height = img.shape[:2]
    img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
    img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
    img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])
    return Image.fromarray(np.uint8(img))

# EndoScene Dataset
class PolypDataset_Aug(Dataset):
    def __init__(self, data_path='/home/cyang/EndoCV/Data/data_C1', train=True):

        super(PolypDataset_Aug, self).__init__() 
        self.data_path = data_path
        self.train = train

        if self.train:
            self.data_dir = os.path.join(data_path, 'train.lst')
        else:
            self.data_dir = os.path.join(data_path, 'test.lst')
        self.img_ids = [i_id.strip() for i_id in open(self.data_dir)]
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.data_path, "images/%s.jpg" % name)
            label_file = osp.join(self.data_path, "masks/%s_mask.jpg" % name)
            self.files.append({
                "image": img_file,
                "label": label_file,
                "name": name
            })
        #print(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]["image"]
        gt_path = self.files[index]["label"]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((256, 256), Image.BICUBIC)
        gt = Image.open(gt_path).convert('L')
        gt = gt.resize((256, 256), Image.NEAREST)
        if self.train:
            img = randomColor(img)
            img, gt = randomRotation(img, gt)
            #img = randomGaussian(img)

        img = np.asarray(img, np.float32)
        img = img / 255
        gt = np.asarray(gt, np.float32)
        gt = gt / 255

        img = img[:, :, ::-1]  # change to BGR
        img = img.transpose((2, 0, 1))

        if self.train:
            flip = np.random.choice(2) * 2 - 1
            img = img[:, :, ::flip]
            gt = gt[:, ::flip]

        data = {'image': img.copy(), 'label': gt.copy()}

        return data, gt_path

    def __len__(self):
        return len(self.files)


def get_data_aug(data_root, batch_size):
    trainset_A = PolypDataset_Aug(data_path=os.path.join(data_root, 'data_C1'), train=True)
    testset_A = PolypDataset_Aug(data_path=os.path.join(data_root, 'data_C1'), train=False)

    trainset_B = PolypDataset_Aug(data_path=os.path.join(data_root, 'data_C2'), train=True)
    testset_B = PolypDataset_Aug(data_path=os.path.join(data_root, 'data_C2'), train=False)

    trainset_C = PolypDataset_Aug(data_path=os.path.join(data_root, 'data_C3'), train=True)
    testset_C = PolypDataset_Aug(data_path=os.path.join(data_root, 'data_C3'), train=False)

    trainset_D = PolypDataset_Aug(data_path=os.path.join(data_root, 'data_C4'), train=True)
    testset_D = PolypDataset_Aug(data_path=os.path.join(data_root, 'data_C4'), train=False)

    trainset_E = PolypDataset_Aug(data_path=os.path.join(data_root, 'data_C5'), train=True)
    testset_E = PolypDataset_Aug(data_path=os.path.join(data_root, 'data_C5'), train=False)

    train_set = torch.utils.data.ConcatDataset([trainset_A, trainset_B, trainset_C, trainset_D, trainset_E])
    test_set = torch.utils.data.ConcatDataset([testset_A, testset_B, testset_C, testset_D, testset_E])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, test_loader


if __name__ == '__main__':
    Source_data = Polyp(root='/home/cyang/SFDA/data/EndoScene', data_dir='/home/cyang/SFDA/dataset/EndoScene_list/train.lst', mode='train', max_iter=15000)
    print(Source_data.__len__())
    # for i in range(15000):
    #     print(Source_data[i])
    train_loader = torch.utils.data.DataLoader(Source_data, batch_size=8, shuffle=True, num_workers=4)
    print(np.max(Source_data[0][0]['image']), np.min(Source_data[0][0]['image']))