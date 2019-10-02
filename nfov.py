# Copyright 2017 Nitish Mutha (nitishmutha.com)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import pi
import numpy as np
from tqdm import tqdm
import imageio as im

class NFOV():
    def __init__(self, height=400, width=800):
        self.FOV = [0.45, 0.45]
        self.PI = pi
        self.PI_2 = pi * 0.5
        self.PI2 = pi * 2.0
        self.height = height
        self.width = width
        self.screen_points = self._get_screen_img()

    def _get_coord_rad(self, isCenterPt, center_point=None):
        # 返回center_point分别乘以pi和0.5倍的pi
        return (center_point * 2 - 1) * np.array([self.PI, self.PI_2]) \
            if isCenterPt \
            else \
            (self.screen_points * 2 - 1) * np.array([self.PI, self.PI_2]) * (
                np.ones(self.screen_points.shape) * self.FOV)

    def _get_screen_img(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width), np.linspace(0, 1, self.height)) # 生成宽 X 高的一个矩阵
        return np.array([xx.ravel(), yy.ravel()]).T # 将矩阵变为一维数组后转置,前者是所有的x轴坐标，后者是所有的y轴坐标

    def _calcSphericaltoGnomonic(self, convertedScreenCoord):
        x = convertedScreenCoord.T[0]  #  x为convertedScreenCoord转置后的第一行
        y = convertedScreenCoord.T[1]  #  y为convertedScreenCoord转置后的第二行

        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(cos_c * np.sin(self.cp[1]) + (y * sin_c * np.cos(self.cp[1])) / rou)
        lon = self.cp[0] + np.arctan2(x * sin_c, rou * np.cos(self.cp[1]) * cos_c - y * np.sin(self.cp[1]) * sin_c)

        lat = (lat / self.PI_2 + 1.) * 0.5
        lon = (lon / self.PI + 1.) * 0.5

        return np.array([lon, lat]).T

    def _bilinear_interpolation(self, screen_coord):
        uf = np.mod(screen_coord.T[0],1) * self.frame_width  # long - width
        vf = np.mod(screen_coord.T[1],1) * self.frame_height  # lat - height

        x0 = np.floor(uf).astype(int)  # coord of pixel to bottom left
        y0 = np.floor(vf).astype(int)
        _x2 = np.add(x0, np.ones(uf.shape).astype(int))  # coords of pixel to top right
        y2 = np.add(y0, np.ones(vf.shape).astype(int))

        x2 = np.mod(_x2, self.frame_width)
        y2 = np.minimum(y2, self.frame_height - 1)
        base_y0 = np.multiply(y0, self.frame_width)
        base_y2 = np.multiply(y2, self.frame_width)

        A_idx = np.add(base_y0, x0)
        B_idx = np.add(base_y2, x0)
        C_idx = np.add(base_y0, x2)
        D_idx = np.add(base_y2, x2)

        flat_img = np.reshape(self.frame, [-1, self.frame_channel])

        A = np.take(flat_img, A_idx, axis=0)
        B = np.take(flat_img, B_idx, axis=0)
        C = np.take(flat_img, C_idx, axis=0)
        D = np.take(flat_img, D_idx, axis=0)

        wa = np.multiply(_x2 - uf, y2 - vf)
        wb = np.multiply(_x2 - uf, vf - y0)
        wc = np.multiply(uf - x0, y2 - vf)
        wd = np.multiply(uf - x0, vf - y0)

        # interpolate
        AA = np.multiply(A, np.array([wa, wa, wa]).T)
        BB = np.multiply(B, np.array([wb, wb, wb]).T)
        CC = np.multiply(C, np.array([wc, wc, wc]).T)
        DD = np.multiply(D, np.array([wd, wd, wd]).T)
        nfov = np.reshape(np.round(AA + BB + CC + DD).astype(np.uint8), [self.height, self.width, 3])
        # import matplotlib.pyplot as plt
        # plt.imshow(nfov)
        # plt.show()
        # im.imwrite('pic.jpg',nfov)
        return nfov

    def toNFOV(self, frame, center_point):
        self.frame = frame
        self.frame_height = frame.shape[0] # 获得原图片高度
        self.frame_width = frame.shape[1] # 获得原图片宽度
        self.frame_channel = frame.shape[2] # 获得原图片通道数

        self.cp = self._get_coord_rad(center_point=center_point, isCenterPt=True)  # 计算cp，cp是啥
        convertedScreenCoord = self._get_coord_rad(isCenterPt=False)
        spericalCoord = self._calcSphericaltoGnomonic(convertedScreenCoord)
        return self._bilinear_interpolation(spericalCoord)



# if __name__ == '__main__':
#     import imageio as im
#     img = im.imread('images/104.jpg')
#     nfov = NFOV()
#     center_point = np.array([(0), (0.9)])  # camera center point (valid range [0,1])
#     nfov.toNFOV(img, center_point)

nfov = NFOV()

latitude = [i for i in np.linspace(0, 1, 3)]
def generatePicNum():
    pic_num = {}
    for lat in latitude:
        # if abs(lat - 0.5) > 0.4:
        #     pic_num[lat] = [i for i in np.linspace(0,1,4)][:-1]
        # elif 0.4 > abs(lat-0.5) > 0.3:
        #     pic_num[lat] = [i for i in np.linspace(0, 1, 6)][:-1]
        # elif 0.3 > abs(lat-0.5) > 0.1:
        #     pic_num[lat] = [i for i in np.linspace(0, 1, 8)][:-1]
        if abs(lat-0.5) == 0:
            pic_num[lat] = [i for i in np.linspace(0, 1, 6)][:-1]
        elif lat == 0 or lat == 1:
            pic_num[lat] = [0.5]
    return pic_num

pic_num = generatePicNum()

testset = []
with open('../dataset/Pano-dataset/ImageSets/Main/test.txt') as f:
    for line in f:
        testset.append(line[:-1])
print(len(testset))
testset = testset[0:100]
path = '../dataset/Pano-dataset/JPEGImages/'
save_path = '/workspace/dataset/FOV/test/type0/'

for lat, longtitude in pic_num.items():
    print(lat)
    for lon in longtitude:
        center_point = np.array([lon, lat])
        print(center_point)

for filename in tqdm(testset):
    pic_path = path + filename +'.jpg'
    e = im.imread(pic_path)
    for lat, longtitude in pic_num.items():
        for lon in longtitude:
            center_point = np.array([lon, lat])
            p = nfov.toNFOV(e, center_point)
            im.imwrite(save_path + filename + '_{0}_{1}.jpg'.format(str(lon)[0:4], str(lat)[0:4]), p)
print('done')
