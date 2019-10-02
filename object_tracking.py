import numpy as np
from math import pi
from tqdm import tqdm
from glob2 import glob
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
import torch
import os
import cv2 as cv
import argparse

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
        return (center_point * 2 - 1) * np.array([self.PI, self.PI_2]) \
            if isCenterPt \
            else \
            (self.screen_points * 2 - 1) * np.array([self.PI, self.PI_2]) * (
                np.ones(self.screen_points.shape) * self.FOV)

    def _get_screen_img(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width), np.linspace(0, 1, self.height))
        return np.array([xx.ravel(), yy.ravel()]).T

    def _calcSphericaltoGnomonic(self, convertedScreenCoord):
        x = convertedScreenCoord.T[0]
        y = convertedScreenCoord.T[1]

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
        return nfov

    def toNFOV(self, frame, center_point):
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]

        self.cp = self._get_coord_rad(center_point=center_point, isCenterPt=True)
        convertedScreenCoord = self._get_coord_rad(isCenterPt=False)
        spericalCoord = self._calcSphericaltoGnomonic(convertedScreenCoord)
        return spericalCoord,self._bilinear_interpolation(spericalCoord)
nfov = NFOV()

def fov2Equir(dets, center_point, equir_img):  # convert bbox coordinate in fov to bbox coordinate in equirectangular
    equir_coor = dets.copy()
    bbox = dets.copy()
    bbox = np.array([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]])
    if bbox[2] >=800:
        bbox[2] = 799
    if bbox[3] >= 400:
        bbox[3] = 399
    bbox_equir = bbox.copy()
    screen_cords,fov_frame = nfov.toNFOV(equir_img, center_point)
    y = np.mod(screen_cords.T[1], 1)
    x = np.mod(screen_cords.T[0], 1)
    # print(bbox)
    for k in range(0, 3, 2):
        pt_FOV = np.array([bbox[k + 1], bbox[k]])
        # print(pt_FOV)
        FoV_cp_ravel = np.ravel_multi_index(pt_FOV, [400, 800])
        pt_in_equir = np.array([x[FoV_cp_ravel], y[FoV_cp_ravel]])
        pix_in_equir = pt_in_equir * np.array([equir_img.shape[1], equir_img.shape[0]])
        bbox_equir[k], bbox_equir[k + 1] = pix_in_equir[0], pix_in_equir[1]
    equir_coor[0:4] = np.array(bbox_equir)

    return equir_coor






img_size = (3840, 1920)   # the image size of equirectangular image
frame_path = '/workspace/dataset/test_frames/sport' # the directory where it store the frames of the video

def get_frames(frame_path,frame_num):
    global center
    images = glob(os.path.join(frame_path, '*.png'))
    images = sorted(images,
                    key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    images = images[frame_num:]
    for img in images:
        frame = cv.imread(img)
        nfov = NFOV()
        center_point = np.array(center)  # camera center point (valid range [0,1])
        print(center_point)
        coor,fov_frame = nfov.toNFOV(frame, center_point)
        yield frame,fov_frame

def main(model,config):
    global img_size
    sot_config_file = config  # the config file of tracker
    cfg.merge_from_file(sot_config_file)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    # create model
    tracker_model = ModelBuilder()
    # load model
    tracker_model.load_state_dict(torch.load(model,
                                     map_location=lambda storage, loc: storage.cpu()))  # load the trakcer model
    tracker_model.eval().to(device)
    # build tracker
    tracker = build_tracker(tracker_model)
    print('initialization complete')

    first_frame = True
    frame_num = 0


    bbox = (1608, 708, 142, 148) # the initial bbox in equirectangular image
    fov_bbox = (362, 159, 69, 76)  # the initial bbox in fov
    center = [(bbox[0] + 0.5 * bbox[2]) / img_size[0], (bbox[1] + 0.5 * bbox[3]) / img_size[1]]  # the initial center point to get fov
    img_size = (3840, 1920)  # the image size of equirectangular image

    print('start tracking')

    for frame,fov_frame in tqdm(get_frames(frame_path,210)):  # frame is the current equirectangular image, here I start the tracking from 211th frame
            if first_frame:
                tracker.init(fov_frame, fov_bbox)
                first_frame = False
            else:
                outputs = tracker.track(fov_frame)
                bbox = list(map(int, outputs['bbox']))
                if min(bbox) < 0:
                    bbox[bbox.index(min(bbox))] = 0  # sometimes it has negative value in bbox
                equir_coor = fov2Equir(bbox,np.array(center),frame)  # convert the bbox coordinate in fov to coordinate in equirectangular
                output = cv.rectangle(fov_frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
                cv.imwrite('./output/'+str(frame_num)+'.jpg', output)
                center = [(equir_coor[0] + 0.5*(equir_coor[2]-equir_coor[0])) / img_size[0], (equir_coor[1] + 0.5 *(equir_coor[3]-equir_coor[1])) / img_size[1]]  # update the center
            frame_num += 1

if __name__ == '__main__':
    global frame_path
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',dest='model',default='model.pth',help='the model for object tacking',type=str)
    parser.add_argument('--config',dest='config',default='config.yaml',help='the config file for object tacking',type=str)
    parser.add_argument('--frame_path', dest='frame_path', help='the directory for frames', type=str)
    args = parser.parse_args()

    frame_path = args.frame_path

    main(args.model,args.config)


