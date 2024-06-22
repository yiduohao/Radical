import numpy as np
import torch
import scipy.io as sio
from detectron2.config import *
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
import torch
from .radatron_augs import transform_instance_annotations, RadatronRandomCrop, RadatronRandomRotation, RadatronRandomBrightness, RadatronRandomContrast, RadatronRandomFlip
from .radatron_aug_proc import apply_augmentations_twostream
import torchvision
import code
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import copy
import random
import glob
import scipy
import pdb
import time

APPLY_AUGS_LOW_RES = True
LOW_RES_INDICES = [0, 1, 2, 3, 11, 12, 13, 14, 46, 47, 48, 49, 50, 51, 52, 53]
DAYS = {"day1": "0305", "day2": "0308", "day3": "0309", "day4": "0312"}
N = 256
VALS_PHASE = np.linspace(-np.pi, np.pi, N+1)
THRESHOLDING = 0

def reconstruct_variable(C, vals):
    return vals[C.astype(int)]

class SSLRadatronMapper:
    def __init__(self, cfg, mode='train', clip_source_folder=None):
        self.cfg = cfg
        self.mode = mode
        self.style = cfg.DATALOADING.INPUT_STYLE
        self.preprocess = cfg.PREPROCESS
        self.dataloading = cfg.DATALOADING
        self.augs = cfg.AUGS
        self.apply_augs = cfg.APPLY_AUGS if mode=="train" else False #Only apply augmentations for training
        self.angle_range = cfg.DATALOADING.ANGLE_RANGE
        self.ssl_aug_list = cfg.SSL.AUG_LIST
        self.ssl_cfg = cfg.SSL
        self.all_files = glob.glob("/mnt/sens_data1/yiduo/jayden_temp/*/*/*.mat")
        for aug in self.ssl_aug_list:
            assert aug in ["flip", "rot", "crop", "threshold", "autothreshold", "vflip", "cutout", "actual_rot"]

        assert clip_source_folder is not None, "clip_source_folder must be specified for SSLRadatronMapper"
        self.clip_source_folder = clip_source_folder

    def __call__(self, dataset_dict):
        input_path = dataset_dict["file_name"] if self.mode=='train' else dataset_dict["file_name"][1:]
        if self.style == "PB":
            raise NotImplementedError("Radical not supports PB style input.")
        elif self.style == "R":

            C, vals, P, X2, i_path = self.input_loader(input_path = input_path, clip_source_folder = self.clip_source_folder)

            X = {
                "C" : C,
                "vals" : vals,
                "P" : P
            }

            CLIP_feat = {"feat" : X2}
        else:
            raise NotImplementedError("Radical only supports R style input for now.")

        #Apply augmentations
        if self.apply_augs:
            raise NotImplementedError("Augementation not supported for Radical yet.")

        else:
            if self.style in ["PB", "PBD"]:
                raise NotImplementedError("Radical only supports R style input for now.")
            
            elif self.style == "R":

                X["C"] = torch.from_numpy(X["C"]) #C,H,W
                X["vals"] = torch.from_numpy(X["vals"]).to(torch.float32) #C,H,W
                X["P"] = torch.from_numpy(X["P"]) #C,H,W

            else:
                raise NotImplementedError("Radical only supports R style input for now.")

        return {
        # create the format that the model expects
        "image": X,
        "clip": CLIP_feat,
        }

    def random_weights(self, X):
        data = X["image_raw"].astype(np.complex64)
        data = data.squeeze(2)
        data = data[40:488, :, :] #2-24.35

        return data


    def extract_name(self, filepath):
        name = "".join([f'{x}_' for x in filepath.split("_")[-4:]])[:-1]
        return name

    def find_file(self, filepath):
        name = self.extract_name(filepath)
        name_split = name.split("_")
        # print(name_split)
        date, exp, number, framedotmat = name_split
        new_filepath = f"/mnt/sens_data1/yiduo/jayden_temp/{date}/{date}_{exp}_RA_randWeight_compress/RA_randWeight_{exp}_file{int(number)}_{framedotmat}"

        return new_filepath

    def load_compressed(self, filepath):
        # Load the MATLAB file
        mat_contents = scipy.io.loadmat(filepath)

        C = mat_contents['mag_quantized']
        vals = mat_contents['mag_residual'][0]
        P = mat_contents['phase_quantized']

        return C, vals, P

    def input_loader(self, input_path, clip_source_folder):

        # david: Fixed bugs on input path
        if input_path[0:3] == 'mnt':
            input_path = '/' + input_path  # add a '/' in front of the path string

        if self.style == "P": #9tx pwr input
            raise NotImplementedError("Radical only supports R style input for now.")
        elif self.style == "P1": #1tx pwr input
            raise NotImplementedError("Radical only supports R style input for now.")
        elif self.style == "P1chip": #1tx pwr input
            raise NotImplementedError("Radical only supports R style input for now.")
        elif self.style == "PB": #both 9tx pwr + 1tx pwr
            raise NotImplementedError("Radical only supports R style input for now.")
        elif self.style == "R":
            print("Using R Input Style")
            new_input_path = self.find_file(input_path)
            clip_left_filename = input_path.replace("heatmap_HighRes", clip_source_folder).replace(".mat", ".npy").replace("radar", "clip")
            self.norm_factor = {
                "1" : self.dataloading.NORM_HR if self.dataloading.COMPENSATION else self.dataloading.NORM_HRNF,
                "2" : self.dataloading.NORM_LR
            }


            C, vals, P = self.load_compressed(new_input_path)

            if not os.path.exists(clip_left_filename):
                print("clip_left_filename not exist: ", clip_left_filename)

            try:
                CLIP_feat = np.load(clip_left_filename)
            except:
                print("Error Loading File: ", clip_left_filename)

            return C, vals, P, CLIP_feat, [new_input_path, clip_left_filename]

        heatmap = sio.loadmat(input_path)
        X = heatmap["heatmap"]
        return X, input_path


    def change_style(self, input):

        input = input[:,:,np.newaxis] #H,W,C
        return input

    def preprocess_input(self, X, normalizer=None, stream=None):
        X = np.asarray(X, dtype=np.float32) #H,W,C R,P,C
        X = X[40:488, :, :] #2-24.35
        norm_factor = self.norm_factor if stream is None else self.norm_factor[f"{stream}"]
        if self.preprocess.NORMALIZE:
            if self.style != "R":
                X = X/(np.max(X) if np.max(X) > norm_factor else norm_factor)
            else:
                X = X/(np.max(X) if np.random.rand() > 0.5 else norm_factor)
        if self.preprocess.TAKE_LOG:
            X = np.log10(1+X)

        return X

