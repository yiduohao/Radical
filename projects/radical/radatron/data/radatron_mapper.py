import numpy as np
import torch
import scipy.io as sio
from detectron2.config import *
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
import torch
from .radatron_augs import transform_instance_annotations, RadatronRandomCrop, RadatronRandomRotation, RadatronRandomBrightness, RadatronRandomContrast, RadatronRandomFlip
from .radatron_aug_proc import apply_augmentations_twostream

class RadatronMapper:
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.mode = mode
        self.style = cfg.DATALOADING.INPUT_STYLE
        self.preprocess = cfg.PREPROCESS
        self.dataloading = cfg.DATALOADING
        self.augs = cfg.AUGS
        self.apply_augs = True if mode=="train" else False #Only apply augmentations for training
        self.angle_range = cfg.DATALOADING.ANGLE_RANGE

    def __call__(self, dataset_dict):
        input_path = dataset_dict["file_name"] if self.mode=='train' else dataset_dict["file_name"][1:]
        if self.style == "PB":
            #First, load the files. For "PB" (Radatron, high-res + low-res), load both the high-res and low-res matfiles
            X1, X2, i_path = self.input_loader(input_path = input_path)
            
            #Transform the loaded files into the desired dimension/format.
            X1 = self.change_style(input=X1)
            X2 = self.change_style(input=X2)
            
            #Apply preprocessing
            X1 = self.preprocess_input(X1, stream=1)
            X2 = self.preprocess_input(X2, stream=2)
            X = {
                "image1" : X1,
                "image2" : X2
            }

        else:
            #If not using the two-stream Radatron, load only a single file.
            X, i_path = self.input_loader(input_path = input_path)  #W,H,C or R,P,C
            
            #Transform the loaded file into the desired dimension/format.
            X = self.change_style(input = X) #H,W,C or R,P,C
            
            #Apply preprocessing
            X = self.preprocess_input(X)#, normalizer=normalizer) #H,W,C or R,P,C
        
        #Apply augmentations
        if self.apply_augs:
            X, annos = self.augment(X, dataset_dict)
            
        else:
            if self.style in ["PB", "PBD"]:
                X["image1"] = torch.from_numpy(X["image1"]).permute(2,0,1).float() #C,H,W
                X["image2"] = torch.from_numpy(X["image2"]).permute(2,0,1).float() #C,H,W
            else:
                X = torch.from_numpy(X).permute(2,0,1).float()

            annos = [
                annotation
                for annotation in dataset_dict.pop("annotations")
            ]  
            

        return {
        # create the format that the model expects
        "image": X, 
        "instances": utils.annotations_to_instances_rotated(annos, [dataset_dict["height"], dataset_dict["width"]]),
        "height": dataset_dict["height"],
        "width": dataset_dict["width"],
        "image_id": dataset_dict["image_id"],
        "annotations": annos,
        "file_name": i_path
        }


    def input_loader(self, input_path):
        
        # david: Fixed bugs on input path
        if input_path[0:3] == 'mnt':
            input_path = '/' + input_path  # add a '/' in front of the path string

        if self.style == "P": #9tx pwr input
            input_path = input_path.replace(f"heatmap", (f"heatmap_HighRes" if self.dataloading.COMPENSATION else f"heatmap_NoFix"))
            self.norm_factor =  self.dataloading.NORM_HR if self.dataloading.COMPENSATION else self.dataloading.NORM_HRNF
        elif self.style == "P1": #1tx pwr input
            input_path = input_path.replace(f"heatmap", f"heatmap_LowRes")
            self.norm_factor = self.dataloading.NORM_LR
        elif self.style == "P1chip": #1tx pwr input
            input_path = input_path.replace(f"heatmap", f"heatmap_1chip")
            self.norm_factor = self.dataloading.NORM_1CHIP
        elif self.style == "PB": #both 9tx pwr + 1tx pwr
            input_path1 = input_path.replace(f"heatmap", (f"heatmap_HighRes" if self.dataloading.COMPENSATION else f"heatmap_NoFix"))  #9tx 
            input_path2 = input_path.replace(f"heatmap", f"heatmap_LowRes") #1tx
            self.norm_factor = {
                "1" : self.dataloading.NORM_HR if self.dataloading.COMPENSATION else self.dataloading.NORM_HRNF,
                "2" : self.dataloading.NORM_LR
            }

            # david: Fixed bugs on input path
            if input_path1[0:3] == 'mnt':
                input_path2 = '/' + input_path2  # add a '/' in front of the path string
            if input_path2[0:3] == 'mnt':
                input_path1 = '/' + input_path1  # add a '/' in front of the path string

            heatmap9tx = sio.loadmat(input_path1)
            heatmap1tx = sio.loadmat(input_path2)

            X1 = heatmap9tx["heatmap"]
            X2 = heatmap1tx["heatmap"]            
            return X1, X2, [input_path1, input_path2]

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
            X = X/(np.max(X) if np.max(X) > norm_factor else norm_factor)
        if self.preprocess.TAKE_LOG:
            X = np.log10(1+X)
            
        return X
        

    def augment(self, X, dataset_dict):
        input_shape = X["image1"].shape if self.style in ["PB"] else X.shape
        transform_list = self.get_augs(cfg=self.augs, input_shape=input_shape, output_shape=[dataset_dict["height"], dataset_dict["width"]])

        X, transforms = apply_augmentations_twostream(transform_list, X) if self.style in ["PB"] else T.apply_transform_gens(transform_list, X)

        if self.style in ["PB"]:
            X1 = X["image1"]
            X2 = X["image2"]
            if len(X1.shape)==2:
                X1 = X1[:,:,np.newaxis]
            if len(X2.shape)==2:
                X2 = X2[:,:,np.newaxis]
            X["image1"] = torch.from_numpy(X1).permute(2,0,1).float() #C,H,W
            X["image2"] = torch.from_numpy(X2).permute(2,0,1).float() #C,H,W
        else:
            if len(X.shape)==2:
                X = X[:,:,np.newaxis]
            X = torch.from_numpy(X).permute(2,0,1)
    
        annos=[]
        for annotation in dataset_dict.pop("annotations"):
            annot_trans = transform_instance_annotations(annotation, transforms, [dataset_dict["height"], dataset_dict["width"]])
            if annot_trans["bbox"].shape[0]!=0:
                annot_trans["bbox"][4] = (annot_trans["bbox"][4] + 180) % 360 -180 if self.angle_range==360 else (annot_trans["bbox"][4] + 90) % 180 - 90
                annos.append(annot_trans)
        
        return X, annos

    def get_augs(self, cfg, input_shape, output_shape):
        rot_center = ((int(input_shape[1]/2), 0))
        gt_rot_center = ((int(self.dataloading.X_BINS/2), 0))

        transform_list = []
        if "flip" in cfg.AUG_LIST:
            transform_list.append(RadatronRandomFlip(prob=self.augs.FLIP_PROB, horizontal=True, vertical=False, gt_grid_width=output_shape[1])) 
        if "rot" in cfg.AUG_LIST:
            transform_list.append(T.RandomApply(RadatronRandomRotation(self.augs.ROT_AUG.RANGE,sample_style=self.augs.ROT_AUG.SAMPLE_STYLE, expand=self.augs.ROT_AUG.EXPAND, center=rot_center, fill_zeros=self.augs.ROT_AUG.FILL_ZEROS, gt_grid_w=output_shape[1], gt_grid_h=output_shape[0], gt_center=gt_rot_center), prob=0.8))
        if "rb" in cfg.AUG_LIST:
            transform_list.append(T.RandomApply(RadatronRandomBrightness(1-cfg.RAND_BRIGHT, 1+cfg.RAND_BRIGHT)))
        if "rc" in cfg.AUG_LIST:
            transform_list.append(T.RandomApply(RadatronRandomContrast(1-cfg.RAND_CONTRAST, 1+cfg.RAND_CONTRAST)))
        
        return transform_list


