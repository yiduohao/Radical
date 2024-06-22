import scipy.io as sio
import torch.utils.data as data_utils
import glob
import os
from detectron2.structures import BoxMode
from tqdm import tqdm
import random
import json
import code
import pdb

class RadatronDataset(data_utils.Dataset):

    def __init__(self, cfg, val):
        """
        """
        super().__init__()
        self.val = val
        self.dataroot = cfg.DATAROOT
        self.datapath = cfg.DATAPATHS.TEST if self.val else cfg.DATAPATHS.TRAIN
        self.style = cfg.DATALOADING.INPUT_STYLE
        self.Xbins = cfg.DATALOADING.X_BINS
        self.Ybins = cfg.DATALOADING.Y_BINS 
        self.X_min = cfg.DATALOADING.X_RANGE[0]
        self.X_max = cfg.DATALOADING.X_RANGE[1]
        self.Y_min = cfg.DATALOADING.Y_RANGE[0]
        self.Y_max = cfg.DATALOADING.Y_RANGE[1]

        self.names = self.load_filenames()

        # if not self.val:
        #     print(len(self.names))
        #     self.names = random.sample(self.names, int(13765*0.01))
        #     code.interact(local=locals())
        #     my_names = {}
        #     my_names['names'] = self.names
        #     print('length for 10 percent!!!!!!!!!!!!!!!')
        #     print(len(self.names))
        #     with open('/mnt/sens_data1/yiduo/data/radatron_dataset_16k_exp/random_sample_1_11.json', 'w') as fp:
        #         json.dump(my_names, fp)
        #     code.interact(local=locals())
        if not self.val:
            if cfg.TRAIN_SAMPLE == True:
                print('using train dataset sampling')
                print(cfg.TRAIN_SAMPLE_PATH)
                with open(cfg.TRAIN_SAMPLE_PATH, 'r') as fp:
                    my_names = json.load(fp)
                    self.names = my_names['names']
                    print(len(self.names))

        self.total_len = len(self.names)
        

    def get_dataset_dict(self):
        """ returns the input, groundtruth and the filename of the raw input data 
        for a given index. """
        
        dataset_dicts = []

        for idx in tqdm(range(self.total_len)):
            record={}
            filename = self.get_heatmap_path(self.names[idx])
            
            Y = self.load_groundtruth(idx)
            objs = self.process_gt(Y)
            
            if len(objs)==0:
                continue

            record = {
                "file_name" : filename,
                "image_id" : idx,
                "height" : self.Ybins,
                "width" : self.Xbins,
                "annotations" : objs
            }

            dataset_dicts.append(record)

        return dataset_dicts

    def process_gt(self, Y):
        objs=[]
        for rectangle in Y:
            angle = -rectangle[4]
            angle = (angle + 180) % 360 -180 if self.val else (angle + 90) % 180 - 90
            
            center_x = (rectangle[0]-self.X_min)*(self.Xbins-1)/(self.X_max-self.X_min)
            center_y = (rectangle[1]-self.Y_min)*(self.Ybins-1)/(self.Y_max-self.Y_min)
            length_bb = rectangle[2]*(self.Xbins-1)/(self.X_max-self.X_min)
            width_bb = rectangle[3]*(self.Ybins-1)/(self.Y_max-self.Y_min)

            obj = {
                "bbox" : [center_x, center_y, width_bb, length_bb, angle],#[Cx, Cy, W, H, A]
                "bbox_mode" : BoxMode.XYWHA_ABS,
                "category_id" : 0,
            }

            objs.append(obj)

        return objs

    def __len__(self):
        """Return the total number of heatmaps in the dataset."""

        return len(self.names)

    def load_groundtruth(self, idx):
        """ loads the groundtruth given the index of the filename """

        y_path = self.names[idx]
        Y = sio.loadmat(y_path)['bb_clwa']
        
        return Y


    def get_heatmap_path(self, x_path):
        """ returns the path of the input given the path of the groundtruth """
        
        y_path = x_path.replace("GT", "heatmap").replace("bb", "radar")
        
        return y_path

    def load_filenames(self):
        """
        finds all heatmap filespaths inside a directory.
        """

        filenames = []
        for path in self.datapath:
            all_names = glob.glob(os.path.join(self.dataroot, path, "GT", "*.mat"))

            filenames += all_names
            

        return filenames

    





