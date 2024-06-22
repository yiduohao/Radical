import scipy.io as sio
import torch.utils.data as data_utils
import glob
import os
from detectron2.structures import BoxMode
from tqdm import tqdm
import code
from radatron.data import SSLRadatronMapper
import json

class RadatronDataset(data_utils.Dataset):

    def __init__(self, cfg, val, ssl=False, args=None):
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
        self.Rbins = cfg.DATALOADING.R_BINS
        self.Phibins = cfg.DATALOADING.PHI_BINS

        self.ssl = cfg.DATALOADING.SSL
        if self.ssl:


            if cfg.TRAIN_SAMPLE == True:
                print('using train dataset sampling')
                print(cfg.TRAIN_SAMPLE_PATH)
                with open(cfg.TRAIN_SAMPLE_PATH, 'r') as fp:
                    my_names = json.load(fp)
                    self.names = my_names['names']
                    print(len(self.names))
            else:
                self.names = self.load_ssl_filenames()


            if "/mnt/sens_data1/yiduo/data/radatron_dataset_152k/0308_exp3/heatmap_HighRes/radar_0308_exp3_0000_frm6.mat" in self.names:
                self.names.remove("/mnt/sens_data1/yiduo/data/radatron_dataset_152k/0308_exp3/heatmap_HighRes/radar_0308_exp3_0000_frm6.mat")
            if "/mnt/sens_data1/yiduo/data/radatron_dataset_152k/0308_exp3/heatmap_HighRes/radar_0308_exp3_0000_frm7.mat" in self.names:
                self.names.remove("/mnt/sens_data1/yiduo/data/radatron_dataset_152k/0308_exp3/heatmap_HighRes/radar_0308_exp3_0000_frm7.mat")
            if "/mnt/sens_data1/yiduo/data/radatron_dataset_152k/0308_exp3/heatmap_HighRes/radar_0308_exp3_0000_frm12.mat" in self.names:
                self.names.remove("/mnt/sens_data1/yiduo/data/radatron_dataset_152k/0308_exp3/heatmap_HighRes/radar_0308_exp3_0000_frm12.mat")
            if "/mnt/sens_data1/yiduo/data/radatron_dataset_152k/0308_exp3/heatmap_HighRes/radar_0308_exp3_0000_frm14.mat" in self.names:
                self.names.remove("/mnt/sens_data1/yiduo/data/radatron_dataset_152k/0308_exp3/heatmap_HighRes/radar_0308_exp3_0000_frm14.mat")
            if "/mnt/sens_data1/yiduo/data/radatron_dataset_152k/0312_exp1/heatmap_HighRes/radar_0312_exp1_0029_frm329.mat" in self.names:
                self.names.remove("/mnt/sens_data1/yiduo/data/radatron_dataset_152k/0312_exp1/heatmap_HighRes/radar_0312_exp1_0029_frm329.mat")
        else:
            self.names = self.load_filenames()
        
        self.total_len = len(self.names)
        assert args is not None
        if self.ssl:
            self.dataset_dicts = self.get_dataset_dict_ssl(clip_source_folder = args.source_folder)
            self.mapper = SSLRadatronMapper(cfg, mode='train', clip_source_folder = args.source_folder)
        
        self.called = 0

    
    def find_file(self, filepath):
        name = self.extract_name(filepath)
        name_split = name.split("_")
        # print(name_split)
        date, exp, number, framedotmat = name_split
        new_filepath = f"/mnt/sens_data1/yiduo/jayden_temp/{date}/{date}_{exp}_RA_randWeight_compress/RA_randWeight_{exp}_file{int(number)}_{framedotmat}"

        return new_filepath


    def extract_name(self, filepath):
        name = "".join([f'{x}_' for x in filepath.split("_")[-4:]])[:-1]
        return name


    def get_dataset_dict_ssl(self, clip_source_folder):
        """ returns the input, groundtruth and the filename of the raw input data 
        for a given index. """
        
        dataset_dicts = []

        for idx in tqdm(range(self.total_len)):
            record={}
            filename = self.names[idx]

            heatmap_HighRes_filename = filename
            clip_left_filename = filename.replace("heatmap_HighRes", clip_source_folder).replace(".mat", ".npy").replace("radar", "clip")
            
            rand_weights_filename = self.find_file(filename)
            if not os.path.exists(rand_weights_filename):
                print("File not found for random weights: ", filename)
                continue

            if (not os.path.exists(heatmap_HighRes_filename)) or (not os.path.exists(clip_left_filename)):
                print("File not found for all three types: ", filename)
                continue


            record = {
                "file_name" : filename,
                "image_id" : idx,
            }

            dataset_dicts.append(record)

        return dataset_dicts
    


    def __len__(self):
        """Return the total number of heatmaps in the dataset."""

        return len(self.names)

    def load_filenames(self):
        """
        finds all heatmap filespaths inside a directory.
        """

        filenames = []
        for path in self.datapath:
            all_names = glob.glob(os.path.join(self.dataroot, path, "GT", "*.mat"))

            filenames += all_names
            

        return filenames
    
    
    def load_ssl_filenames(self):
        """
        finds all image filespaths inside a directory.
        """

        filenames = []
        for path in self.datapath:
            all_names = glob.glob(os.path.join(self.dataroot, path, "heatmap_HighRes", "*.mat"))

            filenames += all_names
            

        for idx, filename in enumerate(filenames):
            new_filename = filename.split("/")[-1]
            frame = new_filename.split("_")[-1][3:-4]
            filled_frame = frame.zfill(4)
            filenames[idx] = filename.replace("frm" + frame, "frm" + filled_frame)


        filenames = sorted(filenames)

        for idx, filename in enumerate(filenames):
            new_filename = filename.split("/")[-1]
            frame = new_filename.split("_")[-1][3:-4]
            filled_frame = frame.lstrip('0')
            filenames[idx] = filename.replace("frm" + frame, "frm" + filled_frame)


        return filenames

    def __getitem__(self, idx):
        self.called += 1
        dict = self.dataset_dicts[idx]
        data_dict = self.mapper.__call__(dict)
        return data_dict, idx
    





