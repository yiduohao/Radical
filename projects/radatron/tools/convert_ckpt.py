import torch


model = torch.load('/mnt/sens_data1/yiduo/output/1021_moco_combine_clipresize_63k_b128_q4096_intraw8_lr0x03/checkpoint_0199.pth.tar', map_location='cpu')
# model2 = torch.load('/mnt/sens_data1/yiduo/output/radatron_pretrained_backbone/0801_MoCo_CLIP_m2_78k_b256.pth.tar', map_location='cpu')
# print(model['state_dict'].keys())

new_state_dict = {}
new_state_dict['state_dict'] = {}

for key in model['state_dict'].keys():
    if 'bottom_up' in key:
        print(key)
        new_key = key.split('bottom_up.')[1]
        print(new_key)
        new_state_dict['state_dict'][new_key] = model['state_dict'][key]

torch.save(new_state_dict, '/mnt/sens_data1/yiduo/output/radatron_pretrained_backbone/1021_moco_combine_clipresize_63k_b128_q4096_intraw8_lr0x03_200e.pth.tar')

print(len(new_state_dict['state_dict'].keys()))
# print(len(model2['state_dict'].keys()))