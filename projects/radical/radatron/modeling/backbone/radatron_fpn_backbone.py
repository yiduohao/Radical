from detectron2.config import *
from detectron2.modeling.backbone.fpn import FPN
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import torch
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling import BACKBONE_REGISTRY, ShapeSpec
from detectron2.modeling.backbone.resnet import build_resnet_backbone
from .radatron_resnet_backbone import build_radatron_resnet_backbone
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
import torch, os
import numpy as np
from detectron2.modeling.backbone.fpn import FPN

class RadatronFPN(FPN):
    def __init__(self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum", p2c_weights_path='None'):
        super(RadatronFPN, self).__init__(bottom_up=bottom_up, in_features=in_features, out_channels=out_channels, norm=norm, top_block=top_block, fuse_type=fuse_type)
        p2c_weights = sio.loadmat(p2c_weights_path)['p2c_weights']
        self.p2c_W = {}
        self.p2c_r1 = {}
        self.p2c_r2 = {}
        self.p2c_t1 = {}
        self.p2c_t2 = {}
        self.device = torch.cuda.current_device()
        for idx, p in enumerate(self._out_features):
            self.p2c_W[p] = torch.tensor(p2c_weights[0,0][0,idx]).unsqueeze(0).repeat(256,1,1,1).unsqueeze(0).to("cuda:"+str(self.device))
            self.p2c_r1[p] = torch.Tensor.long(torch.tensor(np.int16(p2c_weights[0,1][0,idx][:,:,0]-1))).to("cuda:"+str(self.device))
            self.p2c_r2[p] = torch.Tensor.long(torch.tensor(np.int16(p2c_weights[0,1][0,idx][:,:,1]-1))).to("cuda:"+str(self.device))
            self.p2c_t1[p] = torch.Tensor.long(torch.tensor(np.int16(p2c_weights[0,2][0,idx][:,:,0]-1))).to("cuda:"+str(self.device))
            self.p2c_t2[p] = torch.Tensor.long(torch.tensor(np.int16(p2c_weights[0,2][0,idx][:,:,1]-1))).to("cuda:"+str(self.device))

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.bottom_up(x)
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        new_results = []
        ps = self._out_features
        for idx, result in enumerate(results):
            
            P1 = result[:, :, self.p2c_r1[ps[idx]], self.p2c_t1[ps[idx]]]
            P2 = result[:, :, self.p2c_r2[ps[idx]], self.p2c_t1[ps[idx]]]
            P3 = result[:, :, self.p2c_r1[ps[idx]], self.p2c_t2[ps[idx]]]
            P4 = result[:, :, self.p2c_r2[ps[idx]], self.p2c_t2[ps[idx]]]
            P = torch.stack((P1,P2,P3,P4), dim=0).to("cuda:"+str(self.device))
            P = P.permute(1,2,3,4,0)
            W = self.p2c_W[ps[idx]].repeat(P.shape[0],1,1,1,1)
            W2 = torch.sum(W, 4)
            W2[W2==0] = 1
            Q = torch.divide(torch.sum(torch.multiply(W,P), 4),W2)
            new_results.append(Q.float())
            
        ret_dict =  {f: res for f, res in zip(self._out_features, new_results)}
        return ret_dict


@BACKBONE_REGISTRY.register()
def build_radatron_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_radatron_resnet_backbone(cfg, input_shape).cuda() if cfg.DATALOADING.INPUT_STYLE=="PB" else build_resnet_backbone(cfg, input_shape).cuda()
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = RadatronFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        p2c_weights_path=os.path.join(cfg.DATAROOT,cfg.DATALOADING.P2C_WEIGHTS)
    )
    return backbone




