from detectron2.engine import default_setup
from detectron2.config import *

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = CfgNode(new_allowed=True)
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(get_cfg())
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.DATALOADING.INPUT_STYLE in ["P", "P1", "PB", "P1CHIP"]:
        cfg.MODEL.PIXEL_MEAN = (0,)
        cfg.MODEL.PIXEL_STD = (1,)
    if cfg.DATALOADING.INPUT_STYLE in ["P", "P1", "P1CHIP"]:
        cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg