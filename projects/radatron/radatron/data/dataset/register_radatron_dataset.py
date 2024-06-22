import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from .RadatronDataset import RadatronDataset
from ..radatron_coco_utils import convert_to_coco_json
from detectron2.data.datasets import register_coco_instances

def register_radatron(cfg):
    train_set = RadatronDataset(cfg=cfg, val=False)
    test_set = RadatronDataset(cfg=cfg, val=True)

    DatasetCatalog.register("radar_train", lambda d="train": train_set.get_dataset_dict())
    MetadataCatalog.get("radar_train").set(thing_classes=cfg.DATAPATHS.CATEGORIES)

    coco_eval_path = os.path.join(cfg.OUTPUT_DIR, "Test_coco_format.json")
    convert_to_coco_json(coco_eval_path, test_set.get_dataset_dict(), cfg.DATAPATHS.CATEGORIES)
    register_coco_instances("radar_val", {}, coco_eval_path, "/")
