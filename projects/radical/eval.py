import logging
from detectron2.data import build_detection_test_loader
from detectron2.engine import default_argument_parser, launch
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import inference_on_dataset
import os
from radatron.data import RadatronMapper
from radatron.evaluation import RadatronEvaluator
from radatron.config.radatron_setup import setup
from radatron.data.dataset.register_radatron_dataset import register_radatron

from train import seed_everything

logger = logging.getLogger("detectron2")

args = default_argument_parser().parse_args()
cfg = setup(args)

def main(args):
    cfg = setup(args)
    seed_everything(cfg.SEED)
    
    cfg.defrost()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST =0.05
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST=0.05
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    coco_eval_path = os.path.join(cfg.OUTPUT_DIR, "Test_coco_format.json")
    cfg.DATASETS.TEST = ("Test_coco",)

    register_radatron(cfg)
     
    evaluator = RadatronEvaluator("Test_coco", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "Test_coco", mapper=RadatronMapper(cfg, mode='eval')) 

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.train(False)

    outputs = inference_on_dataset(model, val_loader, evaluator)   
    

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    
    



  
  