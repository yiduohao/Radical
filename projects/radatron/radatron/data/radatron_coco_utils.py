import datetime
import os
import json
from fvcore.common.file_io import PathManager


def convert_to_coco_dict(dataset_dicts, class_name_list):
    
    coco_images = []
    coco_annotations = []
    coco_categories = [{"id": index, "name": value, "supercategory": "empty"} for
                  index, value in enumerate(class_name_list)]

    for image_dict in dataset_dicts:
        coco_image = {
            "id": int(image_dict["image_id"]),
            "width": image_dict["width"],
            "height": image_dict["height"],
            "file_name": image_dict["file_name"],
        }
        coco_images.append(coco_image)

        for annotation in image_dict["annotations"]:
            coco_annotation = {}

            # COCO requirement: XYWH box format
            bbox = annotation["bbox"]
            bbox_mode = annotation["bbox_mode"]
            area = bbox[2]*bbox[3]
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            coco_annotation["area"] = float(area)
            coco_annotation["iscrowd"] = 0
            coco_annotation["category_id"] = annotation["category_id"]

            coco_annotations.append(coco_annotation)

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }

    coco_dict = {
        "info": info,
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
        "licenses": None,
    }
    return coco_dict


def convert_to_coco_json(output_file, dataset_dicts, class_name_list):
    coco_dict = convert_to_coco_dict(dataset_dicts, class_name_list)

    PathManager.mkdirs(os.path.dirname(output_file))
    with PathManager.open(output_file, "w") as f:
      json.dump(coco_dict, f)
 