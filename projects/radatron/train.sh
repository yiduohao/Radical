python ./tools/train.py --config ./configs/radatron_config.yaml \
    --use_wandb --wandb_name test \
    OUTPUT_DIR "/output" \
    PRETRAINED_BACKBONE_PATH '/radatron_pretrained_backbone.pth.tar' \
    PRETRAINED_BACKBONE_PART 'bottom_up' \
    TRAIN_SAMPLE "False"