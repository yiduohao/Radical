python radical/main_radical.py \
    --use_radical --single_gpu \
    -b 32 --lr 0.03 --cos --moco-k 4096 --moco-dim 640 --workers 8 \
    --in_batch_loss --intra_weight 1 --symmetric_loss --symmetric_loss_version 3 \
    --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 --gpu 0 \
    --output_dir /output \
    --source_folder CLIP_Left_resize \
    --rw_binary 1.0 --rw_phase 0.0 \
    --config ./configs/radical_config.yaml OUTPUT_DIR "/output" TRAIN_SAMPLE "True" TRAIN_SAMPLE_PATH "/all_file_path.json" \
    --h_flip --crop 

# # for multi-gpu training
# python radical/main_radical.py \
#     --ssl_radatron \
#     -b 256 --lr 0.03 --cos --moco-k 4096 --moco-dim 640 \
#     --in_batch_loss --intra_weight 1 --symmetric_loss --symmetric_loss_version 3 \  
#     --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
#     --output_dir /output \
#     --use_wandb --wandb_name test \
#     --source_folder CLIP_Left_resize \
#     --config ./configs/radical_config.yaml OUTPUT_DIR "/output"