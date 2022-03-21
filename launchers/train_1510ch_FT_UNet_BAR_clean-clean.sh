GPU_ID="0"

# felix asus
#TRAIN_DIR_SG1ch="${HOME}/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch-nD/training-runs/00010--dsmult3-xflip-yflip-rot-clean_and_blurry-twoheads-3_128_128-low_shot_mvds-batch16-color-translation-cutout/out/"
#VAL_DIR_SG1ch="${HOME}/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch-nD/training-runs/00010--dsmult3-xflip-yflip-rot-clean_and_blurry-twoheads-3_128_128-low_shot_mvds-batch16-color-translation-cutout/out/"
#TRAIN_DIR_SG="${HOME}/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch-nD/training-runs/00017--dsmult3-xflip-yflip-rot-clean_and_blurry-twoheads-10_128_128-low_shot_mvds-batch16-color-translation-cutout/out/"
#VAL_DIR_SG="${HOME}/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch-nD/training-runs/00017--dsmult3-xflip-yflip-rot-clean_and_blurry-twoheads-10_128_128-low_shot_mvds-batch16-color-translation-cutout/out/"
TRAIN_DIR_FT="${HOME}/Scratch/nus/00_data/data_Tom/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/"
VAL_DIR_FT="${HOME}/Scratch/nus/00_data/data_Tom/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/stacks/"

# Fine tunning
python3 ./train.py \
  --env deblur_SGgenMVDS_TRIALS_UNet_ASUS_bs4_1-1D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64_sharpsharp \
  --arch UNet \
  --batch_size 4 \
  --gpu ${GPU_ID} \
  --train_ps 64 \
  --nepoch 1500 \
  --train_dir ${TRAIN_DIR_FT} \
  --val_dir ${VAL_DIR_FT} \
  --embed_dim 16 \
  --warmup \
  --train_workers 8 \
  --eval_workers 4 \
  --use_mixup_from_epoch 6 \
  --pretrain_weights ./log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_1-1D/models/model_best_PSNR_SSIM.pth \
  --resume \
  --reset_optimizer \
  --token _down \
  --data_multiplier 480 \
  --in_channel 1 \
  --out_channel 1 \
  --sharpsharp 0.25