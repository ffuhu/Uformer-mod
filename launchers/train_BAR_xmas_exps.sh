#GPU_ID="7"
GPU_ID="4"
# RUNNING NOW (211222_1725)


# UNET

## training with SGEN
#python3 ./train.py \
#  --env deblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_3-1D \
#  --arch UNet \
#  --batch_size 4 \
#  --gpu ${GPU_ID} \
#  --train_ps 64 \
#  --nepoch 150 \
#  --train_dir /home/ffuentes/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch-nD/training-runs/00010--dsmult3-xflip-yflip-rot-clean_and_blurry-twoheads-3_128_128-low_shot_mvds-batch16-color-translation-cutout/out/ \
#  --val_dir /home/ffuentes/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch-nD/training-runs/00010--dsmult3-xflip-yflip-rot-clean_and_blurry-twoheads-3_128_128-low_shot_mvds-batch16-color-translation-cutout/out/ \
#  --embed_dim 16 \
#  --warmup \
#  --train_workers 8 \
#  --eval_workers 4 \
#  --use_mixup_from_epoch 6 \
#  --in_channel 3 \
#  --out_channel 1
#
#python3 ./train.py \
#  --env deblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_3-3D \
#  --arch UNet \
#  --batch_size 4 \
#  --gpu ${GPU_ID} \
#  --train_ps 64 \
#  --nepoch 150 \
#  --train_dir /home/ffuentes/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch-nD/training-runs/00010--dsmult3-xflip-yflip-rot-clean_and_blurry-twoheads-3_128_128-low_shot_mvds-batch16-color-translation-cutout/out/ \
#  --val_dir /home/ffuentes/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch-nD/training-runs/00010--dsmult3-xflip-yflip-rot-clean_and_blurry-twoheads-3_128_128-low_shot_mvds-batch16-color-translation-cutout/out/ \
#  --embed_dim 16 \
#  --warmup \
#  --train_workers 8 \
#  --eval_workers 4 \
#  --use_mixup_from_epoch 6 \
#  --in_channel 3 \
#  --out_channel 3

## TODO: FT with manually_selected # need to implement the patch/slices cropping
## TODO: get this data ready: /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2
#python3 ./train.py \
#  --env deblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_3-1D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64 \
#  --arch UNet \
#  --batch_size 4 \
#  --gpu ${GPU_ID} \
#  --train_ps 64 \
#  --nepoch 1500 \
#  --train_dir /home/ffuentes/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/ \
#  --val_dir /home/ffuentes/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/ \
#  --embed_dim 16 \
#  --warmup \
#  --train_workers 8 \
#  --eval_workers 4 \
#  --use_mixup_from_epoch 6 \
#  --pretrain_weights /home/ffuentes/Scratch/nus/20_deblurring_uformer/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4/models/model_best.pth \
#  --resume \
#  --reset_optimizer \
#  --token _down \
#  --data_multiplier 480


# UFORMER

# training with SGEN
python3 ./train.py \
  --env deblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_3-1D \
  --arch Uformer \
  --batch_size 4 \
  --gpu ${GPU_ID} \
  --train_ps 64 \
  --nepoch 150 \
  --train_dir /home/ffuentes/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch-nD/training-runs/00010--dsmult3-xflip-yflip-rot-clean_and_blurry-twoheads-3_128_128-low_shot_mvds-batch16-color-translation-cutout/out/ \
  --val_dir /home/ffuentes/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch-nD/training-runs/00010--dsmult3-xflip-yflip-rot-clean_and_blurry-twoheads-3_128_128-low_shot_mvds-batch16-color-translation-cutout/out/ \
  --embed_dim 16 \
  --warmup \
  --train_workers 8 \
  --eval_workers 4 \
  --use_mixup_from_epoch 6 \
  --in_channel 3 \
  --out_channel 1

python3 ./train.py \
  --env deblur_SGgenMVDS_TRIALS_Uformer_BAR_bs4_3-3D \
  --arch Uformer \
  --batch_size 4 \
  --gpu ${GPU_ID} \
  --train_ps 64 \
  --nepoch 150 \
  --train_dir /home/ffuentes/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch-nD/training-runs/00010--dsmult3-xflip-yflip-rot-clean_and_blurry-twoheads-3_128_128-low_shot_mvds-batch16-color-translation-cutout/out/ \
  --val_dir /home/ffuentes/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch-nD/training-runs/00010--dsmult3-xflip-yflip-rot-clean_and_blurry-twoheads-3_128_128-low_shot_mvds-batch16-color-translation-cutout/out/ \
  --embed_dim 16 \
  --warmup \
  --train_workers 8 \
  --eval_workers 4 \
  --use_mixup_from_epoch 6 \
  --in_channel 3 \
  --out_channel 3

## TODO: FT with manually_selected # need to implement the patch/slices cropping
## TODO: get this data ready: /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2
#python3 ./train.py \
#  --env deblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_3-1D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64 \
#  --arch UNet \
#  --batch_size 4 \
#  --gpu ${GPU_ID} \
#  --train_ps 64 \
#  --nepoch 1500 \
#  --train_dir /home/ffuentes/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/ \
#  --val_dir /home/ffuentes/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/ \
#  --embed_dim 16 \
#  --warmup \
#  --train_workers 8 \
#  --eval_workers 4 \
#  --use_mixup_from_epoch 6 \
#  --pretrain_weights /home/ffuentes/Scratch/nus/20_deblurring_uformer/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4/models/model_best.pth \
#  --resume \
#  --reset_optimizer \
#  --token _down \
#  --data_multiplier 480


# MetaUFormer (Leff + WindPool)

# training with SGEN
python3 ./train.py \
  --env deblur_SGgenMVDS_TRIALS_MetaUformer_LeFF_WindPool_BAR_bs4_3-1D \
  --arch MetaUformer \
  --batch_size 4 \
  --gpu ${GPU_ID} \
  --train_ps 64 \
  --nepoch 150 \
  --train_dir /home/ffuentes/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch-nD/training-runs/00010--dsmult3-xflip-yflip-rot-clean_and_blurry-twoheads-3_128_128-low_shot_mvds-batch16-color-translation-cutout/out/ \
  --val_dir /home/ffuentes/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch-nD/training-runs/00010--dsmult3-xflip-yflip-rot-clean_and_blurry-twoheads-3_128_128-low_shot_mvds-batch16-color-translation-cutout/out/ \
  --embed_dim 16 \
  --warmup \
  --train_workers 8 \
  --eval_workers 4 \
  --use_mixup_from_epoch 6 \
  --in_channel 3 \
  --out_channel 1 \
  --token_mixing wind_pool \
  --token_mlp leff

python3 ./train.py \
  --env deblur_SGgenMVDS_TRIALS_MetaUformer_LeFF_WindPool_BAR_bs4_3-3D \
  --arch MetaUformer \
  --batch_size 4 \
  --gpu ${GPU_ID} \
  --train_ps 64 \
  --nepoch 150 \
  --train_dir /home/ffuentes/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch-nD/training-runs/00010--dsmult3-xflip-yflip-rot-clean_and_blurry-twoheads-3_128_128-low_shot_mvds-batch16-color-translation-cutout/out/ \
  --val_dir /home/ffuentes/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch-nD/training-runs/00010--dsmult3-xflip-yflip-rot-clean_and_blurry-twoheads-3_128_128-low_shot_mvds-batch16-color-translation-cutout/out/ \
  --embed_dim 16 \
  --warmup \
  --train_workers 8 \
  --eval_workers 4 \
  --use_mixup_from_epoch 6 \
  --in_channel 3 \
  --out_channel 3 \
  --token_mixing wind_pool \
  --token_mlp leff

## TODO: FT with manually_selected # need to implement the patch/slices cropping
## TODO: get this data ready: /mnt/data/data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2
#python3 ./train.py \
#  --env deblur_SGgenMVDS_TRIALS_UNet_BAR_bs4_3-1D_FT_MVDSmanually_selected_STACK_5_2_bs4_dmult480_ep1500_ps64 \
#  --arch UNet \
#  --batch_size 4 \
#  --gpu ${GPU_ID} \
#  --train_ps 64 \
#  --nepoch 1500 \
#  --train_dir /home/ffuentes/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/ \
#  --val_dir /home/ffuentes/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/ \
#  --embed_dim 16 \
#  --warmup \
#  --train_workers 8 \
#  --eval_workers 4 \
#  --use_mixup_from_epoch 6 \
#  --pretrain_weights /home/ffuentes/Scratch/nus/20_deblurring_uformer/log/UNetdeblur_SGgenMVDS_TRIALS_UNet_BAR_bs4/models/model_best.pth \
#  --resume \
#  --reset_optimizer \
#  --token _down \
#  --data_multiplier 480

