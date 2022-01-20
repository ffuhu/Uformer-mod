export CUDA_VISIBLE_DEVICES=6

# Uformer16
# python3 ./train.py --arch Uformer --batch_size 32 --gpu '0,1' \
#     --train_ps 128 --train_dir /cache/SIDD/train --env 16_0701_1 \
#     --val_dir /cache/SIDD/val --embed_dim 16 --warmup

# Uformer32
# python3 ./train.py --arch Uformer --batch_size 32 --gpu '0,1' \
#     --train_ps 128 --train_dir /cache/SIDD/train --env 32_0701_1 \
#     --val_dir /cache/SIDD/val --embed_dim 32 --warmup

    
# UNet
# python3 ./train.py --arch UNet --batch_size 32 --gpu '0,1' \
#     --train_ps 128 --train_dir /cache/SIDD/train --env 32_0701_1 \
#     --val_dir /cache/SIDD/val --embed_dim 32 --warmup

# python3 ./train.py
#  --env delur_SGgenMVDS \
#  --arch Uformer \
#  --batch_size 4 \
#  --gpu "0" \
#  --train_ps 64 \
#  --nepoch 75 \
#  --train_dir /home/felix/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch/training-runs/00009--low_shot-color-translation-cutout_ABdiff_ps128px/out/ \
#  --val_dir /home/felix/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch/training-runs/00009--low_shot-color-translation-cutout_ABdiff_ps128px/out/ \
#  --embed_dim 16 \
#  --warmup \
#  --train_workers 0 \
#  --eval_workers 0 \
#  --use_mixup_from_epoch 6


#python3 ./train.py --env delur_SGgenMVDS_FT_MVDSmanually_selected_5_2 \
#--arch Uformer \
#--batch_size 4 \
#--gpu "0" \
#--train_ps 64 \
#--nepoch 75 \
#--train_dir /home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/ \
#--val_dir /home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/ \
#--embed_dim 16 \
#--warmup \
#--train_workers 8 \
#--eval_workers 4 \
#--use_mixup_from_epoch 6 \
#--pretrain_weights /home/felix/Scratch/nus/20_deblurring_uformer/log/Uformerdelur_SGgenMVDS/models/model_best.pth \
#--resume \
#--reset_optimizer \
#--token _down \
#--data_multiplier 16

#python3 ./train.py \
#  --env delur_SGgenMVDS_FT_MVDSmanually_selected_5_2_resume_noRstOpt_500ep+ \
#  --arch Uformer \
#  --batch_size 16 \
#  --gpu "0" \
#  --train_ps 64 \
#  --nepoch 500 \
#  --train_dir /home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/ \
#  --val_dir /home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/ \
#  --embed_dim 16 \
#  --warmup \
#  --train_workers 8 \
#  --eval_workers 4 \
#  --use_mixup_from_epoch 6 \
#  --pretrain_weights /home/felix/Scratch/nus/20_deblurring_uformer/log/Uformerdelur_SGgenMVDS_FT_MVDSmanually_selected_5_2/models/model_best.pth \
#  --resume \
#  --token _down \
#  --data_multiplier 480

#python3 ./train.py \
#  --env delur_SGgenMVDS_FT_MVDSmanually_selected_5_2_resume_noRstOpt_500ep+ \
#  --arch Uformer \
#  --batch_size 16 \
#  --gpu "0" \
#  --train_ps 64 \
#  --nepoch 1500 \
#  --train_dir /home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/ \
#  --val_dir /home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/ \
#  --embed_dim 16 \
#  --warmup \
#  --train_workers 8 \
#  --eval_workers 4 \
#  --use_mixup_from_epoch 6 \
#  --pretrain_weights /home/felix/Scratch/nus/20_deblurring_uformer/log/Uformerdelur_SGgenMVDS_FT_MVDSmanually_selected_5_2_resume_noRstOpt_500ep+/models/model_best.pth \
#  --resume \
#  --token _down \
#  --data_multiplier 480

#python3 ./train.py \
#  --env deblur_SGgenMVDS_FT_MVDSmanually_selected_5_2_5000ep_bs32_ps64px_dmult480 \
#  --arch Uformer \
#  --batch_size 32 \
#  --gpu "6" \
#  --train_ps 64 \
#  --nepoch 5000 \
#  --train_dir ${HOME}/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/ \
#  --val_dir ${HOME}/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/ \
#  --embed_dim 16 \
#  --warmup \
#  --train_workers 8 \
#  --eval_workers 4 \
#  --use_mixup_from_epoch 6 \
#  --pretrain_weights ${HOME}/Scratch/nus/20_deblurring_uformer/log/Uformerdelur_SGgenMVDS/models/model_best.pth \
#  --resume \
#  --token _down \
#  --data_multiplier 480


## TEST META_UFORMER
#
#python3 ./train.py \
#  --env deblur_SGgenMVDS_TRIALS_MetaUFORMER_LeFF_Pool_ASUS_bs4 \
#  --arch MetaUformer \
#  --batch_size 4 \
#  --gpu "0" \
#  --train_ps 64 \
#  --nepoch 75 \
#  --train_dir /home/felix/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch/training-runs/00009--low_shot-color-translation-cutout_ABdiff_ps128px/out/ \
#  --val_dir /home/felix/Scratch/nus/24-data-efficient-gans/DiffAugment-stylegan2-pytorch/training-runs/00009--low_shot-color-translation-cutout_ABdiff_ps128px/out/ \
#  --embed_dim 16 \
#  --warmup \
#  --train_workers 8 \
#  --eval_workers 4 \
#  --use_mixup_from_epoch 6 \
#  --token_mixing pool \
#  --token_mlp leff
#
## todo: finetuning SGgenMVDS with manually_selected_5_2 POOL
#python3 ./train.py \
#  --env deblur_SGgenMVDS_TRIALS_MetaUFORMER_LeFF_Pool_ASUS_bs4_FT_MVDSmanually_selected_5_2_bs4_dmult480_ep1500_ps64 \
#  --arch MetaUformer \
#  --batch_size 4 \
#  --gpu "0" \
#  --train_ps 64 \
#  --nepoch 1500 \
#  --train_dir /home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/ \
#  --val_dir /home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/ \
#  --embed_dim 16 \
#  --warmup \
#  --train_workers 8 \
#  --eval_workers 4 \
#  --use_mixup_from_epoch 6 \
#  --pretrain_weights /home/felix/Scratch/nus/20_deblurring_uformer/log/MetaUformerdeblur_SGgenMVDS_TRIALS_MetaUFORMER_LeFF_Pool_ASUS_bs4/models/model_best.pth \
#  --resume \
#  --reset_optimizer \
#  --token _down \
#  --token_mixing pool \
#  --token_mlp leff \
#  --data_multiplier 480

## TEST NO SGAN DATA - DONE: WORKS WORSE
#python3 ./train.py \
#  --env deblur_MVDSmanually_selected_5_2_ep500 \
#  --arch Uformer \
#  --batch_size 4 \
#  --gpu "0" \
#  --train_ps 64 \
#  --nepoch 500 \
#  --train_dir /home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/ \
#  --val_dir /home/felix/Scratch/nus/00_data/data_Florian/Dataset_Multiview/denoised/cropped_110_110_1024_1024/denoised_cropped_down_512px/good/manually_selected_5-2/ \
#  --embed_dim 16 \
#  --warmup \
#  --train_workers 8 \
#  --eval_workers 4 \
#  --use_mixup_from_epoch 6 \
#  --token _down \
#  --data_multiplier 16
