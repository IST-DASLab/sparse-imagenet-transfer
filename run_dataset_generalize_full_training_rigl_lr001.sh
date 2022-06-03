declare -a gpu=$1
declare -a manual_seed=(21)
declare -a checkpoints=(rigl_erk5x_90)
declare -a transfer_dset=$2
declare -a dset_path=$3

for ((j=0;j<${#manual_seed[@]};++j));
do
for ((i=0;i<${#checkpoints[@]};++i)); 
do
python main.py \
	--dset=${transfer_dset} \
	--dset_path=${dset_path} \
	--interpolation="bicubic" \
	--config_path="./configs/imagenet_transfer_small_lr.yaml" \
	--workers=4 \
	--epochs=150 \
	--batch_size=64 \
	--from_checkpoint_path="../upstream_checkpoints/${checkpoints[i]}.pth" \
	--gpus=${gpu} \
        --manual_seed=${manual_seed[j]} \
	--experiment_root_path "./experiments_transfer" \
	--transfer_config_path="configs/imagenet_transfer_fullnetwork_transfer_rigl.yaml" \
	--exp_name="${checkpoints[i]}-${transfer_dset}-bicubic-small-lr" \
	--use_wandb \
	--wandb_name="${checkpoints[i]}-${transfer_dset}-bicubic-small-lr" \
	--wandb_group="${transfer_dset}" \
        --wandb_project "sparse-imagenet-transfer-fullnetwork" 
done
done
