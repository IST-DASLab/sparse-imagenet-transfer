declare -a manual_seed=(21)
declare -a checkpoints=(str_80_noLS)
declare -a transfer_dset=$1
declare -a dset_path=$2

for ((j=0;j<${#manual_seed[@]};++j));
do
for ((i=0;i<${#checkpoints[@]};++i)); 
do
python main.py \
	--dset=${transfer_dset} \
	--dset_path=${dset_path} \
	--config_path="./configs/imagenet_transfer_deepsparse.yaml" \
	--cpu \
	--apply_deepsparse \
	--workers=4 \
	--epochs=150 \
	--batch_size=64 \
	--from_checkpoint_path="../upstream_checkpoints/${checkpoints[i]}.pth" \
	--gpus=${gpu} \
        --manual_seed=${manual_seed[j]} \
	--experiment_root_path "./experiments_transfer" \
	--transfer_config_path="configs/imagenet_transfer_transfer.yaml" \
	--exp_name="${checkpoints[i]}-${transfer_dset}" \
	--use_wandb \
	--wandb_name="${checkpoints[i]}-${transfer_dset}" \
	--wandb_group="${transfer_dset}" \
        --wandb_project "sparse-imagenet-transfer-deepsparse" 
done
done
