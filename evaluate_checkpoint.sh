declare -a gpu=$1
declare -a manual_seed=(21)
declare -a checkpoints=(str_80)
declare -a transfer_dsets=$2
declare -a dset_paths=$3

for ((j=0;j<${#manual_seed[@]};++j));
do
for ((i=0;i<${#checkpoints[@]};++i)); 
do
python main.py \
	--dset=$2 \
	--dset_path=$3 \
	--config_path=./configs/imagenet_transfer_small_lr.yaml \
	--workers=4 \
	--epochs=1 \
	--eval_only \
	--batch_size=64 \
	--from_checkpoint_path="../full_transfer_checkpoints/pets/AC-DC/0.8/seed_23.pth" \
	--gpus=$1 \
        --manual_seed=${manual_seed[j]} \
	--experiment_root_path "./experiments_transfer" \
	--exp_name="${checkpoints[i]}-${transfer_dsets}-small-lr" \
	--wandb_name="${checkpoints[i]}-${transfer_dsets}-small-lr" \
	--wandb_group="${transfer_dsets}" \
        --wandb_project "sparse-imagenet-transfer-fullnetwork" 


done
done
