# Run experiments
conda init
conda activate ics3

export PYTHONPATH=$(pwd)
export HYDRA_FULL_ERROR=1

python src/run.py --config-name unet.yaml
python src/run.py --config-name vit.yaml
python src/run.py --config-name ics3.yaml