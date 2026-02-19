# Jaguar ReId

## Environment
Create a `.env` file in the root directory with the following variables:
```
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_api_key
```

```
poetry install
source $(poetry env info --path)/bin/activate
```

## Cluster
```
salloc --partition gpu-interactive --account=sci-demelo-computer-vision --cpus-per-task=32 --mem=100G --gpus=1 --time=08:00:00
```

## Experiments

## Exploratory Data Analysis


## Leaderboard Experiments
