# universal-voice-activity-detection
Universal VAD pipeline, trained on a variety of datasets.


Set-up

```
conda create --file environment.yml
```

If you plan to use weights and biases for logging, then first create an `.env` file in the root directory with the following structure

```
WANDB_API_KEY=
```

Then, run the follwoing command on the console
```
wandb login
```