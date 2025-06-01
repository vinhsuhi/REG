
### 1. Environment setup

```bash
conda create -n reg python=3.9 -y
conda activate reg
pip install -r requirements.txt
```

### 2. Dataset

#### Dataset download

Currently, we provide experiments for ImageNet. You can place the data that you want and can specifiy it via `--data-dir` arguments in training scripts.

#### Preprocessing data
Please refer to preprocessing guide.

### 3. Training
Run train.sh
```bash
bash train.sh
```

train.sh contains the following content
```bash
accelerate launch --multi_gpu --num_processes $NUM_GPUS train.py \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="SiT-B/2" \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.5 \
    --encoder-depth=4 \     #SiT-L/XL use 8, SiT-B use 4
    --output-dir="your_path" \
    --exp-name="linear-dinov2-b-enc4" \
    --batch-size=256 \
    --data-dir="data_path/imagenet_vae" \
    --cls=0.03
```

Then this script will automatically create the folder in `exps` to save logs and checkpoints. You can adjust the following options:

- `--models`: `[SiT-B/2, SiT-L/2, SiT-XL/2]`
- `--enc-type`: `[dinov2-vit-b]`
- `--proj-coeff`: Any values larger than 0
- `--encoder-depth`: Any values between 1 to the depth of the model
- `--output-dir`: Any directory that you want to save checkpoints and logs
- `--exp-name`: Any string name (the folder will be created under `output-dir`)
- `--cls`: Weight coefficients of REG loss

For DINOv2 models, it will be automatically downloaded from `torch.hub`. For CLIP models, it will be also automatically downloaded from the CLIP repository. For other pretrained visual encoders, please download the model weights from the below links and place into the following directories with these names:

We also support training on 512x512 resolution. Please use the following script:
```bash
accelerate launch train.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-XL/2" \
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.5 \
  --encoder-depth=8 \
  --output-dir="exps" \
  --exp-name="linear-dinov2-b-enc8-in512" \
  --resolution=512 \
  --data-dir="data_path/imagenet_vae" \
  --cls=0.03
```

You also need a new data preprocessing that resizes each image to 512x512 resolution and encodes each image as 64x64 resolution latent vectors (using stable-diffusion VAE). This script is also provided in our preprocessing guide.


### 4. Generate images and evaluation
You can generate images and get the results through the following script:

```bash
bash eval.sh
```







