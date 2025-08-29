conda create -n reg python=3.10.16 -y
conda activate reg
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1
pip install -r requirements.txt
