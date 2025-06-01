


#512
#python preprocessing/dataset_tools.py convert \
#    --source=/home/share/imagenet/train \
#    --dest=/home/share/imagenet_vae/imagenet_512_vae \
#    --resolution=512x512 \
#    --transform=center-crop-dhariwal


#256
python preprocessing/dataset_tools.py convert \
    --source=/home/share/imagenet/train \
    --dest=/home/share/imagenet_vae/imagenet_256_vae \
    --resolution=256x256 \
    --transform=center-crop-dhariwal