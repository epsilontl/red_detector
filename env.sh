#! /bin/bash
conda create -n red python=3.6 -y
conda activate red
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ -y
pip install --upgrade pip
pip install opencv-python tensorboard tqdm pyyaml matplotlib numba
cd event_representation_tool
python setup.py install
cd ../apex
python setup.py install
