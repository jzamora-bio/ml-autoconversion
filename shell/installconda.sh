#!/bin/bash

d="/opt/lufac/nn-conda"

# [[ -d ${d} ]] && rm -rf ${d}

# wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
# bash Miniconda3-latest-Linux-x86_64.sh -p ${d} -b -f
export PATH=${d}/bin:$PATH
conda init bash
conda activate ${d}
conda info
echo "##############################"

conda update -y -n base -c defaults conda
#conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -y pytorch torchvision -c pytorch
conda install -y matplotlib pandas scikit-learn colorama
conda install -y tensorflow-gpu

rm -rf Miniconda3-latest-Linux-x86_64.sh