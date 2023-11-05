#/bin/bash
contentvec="https://huggingface.co/spaces/Stonty/sovits-models/resolve/main/checkpoint_best_legacy_500.pt"
D_0="https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/resolve/main/sovits_768l12_pre_large_320k/clean_D_320000.pth"
G_0="https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/resolve/main/sovits_768l12_pre_large_320k/clean_G_320000.pth"
diffusion="https://huggingface.co/ChiTu/Diffusion-SVC/resolve/main/v0.1/contentvec768l12.7z"

nsf_hifigan="https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip"
rmvpe="https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip"
fcpe="https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/fcpe.pt"

work_dir=$(pwd)

sudo apt-get update && sudo apt-get dist-upgrade -y
sudo apt-get install git curl wget unzip tar python3 python3-pip python3-venv p7zip -y
git clone https://github.com/sovits-combined/so-vits-svc.git main
main=${work_dir}/main

curl -L -C - -o ${main}/logs/44k/D_0.pth ${D_0}

curl -L -C - -o ${main}/logs/44k/G_0.pth ${G_0}

curl -L -C - -o ${main}/logs/44k/diffusion/model.7z ${diffusion}
cd ${main}/logs/44k/diffusion
p7zip -d model.7z
cd ${work_dir}

curl -L -C - -o ${main}/pretrain/checkpoint_best_legacy_500.pt ${contentvec}

curl -L -C - -o ${main}/pretrain/nsf_hifigan.zip ${nsf_hifigan}
unzip -d ${main}/pretrain/ ${main}/pretrain/nsf_hifigan.zip
rm -rf ${main}/pretrain/nsf_hifigan.zip

curl -L -C - -o ${main}/pretrain/rmvpe.zip ${rmvpe}
unzip -d ${main}/pretrain/ ${main}/pretrain/rmvpe.zip
rm -rf ${main}/pretrain/rmvpe.zip
mv ${main}/pretrain/model.pt ${main}/pretrain/rmvpe.pt

curl -L -C - -o ${main}/pretrain/fcpe.pt ${fcpe}

python3 -m venv ${work_dir}/venv
source ${work_dir}/venv/bin/activate

pip install --upgrade pip
pip install wheel
pip install -r ${main}/requirements.txt