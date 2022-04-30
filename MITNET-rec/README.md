# MITNET-rec

## Installation

```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-lightning==0.8.5
pip install cnn-finetune==0.6.0
pip install scikit-learn==0.24.1
pip install timm==0.5.4
```
# Training

```
python train.py --savedir '../pt/' --datadir '../data/' 
```

# Testing
```
python test.py --test-path '../test/' --model-weight-path 'HE_mitosis.pt' 
```

### Note: Model weights will not be shared. Only the method is shared.
