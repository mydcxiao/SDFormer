# Diffusion Model As an Image Encoder + CGFormer
The official PyTorch implementation of diffusion model as an image encoder and CGFormer as an decoder, which is from the CVPR 2023 paper "Contrastive Grouping with Transformer for Referring Image Segmentation". The paper first introduces learnable query tokens to represent objects and then alternately queries linguistic features and groups visual features into the query tokens for object-aware cross-modal reasoning. CGFormer achieves cross-level interaction by jointly updating the query tokens and decoding masks in every two consecutive layers.

## Framework
<p align="center">
  <img src="image/framework.png" width="1000">
</p>

## Preparation

1. Environment
   - [PyTorch](www.pytorch.org)
   - [Stable Diffusion dependencies](https://github.com/CompVis/stable-diffusion)
   - Other dependencies in `requirements.txt`
2. Datasets
   - The detailed instruction is in [prepare_datasets](data/READEME.md)
3. Pretrained weights
   - [Swin-Base-window12](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth)
   - Diffusion Encoder of ODISE, refer to [ODISE](https://github.com/NVlabs/ODISE/tree/main)

## Train and Test (RIS)

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported. Besides, the evaluation only supports single-gpu mode.

To do training of CGFormer with multiple GPUs, run:

```
python -u train.py --config config/config.yaml
```

To do evaluation of CGFormer with 1 GPU, run:
```
CUDA_VISIBLE_DEVICES=0 python -u test.py \
      --config config/config.yaml \
      --opts TEST.test_split val \
             TEST.test_lmdb data/refcoco/val.lmdb
```
## Results

|     Dataset     | P@0.5 | P@0.6 | P@0.7 | P@0.8 | P@0.9 | Overall IoU | Mean IoU |
|:---------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----------:|:--------:|
| RefCOCO val     | 72.12 | 66.31 | 59.29 | 49.05 | 24.53 |    63.41    |   64.65  |
| RefCOCO test A  | 75.75 | 70.81 | 64.35 | 52.82 | 26.62 |    66.14    |   67.39  |
| RefCOCO test B  | 67.81 | 61.96 | 54.64 | 43.38 | 22.81 |    60.74    |   61.66  |

## License

This project is under the MIT license. See [LICENSE](LICENSE) for details.


Some code changes come from [CRIS](https://github.com/DerrickWang005/CRIS.pytorch/tree/master), [CGFormer](https://github.com/Toneyaya/CGFormer), [ODISE](https://github.com/NVlabs/ODISE/tree/main) and [LAVT](https://github.com/yz93/LAVT-RIS).
