## DFC-ReID: Discriminative Feature Compensation for Visible-Infrared Person Re-Identification

Implementation of the paper "<u>DFC-ReID: Discriminative Feature Compensation for Visible-Infrared Person Re-Identification</u>".

This code is built on the official implementation of IRL [6] (*Low-light Invariant Representation Learning for Visible-Infrared Person Re-identification*, ACM MM 2025, [Code](https://github.com/Mapzzone/2025-ACMMM-IRL), MIT License). DFC-ReID adds a Discriminative Compensation Loss (DCL) on top of the IRL baseline (`--method dcl`). See [NOTICE.md](NOTICE.md) for exactly which files come from IRL and which were added or changed.

*The original paper's `L_cl` loss is not included; the base is the publicly released two-loss IRL code.*

### 1. Requirements

Our experiments are conducted under the following environments:

- Python 3.8.20
- Pytorch == 2.1.2 (CUDA 12.1)
- torchvision == 0.16.2
- GPU: NVIDIA GeForce RTX 3090 (24 GB)

```
conda create -n dfc python=3.8
conda activate dfc
# install torch 2.1.2 / torchvision 0.16.2 for your CUDA from https://pytorch.org
pip install -r requirements.txt
```

### 2. Datasets

- SYSU-MM01 [2]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

  - Place it as `dataset/sysu-mm01/` (or pass the parent directory by `--data_root`), then run `tools/pre_process_sysu.py` to prepare the dataset, the training data will be stored in ".npy" format.

    ```
    python tools/pre_process_sysu.py dataset/sysu-mm01
    ```

Only SYSU-MM01 was used for the experiments reported here. The RegDB [1] (`<data_root>/RegDB`) and LLCM [5] (`<data_root>/LLCM`) branches inherited from IRL are kept in the code but were not run.

### 3. Training

**Train DFC-ReID (DCL only) by**

```
python train_idfc.py --dataset sysu --method dcl --epochs 150 --arch resnet50 --backbone AGW \
  --gpu 0 --seed 0 --trial 1 --mode all --batch-size 4 --num_pos 4 --img_w 192 --img_h 384 \
  --lr 0.1 --optim SGD --loss_tri QCT --margin 0.7 --lambda_dcl 0.05 --dcl_margin 0.02 \
  --test-batch 64 --workers 4 --eval_every 5 --save_epoch 20 \
  --model_path result/dcl_l005/ --log_path result/dcl_l005/ --vis_log_path result/dcl_l005/
```

**Train the baseline (IRL without DCL) by**

```
python train_idfc.py --dataset sysu --method baseline --gpu 0 --batch-size 4 --num_pos 4 \
  --epochs 150 --eval_every 5 --workers 4 --model_path result/ --log_path result/ --vis_log_path result/
```

- `--method`: "baseline" or "dcl" (DFC-ReID). The code also contains "icg" and "idfc" (ICG + DCL), which are additional experimental variants and are not part of the final method.
- `--lambda_dcl`: weight of the DCL loss.
- `--dcl_margin`: margin of the DCL loss.
- `--data_root`: directory containing `sysu-mm01/` (default `./dataset`).
- `--gpu`: which gpu to run.
- `--epochs 1 --max_steps 2 --workers 0`: smoke test (2 steps, evaluation and checkpoints skipped).
- `--resume`: basename of a `*_latest.t` checkpoint (with the same `--model_path`). Resuming restores model, optimizer, scheduler, GradScaler and epoch, but not RNG/sampler state, so it is not bitwise reproducible.

### 4. Testing

**Test a model on SYSU-MM01 dataset by**

```
python test_idfc.py --dataset sysu --gpu 0 --test-batch 64 --idfc_method dcl \
  --model_path result/dcl_l005/ --resume sysu_dcl_p4_n4_lr_0.1_seed_0_latest.t --mode all
```

  - `--mode`: "all" or "indoor".
  - `--idfc_method`: "baseline" or "dcl"; it must match the `--method` used for training.
  - `--model_path`, `--resume`: directory and file name of the saved model.
  - `--data_root`: directory containing `sysu-mm01/`.

The script evaluates 10 gallery trials and prints the average (FC = final embedding, POOL = pooled feature).

**Analysis and visualization**

```
python tools/measure_dcl_margin.py --data_path dataset/sysu-mm01 --gpu 0 \
  --model baseline:baseline:<ckpt> --model dcl:dcl:<ckpt>

python tools/plot_retrieval.py --data_path dataset/sysu-mm01 --baseline <baseline_ckpt> --ours <dcl_ckpt> \
  --rows improved:4 --rows failure:1 --topk 10
```

### 5. Results

We adopt the CNN-based AGW [3] as backbone. SYSU-MM01, FC feature, average over 10 gallery trials, checkpoint at epoch 150, single run (seed 0).

| Datasets | Mode | Method | Rank@1 | Rank@10 | mAP | mINP |
| :------: | :--: | :----: | :----: | :-----: | :---: | :---: |
| SYSU-MM01 | all | Baseline (IRL) | 74.89% | 96.96% | 70.63% | 56.54% |
| SYSU-MM01 | all | **DFC-ReID** | 77.88% | 97.29% | 73.69% | 60.53% |
| SYSU-MM01 | indoor | Baseline (IRL) | 80.78% | 98.55% | 83.63% | 79.98% |
| SYSU-MM01 | indoor | **DFC-ReID** | 82.51% | 98.22% | 84.83% | 81.51% |

**\*The results are from a single run per setting and may exhibit fluctuations due to random splitting and seeds.**

### 6. Citation

Most of the code of our backbone are borrowed from [AGW](https://github.com/mangye16/Cross-Modal-Re-ID-baseline) [3] and [IRL](https://github.com/Mapzzone/2025-ACMMM-IRL) [6]. Most of the code related to LLCM dataset are borrowed from [DEEN](https://github.com/ZYK100/LLCM) [5].

Thanks a lot for the author's contribution.

### Repository layout

```
train_idfc.py  test_idfc.py      # entry points
models/    model_idfc.py, idfc_modules.py, resnet.py, LPA.py, NAF.py
data/      data_loader.py, data_manager.py, channel_aug.py, build_transforms.py
engine/    loss.py, eval_metrics.py, utils.py
solver/    lr_scheduler.py, build_lrscheduler.py, build_optimizer.py
tools/     pre_process_sysu.py, measure_dcl_margin.py, plot_retrieval.py
```

### License

MIT, see [LICENSE](LICENSE) (original copyright notice retained) and [NOTICE.md](NOTICE.md).

###  References

[1] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.

[2] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.

[3] Ye M, Shen J, Lin G, et al. Deep learning for person re-identification: A survey and outlook[J]. IEEE transactions on pattern analysis and machine intelligence, 2021, 44(6): 2872-2893.

[4] Ye M, Ruan W, Du B, et al. Channel augmented joint learning for visible-infrared recognition[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 13567-13576.

[5] Zhang Y, Wang H. Diverse Embedding Expansion Network and Low-Light Cross-Modality Benchmark for Visible-Infrared Person Re-identification[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 2153-2162.

[6] Wang D, Xing G, Liu Y. Low-light Invariant Representation Learning for Visible-Infrared Person Re-identification[C]//Proceedings of the 33rd ACM International Conference on Multimedia. 2025: 8645-8653.
