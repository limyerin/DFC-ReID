# Third-party code and attribution

This repository is derived from the official implementation of
"Low-light Invariant Representation Learning for Visible-Infrared Person
Re-identification" (IRL, ACM MM 2025) by Dengwen Wang, Guanyu Xing and Yanli Liu
(<https://github.com/Mapzzone/2025-ACMMM-IRL>, MIT License, Copyright (c) 2025 Mapzzone).
That code in turn borrows from AGW (Ye et al.), DEEN/LLCM (Zhang & Wang) and
CAJ (Ye et al.), as stated in the original README. The original MIT license
notice is kept in `LICENSE`.

## Files taken from the IRL repository (unchanged or lightly modified)

`solver/build_lrscheduler.py`, `solver/build_optimizer.py`, `solver/lr_scheduler.py`,
`data/channel_aug.py`, `data/data_loader.py`, `data/data_manager.py`,
`engine/eval_metrics.py`, `engine/loss.py`, `models/LPA.py`, `models/NAF.py`,
`models/resnet.py`, `tools/pre_process_sysu.py`

- `data/build_transforms.py`: added `transforms.Resize((H, W))` to the four training transforms.
- `engine/utils.py`: `Logger` gained a `mode` argument and flushes after each write.
- `tools/pre_process_sysu.py`: dataset path may be passed as a command-line argument.

## Files derived from the IRL repository with substantial changes

- `models/model_idfc.py` (from `model.py`): optional ICG gate at NAF1/NAF2; exposes pre-/post-compensation pooled features for DCL.
- `train_idfc.py` (from `train.py`): `--method {baseline,icg,dcl,idfc}`, DCL loss, SGD setup, checkpoint/resume, smoke-test options, `--data_root`.
- `test_idfc.py` (from `test.py`): `--idfc_method` to build the matching model, `--data_root`.

## Files written for this work

`models/idfc_modules.py`, `tools/measure_dcl_margin.py`, `tools/plot_retrieval.py`

## Not included

The original `train.py`, `test.py` (and the file layout of the IRL repository, which was reorganised into packages here), `testD.py`, `CLIP_resnet/`, `CRS.py` and `VisualizeH/`
(including Grad-CAM code with its own license) are not part of this repository.
Datasets (SYSU-MM01, RegDB, LLCM) and trained checkpoints are not redistributed.
