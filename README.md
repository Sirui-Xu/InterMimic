<p align="center">
<!-- <h1 align="center">InterDiff: Generating 3D Human-Object Interactions with Physics-Informed Diffusion</h1> -->
<h1 align="center"><sup><img src="assets/logo.png" align="center" width=4% ></sup> <strong>InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions</strong></h1>
  <p align="center">
    <a href='https://sirui-xu.github.io/' target='_blank'>Sirui Xu</a><sup><img src="assets/Illinois.jpg" align="center" width=2% ></sup>&emsp;
    <a href='https://hungyuling.com/' target='_blank'>Hung Yu Ling</a> <sup><img src="assets/Electronic-Arts-Logo.png" align="center" width=1.5% ></sup>&emsp;
    <a href='https://yxw.web.illinois.edu/' target='_blank'>Yu-Xiong Wang</a><sup><img src="assets/Illinois.jpg" align="center" width=2% ></sup>&emsp;
    <a href='https://lgui.web.illinois.edu/' target='_blank'>Liang-Yan Gui</a><sup><img src="assets/Illinois.jpg" align="center" width=2% ></sup>&emsp;
    <br>
    <sup><img src="assets/Illinois.jpg" align="center" width=2% ></sup>University of Illinois Urbana-Champaign, <sup><img src="assets/Electronic-Arts-Logo.png" align="center" width=1.5% ></sup> Electronic Arts
    <br>
    <strong>CVPR 2025 Highlight 🏆</strong>
  </p>
</p>

</p>
<p align="center">
  <a href='https://arxiv.org/abs/2502.20390'>
    <img src='https://img.shields.io/badge/Arxiv-2502.20390-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
  <a href='https://arxiv.org/pdf/2502.20390.pdf'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'></a>
  <a href='https://sirui-xu.github.io/InterMimic/'>
    <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green'></a>
  <a href='https://youtu.be/ZJT387dvI9w'>
    <img src='https://img.shields.io/badge/YouTube-Video-EA3323?style=flat&logo=youtube&logoColor=EA3323'></a>
  <a href='https://www.bilibili.com/video/BV1nW9KYFEUX/'>
    <img src='https://img.shields.io/badge/Bilibili-Video-4EABE6?style=flat&logo=Bilibili&logoColor=4EABE6'></a>
  <a href='https://github.com/Sirui-Xu/InterMimic'>
    <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>
</p>

## 🏠 Overview
<div align="center">
  <img src="assets/teaser.png" width="100%" alt="InterMimic teaser"/>
</div>

> **InterMimic** features **one** unified policy, spanning **diverse full-body interactions** with **dynamic, heterogeneous objects**—and it works out-of-the-box for both **SMPL-X** and **Unitree G1** humanoids.


## 📹 Demo
<p align="center">
    <img src="assets/InterMimic.gif" align="center" width=60% >
</p>

## 🔥 News
- **[2025-12-17]** 🚀 Isaac Gym checkpoints are compatible with IsaacLab inference?! Check out the newly released implementation.
- **[2025-12-15]** 🚀 IsaacLab support is underway! Data replay is ready—more coming in the next release ☕️
- **[2025-12-07]** 🚀 Release a data conversion pipeline for bringing [InterAct](https://github.com/wzyabcas/InterAct) into simulation. The processing code is available in the [InterAct repository](https://github.com/wzyabcas/InterAct).
- **[2025-06-10]** Release the instruction for the student policy inference.
- **[2025-06-03]** Initial release of PSI and the processed data. Next release: teacher policy inference for [dynamics-aware retargeting](InterAct/OMOMO_retarget), and student policy inference.
- **[2025-05-26]** It's been a while! The student policy training pipeline has been released! The PSI and other data construction pipelines will follow soon.
- **[2025-04-18]** Release a checkpoint with high‑fidelity physics and enhanced contact precision.
- **[2025-04-11]** The training code for teacher policies is live—try training your own policy!
- **[2025-04-05]** We're excited by the overwhelming interest in humanoid robot support and are ahead of schedule in open-sourcing our Unitree-G1 integration—starting with a small demo with support for G1 with its original three-finger dexterous hands. Join us in exploring whole-body loco-manipulation with humanoid robots!
- **[2025-04-04]** InterMimic has been selected as a CVPR Highlight Paper 🏆. More exciting developments are on the way!
- **[2025-03-25]** We’ve officially released the codebase and checkpoint for teacher policy inference demo — give it a try! ☕️  

## 📖 Getting Started

### Dependencies

#### Isaac Gym environment

1. Create a dedicated conda environment (Python 3.8) and install PyTorch + repo deps:

    ```bash
    conda create -n intermimic-gym python=3.8
    conda activate intermimic-gym
    conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
    pip install -r requirement.txt
    ```

    (Alternatively, start from [environment.yml](environment.yml), though it includes some optional extras.)

2. Install [Isaac Gym](https://developer.nvidia.com/isaac-gym) following NVIDIA’s instructions.

3. Fix the Isaac Gym shared-library lookup when using conda by exporting:

    ```bash
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ```

    Do this after every `conda activate intermimic-gym` before launching Gym scripts; it ensures `libpython3.8.so` is discoverable.

#### Isaac Lab environment

- Install Isaac Lab separately by following the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) and keep that environment isolated (typically via Isaac Sim’s python or the provided uv/conda env).
- Export `ISAACLAB_PATH` once per shell session so our helper scripts (which source `$ISAACLAB_PATH/isaaclab.sh`) can locate your install:

    ```bash
    export ISAACLAB_PATH=/path/to/your/IsaacLab
    ```
- Optional: if you plan to use the `--record-video` flag in our replay script, install `imageio` (and `imageio-ffmpeg` for MP4 support) inside the Isaac Lab Python environment:

    ```bash
    $ISAACLAB_PATH/isaaclab.sh -p -m pip install --upgrade imageio imageio-ffmpeg
    ```

#### Shared data

- Download the [dataset](https://drive.google.com/file/d/141YoPOd2DlJ4jhU2cpZO5VU5GzV_lm5j/view?usp=sharing), unzip it, and move the extracted folder to `InterAct/OMOMO_new/`. *This build contains minor fixes to the original release, so your results may deviate slightly from those reported in the paper.*

### Data Replay

To replay the ground-truth data you now have two options:

**Isaac Gym (legacy)**

```bash
sh isaacgym/scripts/data_replay.sh
```

**Isaac Lab / Isaac Sim**

```bash
./isaaclab/scripts/run_data_replay.sh --num-envs 8 --motion-dir InterAct/OMOMO_new
```

Helpful flags for the Isaac Lab demo:

- `--num-envs`: sets both `cfg.num_envs` and `cfg.scene.num_envs`.
- `--headless`: launches Isaac Sim without the viewer.
- `--motion-dir`: dataset directory relative to `$INTERMIMIC_PATH`.
- `--no-playback`: disables dataset playback so you can step physics manually.
- `--record-video /path/to/video.mp4`: captures RGB frames each step (requires `imageio`).
- `--video-fps`: frame rate for `--record-video` captures (defaults to 30 FPS).

### Teacher Policy Training


To train a teacher policy, execute the following commands:

  ```bash
  sh isaacgym/scripts/train_teacher.sh
  ```

A higher‑fidelity simulation enough for low-dynamic interaction (trading off some efficiency for realism):

  ```bash
  sh isaacgym/scripts/train_teacher_new.sh
  ```

**How to enable PSI**

Open the training config, for example, [`omomo_train_new.yaml`](./isaacgym/src/intermimic/data/cfg/omomo_train_new.yaml). Set

   ```yaml
   physicalBufferSize: <integer greater than 1>
   ```

### Student Policy Training


Download the [data from teacher's retargeting and correction](https://drive.google.com/file/d/1l2E5qR97Ap8jrLrJPHmtNT8DDW1qKhY_/view?usp=sharing), to train a student policy with distillation, execute the following commands:

  ```bash
  sh isaacgym/scripts/train_student.sh
  ```

### Teacher Policy Inference


We’ve released a checkpoint for one (out of 17) teacher policy on OMOMO, along with some sample data. To get started:

1. Download the [checkpoints](https://drive.google.com/drive/folders/1biDUmde-h66vUW4npp8FVo2w0wOcK2_k?usp=sharing) and place them in the current directory.
2. Then, run the following commands:

    ```bash
    sh isaacgym/scripts/test_teacher.sh
    ```

3. Run the high‑fidelity modeling (trading off some efficiency for realism):

    ```bash
    sh isaacgym/scripts/test_teacher_new.sh
    ```

4. 🔥 To try it on the Unitree G1 with its three-fingered dexterous hand—directly learned from MoCap without any external retargeting:

    ```bash
    sh isaacgym/scripts/test_g1.sh
    ```

5. 🔥 Test policy inference on IsaacLab using Isaac Gym checkpoints (requires `ISAACLAB_PATH` to be set):

    ```bash
    export ISAACLAB_PATH=/path/to/IsaacLab
    ./isaaclab/scripts/test_policy.sh --checkpoint checkpoints/smplx_teachers/sub2.pth --num_envs 16
    ```

    Options:
    - `--checkpoint PATH`: Path to checkpoint file
    - `--num_envs N`: Number of environments (default: 16)
    - `--headless`: Run without rendering

    The test suite validates environment creation, observation dimensions, and basic stepping. Core testing logic is in `isaaclab/examples/test_policy_inference.py`.

### Student Policy Inference


After finish the student policy training, run the inference with

  ```bash
  sh isaacgym/scripts/test_student.sh
  ```

Alternatively, you may try one of our pre-trained [checkpoints](https://drive.google.com/file/d/1GNFOjBRmiIIxYtfnG9WvK4fELKnDWroR/view?usp=sharing)

## 📝 TODO List  
- [x] Release inference demo for the teacher policy  
- [x] Add support for Unitree-G1 with dexterous robot hands
- [x] Release training pipeline for the teacher policy 
- [x] Release student policy distillation training
- [x] Release processed MoCap
- [x] Release inference pipeline for the student policy 
- [x] Distilled reference data (physically correct HOI data❗️)
- [x] Release all related checkpoints   
- [x] Release all data and processing scripts alongside the [InterAct](https://github.com/wzyabcas/InterAct) launch  
- [ ] Release physics-based text-to-HOI and interaction prediction demo  


## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{xu2025intermimic,
  title = {{InterMimic}: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions},
  author = {Xu, Sirui and Ling, Hung Yu and Wang, Yu-Xiong and Gui, Liang-Yan},
  booktitle = {CVPR},
  year = {2025},
}
```

Our data is sourced from **InterAct**. Please consider citing:

```bibtex
@inproceedings{xu2025interact,
  title = {{InterAct}: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation},
  author = {Xu, Sirui and Li, Dongting and Zhang, Yucheng and Xu, Xiyan and Long, Qi and Wang, Ziyin and Lu, Yunzhi and Dong, Shuchang and Jiang, Hezi and Gupta, Akshat and Wang, Yu-Xiong and Gui, Liang-Yan},
  booktitle = {CVPR},
  year = {2025},
}
```
Please also consider citing the specific sub-dataset you used from **InterAct**.

Our integrated kinematic model builds upon **InterDiff**, **HOI-Diff**, and **InterDreamer**. Please consider citing the following if you find this component useful:

```bibtex
@inproceedings{xu2024interdreamer,
  title = {{InterDreamer}: Zero-Shot Text to 3D Dynamic Human-Object Interaction},
  author = {Xu, Sirui and Wang, Ziyin and Wang, Yu-Xiong and Gui, Liang-Yan},
  booktitle = {NeurIPS},
  year = {2024},
}

@inproceedings{xu2023interdiff,
  title = {{InterDiff}: Generating 3D Human-Object Interactions with Physics-Informed Diffusion},
  author = {Xu, Sirui and Li, Zhengyuan and Wang, Yu-Xiong and Gui, Liang-Yan},
  booktitle = {ICCV},
  year = {2023},
}

@article{peng2023hoi,
  title = {HOI-Diff: Text-Driven Synthesis of 3D Human-Object Interactions using Diffusion Models},
  author = {Peng, Xiaogang and Xie, Yiming and Wu, Zizhao and Jampani, Varun and Sun, Deqing and Jiang, Huaizu},
  journal = {arXiv preprint arXiv:2312.06553},
  year = {2023}
}
```

Our SMPL-X-based humanoid model is adapted from PHC. Please consider citing:

```bibtex
@inproceedings{Luo2023PerpetualHC,
  author = {Zhengyi Luo and Jinkun Cao and Alexander W. Winkler and Kris Kitani and Weipeng Xu},
  title = {Perpetual Humanoid Control for Real-time Simulated Avatars},
  booktitle = {ICCV},
  year = {2023}
}
```

## 👏 Acknowledgements and 📚 License

This repository builds upon the following excellent open-source projects:

- [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs): Contributes to the environment code
- [rl_games](https://github.com/Denys88/rl_games): Serves as the core reinforcement learning framework
- [PHC](https://github.com/ZhengyiLuo/PHC): Used for data construction  
- [PhysHOI](https://github.com/wyhuai/PhysHOI): Contributes to the environment code  
- [InterAct](https://github.com/wzyabcas/InterAct), [OMOMO](https://github.com/lijiaman/omomo_release): Core resource for dataset construction  
- [InterDiff](https://github.com/Sirui-Xu/InterDiff): Supports kinematic generation  
- [HOI-Diff](https://github.com/neu-vi/HOI-Diff): Supports kinematic generation  

This codebase is released under the [MIT License](LICENSE).  
Please note that it also relies on external libraries and datasets, each of which may be subject to their own licenses and terms of use.


## 🌟 Star History

<p align="center">
    <a href="https://star-history.com/#Sirui-Xu/InterMimic&Date" target="_blank">
        <img width="500" src="https://api.star-history.com/svg?repos=Sirui-Xu/InterMimic&type=Date" alt="Star History Chart">
    </a>
<p>
