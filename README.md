
![Cosmos Logo](assets/cosmos-logo.png)

--------------------------------------------------------------------------------
### [Website](https://www.nvidia.com/en-us/ai/cosmos/) | [HuggingFace](https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6) | [GPU-free Preview](https://build.nvidia.com/explore/discover) | [Paper](https://arxiv.org/abs/2501.03575) | [Paper Website](https://research.nvidia.com/labs/dir/cosmos1/)

[NVIDIA Cosmos](https://www.nvidia.com/cosmos/) is a developer-first world foundation model platform designed to help Physical AI developers build their Physical AI systems better and faster. Cosmos contains

1. pre-trained models, available via [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6) under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) that allows commercial use of the models for free
2. training scripts under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0), offered through [NVIDIA Nemo Framework](https://github.com/NVIDIA/NeMo) for post-training the models for various downstream Physical AI applications

Details of the platform is described in the [Cosmos paper](https://research.nvidia.com/publication/2025-01_cosmos-world-foundation-model-platform-physical-ai). Preview access is available at [build.nvidia.com](https://build.nvidia.com).



# Cosmos1 (Text2World and Image2World): GPU Poor version by **DeepBeepMeep**
*01/21/2024: Version 1.1*
- Added support for sdpa attention (this simplifies greatly installation of the application on windows)
- Added support for negative prompts
- Added support for 7B models
- Added workarounds to pytorch compilation to make it run with xformers, sage and sdpa attentions: a full generation with a 14B model with compiled sage mode takes now only 17 minutes at 1280x720 (max VRAM 15.2 GB) !

*01/15/2024: Version 1.0 First release*

https://github.com/user-attachments/assets/37db0c39-1fa2-4dd6-84c9-3493fd0c3eed

https://github.com/user-attachments/assets/8fa1a249-c324-4789-9100-9ae196da08c6

https://github.com/user-attachments/assets/8da7d14d-28b5-4902-97e4-4a0171b6725f



Cosmos1GP offers the following improvements over the original NVidia Cosmos1:
- Reduced greatly the RAM requirements and VRAM requirements
- Much faster on low end configs thanks to compilation and fast loading / unloading
- Support for 8 bits quantization ( int8 quantized models provided)
- 5 profiles in order to able to run the model at a decent speed on a low end consumer config (32 GB of RAM and 12 VRAM) and to run it at a very good speed on a high end consumer config (48 GB of RAM and 24 GB of VRAM)
- Autodownloading of the needed model files (quantized and non quantized models)
- Improved gradio interface with progression bar 
- Multiples prompts / multiple generations per prompt
- Much simpler installation for non Transformer Engine

You can use this application to generate a video based on a prompt (text2world) or a video based on a prompt and an image or another video for continuation (video2world). 

This fork of Nvidia Cosmos1 made by DeepBeepMeep is an integration of the mmpg module on the gradio_server.py.

It is an illustration on how one can set up on an existing model some fast and properly working CPU offloading with changing only a few lines of code in the core model.

For more information on how to use the mmpg module, please go to: https://github.com/deepbeepmeep/mmgp


### Installation Guide for Linux and Windows

You will find a Docker container on the Cosmos1 homepage, however you will have a hard time to make it work on Windows or Windows WSL because this container contains Nvidia Graphics drivers which are not compatible with Windows.\
It is why I suggest instead to follow these installation instructions:


```shell
# 1. Get this Repository
git clone https://github.com/deepbeepmeep/Cosmos1GP
cd Cosmos1GP

# 2. Install Python 3.10.9
conda create -n Cosmos1GP python==3.10.9 

# 3. Install pytorch 2.5.1
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124

# 4. Install pip dependencies
pip install -r requirements.txt

# 5.1 Optional: Xformers attention support 
# a - Linux / WSL
pip install xformers==0.0.29.post1

# b - Windows
pip install https://download.pytorch.org/whl/cu124/xformers-0.0.29.post1-cp310-cp310-win_amd64.whl

# 5.2 Optional: Sage attention support (30% faster, easy to install on Linux but much harder on Windows)
pip install sageattention==1.0.6 

# 6. Optional: Transformer Engine support (builtin compilation and different memory management may be consequently more efficient)
pip install transformer_engine
pip install transformer_engine_torch
pip install flash-attn==2.6.0.post1
```

Step 6 may be quite hard to complete as it requires to compile both the *transformer engine* and *flash attention*.\
If you have trouble compiling please make sure of the following (Linux / Windows WSL):\
**1) Cudnn 9.6.0 is installed:**
https://developer.download.nvidia.com/compute/cudnn/9.6.0/local_installers/cudnn-local-repo-ubuntu2204-9.6.0_1.0-1_amd64.deb

**2) The Paths to the Cuda Libraries and the C++ compilers are properly set (here below for Linux)**
```
export CUDA_HOME=/usr/local/cuda
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++ 
```

**3) The Cudnn header files are in the cuda include directory** 
```
sudo cp /usr/include/cudnn* /usr/local/cuda/include/ 
```

Be aware that compiling *flash attention* may take a couple of hours.

Please note:
-  *Sage attention* is also quite complex to install on Windows but is a 30% faster than the xformers attention at a cost a small degradation of image quality. 
-  *Pytorch Compilation* will work on Windows (without WSL) if you manage to install Triton. It is quite a complex process. Here is a link that explains how to do it : https://github.com/woct0rdho/triton-windows . For simpler installations it is recommended to use Windows with WSL.


### Run a Gradio Server on port 7860 (by default)
To run the text 2 world (video) model:
```bash
python3 gradio_server_t2w.py
```

To run the image or video 2 world (video) model:
```bash
python3 gradio_server_v2w.py
```

You will have the possibility to choose the model you need and configure a RAM / VRAM profile (see section below) by expanding the section *Video Engine Configuration* in the Web Interface.\

If by mistake you have chosen a configuration not supported by your system, you can force a profile while loading the app with the safe profile 5:  
```bash
python3 gradio_server_t2w.py --profile 5
```

To run the the application using the Nvidia Transformer Engine (if you have trouble generating 10s videos and  / or if you have improperly rendered videos, see below):
```bash
python3 gradio_server_v2w.py --use-te
```

Try to use prompts that are a few hundreds characters long as short prompts do not seem to produce great videos. You may get some assistance from a large language model. It is also unclear yet if rendering resolutions other than 1280x720 can give good results. 

Thanks to a highly optimized offloading optimizations provided by the mmgp module, the size of the model (7B or 14B) you chose will have little impact on the VRAM requirements.

However, as general rule of thumb, 7B models take two times less time than 14B models and quantized models are 15% faster.
Sample generation times on a RTX 4090 for 5s 25 steps 1280x720 with sage attention but no Pytorch compile:  
- 14B non quantized: 23 minutes
- 14B quantized: 20 minutes
- 7B non quantized: 12 minutes
- 7B quantized: 10 minutes

If you turn on pytorch compilation with *Sage attention* the 14B non quantized can be run in less than 17 minutes !

Please note that currently only the *xformers* attention works with the Video2World model, although it is not clear if it follows properly the prompt. It is why I recommend to use the Transformer Engine if you can.

### Profiles
You can choose between 5 profiles depending on your hardware:
- HighRAM_HighVRAM  (1): at least 48 GB of RAM and 24 GB of VRAM : the fastest well suited for a RTX 3090 / RTX 4090 but consumes much more VRAM, adapted for fast shorter video
- HighRAM_LowVRAM  (2): at least 48 GB of RAM and 12 GB of VRAM : a bit slower, better suited for RTX 3070/3080/4070/4080 or for RTX 3090 / RTX 4090 with large pictures batches or long videos
- LowRAM_HighVRAM  (3): at least 32 GB of RAM and 24 GB of VRAM : adapted for RTX 3090 / RTX 4090 with limited RAM  but at the cost of VRAM (shorter videos)
- LowRAM_LowVRAM  (4): at least 32 GB of RAM and 12 GB of VRAM :  if you have little VRAM or want to generate longer videos 
- VerylowRAM_LowVRAM  (5): at least 24 GB of RAM and 12 GB of VRAM : if you don't have much it won't be fast but maybe it will work

Profile 2 (High RAM) and 4 (Low RAM)are the most recommended profiles since they are versatile (support for long videos for a slight performance cost).\
However, a safe approach is to start from profile 5 (default profile) and then go down progressively to profile 4 and then to profile 2 as long as the app remains responsive or doesn't trigger any out of memory error.

### Troubleshooting
Unfortunately, as the minimum of images generated is 121 frames (5s) this requires a minimum of VRAM and configs with 12 GB of VRAM may have some issues :

1) You may try lower resolutions. However it seems to be optimized for 1280x720 and the quality seems worse with other resolutions

2) You may try to turn on compilation with Xformers in the Video Engine configuration menu, usually compilation consumes less VRAM

3) The original Nvidia Transformer Engine is more VRAM efficient than xformers or sdpa attentions. In order to use it you will need to install it first (see instructions above) and run the application with the ‘—use-te’ parameter

**Known issue** :
- Transformer Engine may be slow on some systems when ran on unquantized models (issue with original repo)

### Command line parameters for the Gradio Server
--use-te : run the server using the Transformer Engine (see installation abover)
--profile no : default (5) : no of profile between 1 and 5\
--quantize-transformer bool: (default True) : enable / disable on the fly transformer quantization\
--verbose level : default (1) : level of information between 0 and 2\
--server-port portno : default (7860) : Gradio port no\
--server-name name : default (0.0.0.0) : Gradio server name\
--open-browser : open automatically Browser when launching Gradio Server\


### Other Models for the GPU Poor
- HuanyuanVideoGP: https://github.com/deepbeepmeep/HunyuanVideoGP
One of the best open source Text to Video generator

- FluxFillGP: https://github.com/deepbeepmeep/FluxFillGP
One of the best inpainting / outpainting tools based on Flux that can run with less than 12 GB of VRAM.




### Many thanks to:
- NVidia for this great models family
- The ComfyUI team for providing a replacement code for the 'apply_rotary_embeddings' that can run without the Transformer Engine
- Hugginface for their tools (optimim-quanto, accelerate, safetensors, transformers, ...) that were helpful to build the mmgp module
