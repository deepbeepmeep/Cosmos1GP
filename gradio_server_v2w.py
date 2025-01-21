import os
import time
try:
    import triton
except ImportError:
    pass
from pathlib import Path

import gradio as gr
import random
import json
import argparse
import os
import sys
from mmgp import offload, profile_type
use_te=  "--use-te" in sys.argv[1:]
if use_te:
    offload.shared_state["TE"] = True
# orch.enable_grad(False)
from cosmos1.models.diffusion.inference.inference_utils import add_common_arguments, validate_args
from cosmos1.models.diffusion.inference.world_generation_pipeline import DiffusionText2WorldGenerationPipeline, DiffusionVideo2WorldGenerationPipeline
from cosmos1.utils import log, misc
from cosmos1.utils.io import read_prompts_from_file, save_video


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text to world generation demo script")
    # Add common arguments
    add_common_arguments(parser)

    # Add text2world specific arguments
    parser.add_argument(
        "--diffusion_transformer_dir",
        type=str,
        default="Cosmos-1.0-Diffusion-7B-Text2World",
        help="DiT model weights directory name relative to checkpoint_dir",
        choices=[
            "Cosmos-1.0-Diffusion-7B-Text2World",
            "Cosmos-1.0-Diffusion-14B-Text2World",
        ],
    )
    parser.add_argument(
        "--prompt_upsampler_dir",
        type=str,
        default="Cosmos-1.0-Prompt-Upsampler-12B-Text2World",
        help="Prompt upsampler weights directory relative to checkpoint_dir",
    )

    parser.add_argument(
        "--word_limit_to_skip_upsampler",
        type=int,
        default=250,
        help="Skip prompt upsampler for better robustness if the number of words in the prompt is greater than this value",
    )

    parser.add_argument(
        "--quantize-transformer",
        action="store_true",
        help="On the fly 'transformer' quantization"
    )



    # parser.add_argument(
    #     "--lora-weight",
    #     nargs='+',
    #     default=[],
    #     help="List of Lora Path to Weights"
    # )

    # parser.add_argument(
    #     "--lora-multiplier",
    #     nargs='+',
    #     default=[],
    #     help="List of Lora multipliers"
    # )

    parser.add_argument(
        "--profile",
        type=str,
        default=-1,
        help="Profile No"
    )

    parser.add_argument(
        "--use-te",
        action="store_true",
        help="use Transformer Engine"
    )

    parser.add_argument(
        "--video2world",
        action="store_true",
        help="use the Video2World model"
    )

    parser.add_argument(
        "--verbose",
        type=str,
        default=1,
        help="Verbose level"
    )

    parser.add_argument(
        "--server-port",
        type=str,
        default=0,
        help="Server port"
    )

    parser.add_argument(
        "--server-name",
        type=str,
        default="",
        help="Server name"
    )

    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="open browser"
    )


    return parser.parse_args()


args = parse_arguments()
cfg = args

misc.set_random_seed(cfg.seed)
video2world = args.video2world
video2world = True
if video2world:
    inference_type = "video2world"
else:
    inference_type = "text2world"

validate_args(cfg, inference_type)

# lora_weight =args.lora_weight
# lora_multiplier = [float(i) for i in args.lora_multiplier ]
force_profile_no = int(args.profile)
verbose_level = int(args.verbose)
quantizeTransformer = args.quantize_transformer

text_encoder_choices= ["T5XXLEncoder_11B.safetensors", "T5XXLEncoder_11B_quanto_int8.safetensors"]
if video2world:
    transformer_choices = ["cosmo1_14B_video2world.safetensors", "cosmo1_14B_video2world_quanto_int8.safetensors", "cosmo1_7B_video2world.safetensors", "cosmo1_7B_video2world_quanto_int8.safetensors"]
    server_config_filename = "gradio_config_v2w.json"
else:
    transformer_choices = ["cosmo1_14B_text2world.safetensors", "cosmo1_14B_text2world_quanto_int8.safetensors", "cosmo1_7B_text2world.safetensors", "cosmo1_7B_text2world_quanto_int8.safetensors"]
    server_config_filename = "gradio_config_t2w.json"


if not Path(server_config_filename).is_file():
    server_config = {"attention_mode" : "sdpa",  
                    "transformer_filename": transformer_choices[1], 
                    "text_encoder_filename" : text_encoder_choices[1],
                    "compile" : "",
                    "profile" : profile_type.LowRAM_LowVRAM }

    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config))
else:
    with open(server_config_filename, "r", encoding="utf-8") as reader:
        text = reader.read()
    server_config = json.loads(text)

transformer_filename = server_config["transformer_filename"]
text_encoder_filename = server_config["text_encoder_filename"]
attention_mode = server_config["attention_mode"]
attention_mode = "sdpa" if attention_mode == "basic" else attention_mode
profile =  force_profile_no if force_profile_no >=0 else server_config["profile"]
compile = server_config.get("compile", "")

#### test Zone 
#attention_mode="sage"
#attention_mode="xformers"
#attention_mode = "sdpa"
#quantizeTransformer = True
#compile = "transformer"

if use_te:
    compile =""

if compile != None and len(compile) and not quantizeTransformer:
    offload.shared_state["patch_compiler"]= True

offload.shared_state["patch_compiler"]= True 
offload.shared_state["attention_mode"] = attention_mode
def download_models(transformer_filename, text_encoder_filename):
    from huggingface_hub import hf_hub_download, snapshot_download    
    repoId = "DeepBeepMeep/Cosmos1GP" 
    sourceFolderList = ["Cosmos-1.0-Tokenizer-CV8x8x8", "text_encoder",  "transformer" ]
    fileList = [ [], ["config.json", "spiece.model", "tokenizer.json", "config.json", text_encoder_filename] , [transformer_filename] ]
    targetRoot = "checkpoints/" 
    for sourceFolder, files in zip(sourceFolderList,fileList ): 
        if len(files)==0:
            if not Path(targetRoot + sourceFolder).exists():
                snapshot_download(repo_id=repoId,  allow_patterns=sourceFolder +"/*", local_dir= targetRoot)
        else:
            for onefile in files:      
                if not os.path.isfile(targetRoot + sourceFolder + "/" + onefile ):          
                    hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = targetRoot, subfolder=sourceFolder)


download_models(transformer_filename, text_encoder_filename) 


offload.default_verboseLevel = verbose_level



cfg.num_video_frames = 121
cfg.prompt = "a man walking"
cfg.checkpoint_dir = "checkpoints"
cfg.num_steps = 20
cfg.video_save_folder = "outputs"
cfg.disable_prompt_upsampler = True


# text_encoder_filename = "T5XXLEncoder_11B.safetensors" 
# text_encoder_filename = "T5XXLEncoder_11B_quanto_int8.safetensors" 
if video2world:
    # transformer_filename = "cosmo1_14B_video2world.safetensors"
    # transformer_filename = "cosmo1_14B_video2world_quanto_int8.safetensors"
    # transformer_filename = "cosmo1_7B_video2world.safetensors"
    # transformer_filename = "cosmo1_7B_video2world_quanto_int8.safetensors"

    if "14B" in transformer_filename:
        cfg.diffusion_transformer_dir = "Cosmos-1.0-Diffusion-14B-Video2World" 
        checkpoint_name = "Cosmos-1.0-Diffusion-14B-Video2World"
    else:
        cfg.diffusion_transformer_dir = "Cosmos-1.0-Diffusion-7B-Video2World" 
        checkpoint_name = "Cosmos-1.0-Diffusion-7B-Video2World"
else:
    # transformer_filename = "cosmo1_14B_text2world.safetensors"
    # transformer_filename = "cosmo1_14B_text2world_quanto_int8.safetensors"
    # transformer_filename = "cosmo1_7B_text2world.safetensors"
    # transformer_filename = "cosmo1_7B_text2world_quanto_int8.safetensors"

    if "14B" in transformer_filename:
        cfg.diffusion_transformer_dir = "Cosmos-1.0-Diffusion-14B-Text2World" 
        checkpoint_name = "Cosmos-1.0-Diffusion-14B-Text2World"
    else:
        cfg.diffusion_transformer_dir = "Cosmos-1.0-Diffusion-7B-Text2World" 
        checkpoint_name = "Cosmos-1.0-Diffusion-7B-Text2World"

if video2world:
    # Initialize video2world generation model pipeline
    pipeline = DiffusionVideo2WorldGenerationPipeline(
        inference_type=inference_type,
        checkpoint_dir=cfg.checkpoint_dir,
        checkpoint_name=checkpoint_name,
        transformer_filename = transformer_filename,
        text_encoder_filename = text_encoder_filename,
        prompt_upsampler_dir=cfg.prompt_upsampler_dir,
        enable_prompt_upsampler=not cfg.disable_prompt_upsampler,
        offload_network=cfg.offload_diffusion_transformer,
        offload_tokenizer=cfg.offload_tokenizer,
        offload_text_encoder_model=cfg.offload_text_encoder_model,
        offload_prompt_upsampler=cfg.offload_prompt_upsampler,
        offload_guardrail_models=cfg.offload_guardrail_models,
        guidance=cfg.guidance,
        num_steps=cfg.num_steps,
        height=cfg.height,
        width=cfg.width,
        fps=cfg.fps,
        num_video_frames=cfg.num_video_frames,
        seed=cfg.seed,
        num_input_frames=1,
        enable_text_guardrail = False,
        enable_video_guardrail = False,
    )
    pipe = { "transformer" : pipeline.model.model, "text_encoder" : pipeline.text_encoder.text_encoder} #, "vae" : pipeline.model.tokenizer.video_vae 

else:
    # Initialize text2world generation model pipeline
    pipeline = DiffusionText2WorldGenerationPipeline(
        inference_type=inference_type,
        checkpoint_dir=cfg.checkpoint_dir,
        checkpoint_name=checkpoint_name,
        transformer_filename = transformer_filename,
        text_encoder_filename = text_encoder_filename,
        prompt_upsampler_dir=cfg.prompt_upsampler_dir,
        enable_prompt_upsampler=not cfg.disable_prompt_upsampler,
        offload_network=cfg.offload_diffusion_transformer,
        offload_tokenizer=cfg.offload_tokenizer,
        offload_text_encoder_model=cfg.offload_text_encoder_model,
        offload_prompt_upsampler=cfg.offload_prompt_upsampler,
        offload_guardrail_models=cfg.offload_guardrail_models,
        guidance=cfg.guidance,
        num_steps=cfg.num_steps,
        height=cfg.height,
        width=cfg.width,
        fps=cfg.fps,
        num_video_frames=cfg.num_video_frames,
        seed=cfg.seed,
        enable_text_guardrail = False,
        enable_video_guardrail = False,
    )
    #.model
    pipe = { "transformer" : pipeline.model.model, "text_encoder" : pipeline.text_encoder.text_encoder} #, "vae" : pipeline.model.tokenizer.video_vae 
 
pipeline._offload = offload.profile(pipe,  profile_no= profile,  compile  = compile ,  quantizeTransformer = quantizeTransformer) #, 



def apply_changes(
                    transformer_choice,
                    text_encoder_choice,
                    attention_choice,
                    compile_choice,
                    profile_choice,
):
    server_config = {"attention_mode" : attention_choice,  
                     "transformer_filename": transformer_choices[transformer_choice], 
                     "text_encoder_filename" : text_encoder_choices[text_encoder_choice],
                     "compile" : compile_choice,
                     "profile" : profile_choice }

    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config))

    return "<h1>New Config file created. Please restart the Gradio Server</h1>"



def build_callback(state, pipe, progress, status, num_inference_steps):
    def callback(**kwargs):
        step_idx =kwargs["i_th"]
        step_idx += 1         
        if state.get("abort", False):
            pipe._interrupt = True
            raise Exception("aborting")
            # status_msg = status + " - Aborting"    
        elif step_idx  == num_inference_steps:
            status_msg = status + " - VAE Decoding"    
        else:
            status_msg = status + " - Denoising"   

        progress( (step_idx , num_inference_steps) , status_msg  ,  num_inference_steps)
            
    return callback

def abort_generation(state):
    if "in_progress" in state:
        state["abort"] = True
        pipeline._interrupt = True
        return gr.Button(interactive=  False)
    else:
        return gr.Button(interactive=  True)

def refresh_gallery(state):
    file_list = state["file_list"]      
    return file_list
        
def finalize_gallery(state):
    choice = 0
    if "in_progress" in state:
        del state["in_progress"]
        choice = state.get("selected",0)
    time.sleep(0.2)
    return gr.Gallery(selected_index=choice), gr.Button(interactive=  True)

def select_video(state , event_data: gr.EventData):
    data=  event_data._data
    if data!=None:
        state["selected"] = data.get("index",0)
    return 


def generate_video(
    prompt,
    neg_prompt,
    resolution,
    video_length,
    seed,
    num_inference_steps,
    # guidance_scale,
    # flow_shift,
    embedded_guidance_scale,
    repeat_generation,
    # tea_cache,
    image_to_continue,
    video_to_continue,
    max_frames,
    state,
    progress=gr.Progress() #track_tqdm= True

):
    seed = None if seed == -1 else seed
    width, height = resolution.split("x")
    width, height = int(width), int(height)

    pipeline._interrupt = False

    if "abort" in state:
        del state["abort"]
    state["in_progress"] = True
    state["selected"] = 0
 
    os.makedirs(cfg.video_save_folder, exist_ok=True)

    import random
    if seed == None or seed <0:
        seed = random.randint(0, 999999999)

    # misc.set_random_seed(seed)

    file_list = []
    state["file_list"] = file_list    
    
    # prompt = "A first-person POV video capturing highway driving in late afternoon. Through the windshield, show a straight 3-lane highway stretching toward the horizon with sparse traffic. The sun is low but not setting, casting long shadows across the asphalt from guard rails and passing vehicles. The road surface should have a subtle sheen from the angled sunlight. The motion should have gentle, natural camera movement that mimics a driver's perspective. Highway signs and exit markers pass by periodically. Other vehicles occasionally pass by at varying speeds, some closer, some further away in adjacent lanes. The landscape beyond the highway should show rolling terrain with a mix of trees and open spaces. Include natural highway sounds: the steady rhythm of tires on pavement, the whoosh of passing vehicles, and the gentle rush of wind. The view should convey smooth, steady forward motion at highway speed, with subtle variations as the road surface changes and the car adjusts to small elevation changes. The overall mood should be peaceful and meditative, capturing that distinctive feeling of cruising down an open highway in good weather."
    prompts = prompt.replace("\r", "").split("\n")
    video_no = 0
    total_video =  repeat_generation * len(prompts)

    from PIL import Image
    import numpy as np
    import tempfile
    temp_filename = None
    if video2world:

        if image_to_continue is not None:
            PIL_image = Image.fromarray(np.uint8(image_to_continue)).convert('RGB')
            with tempfile.NamedTemporaryFile("w+b", delete = False, suffix=".png") as fp: 
                PIL_image.save(fp, format="png")
                fp.close()

            input_image_or_video_path = fp.name
            temp_filename = input_image_or_video_path
            pipeline.num_input_frames = 1 
            pipeline.max_frames = 1 

        elif video_to_continue != None and len(video_to_continue) >0 :
            input_image_or_video_path = video_to_continue
            pipeline.num_input_frames = max_frames
            pipeline.max_frames = max_frames
        else:
            return
    else:
        input_image_or_video_path = None

    start_time = time.time()
    for current_prompt in prompts:
        for _ in range(repeat_generation):
            video_no += 1
            status = f"Video {video_no}/{total_video}"
            progress(0, desc=status + " - Encoding Prompt" )   
            
            callback = build_callback(state, pipeline, progress, status, num_inference_steps)
            # Generate video

            pipeline.num_video_frames = video_length
            pipeline._callback = callback
            pipeline.seed = seed
            pipeline.height = height
            pipeline.width = width
            pipeline.num_steps = num_inference_steps-1
            pipeline.guidance = embedded_guidance_scale
            # pipeline.fps = fps
            # pipeline.num_video_frames = num_video_frames

            # if True:
            try:
                if video2world:
                    generated_output = pipeline.generate(prompt = current_prompt, negative_prompt=  neg_prompt, image_or_video_path = input_image_or_video_path)
                else:
                    generated_output = pipeline.generate(prompt = current_prompt, negative_prompt=  neg_prompt)

            except Exception as e:
                s = str(e)
                if "abort" in s:
                     generated_output = None
                else:
                    raise

            from datetime import datetime
            
            if generated_output == None:
                end_time = time.time()
                yield f"Abortion Succesful. Total Generation Time: {end_time-start_time:.1f}s"
            else:
                video, prompt = generated_output

                time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss")
                base_path = f"{time_flag}_seed{seed}_{video_no}_{current_prompt[:100].replace('/','').strip()}".replace(':',' ').replace('\\',' ')

                video_save_path = os.path.join(cfg.video_save_folder, f"{base_path}.mp4")
                prompt_save_path = os.path.join(cfg.video_save_folder, f"{base_path}.txt")

                # Save video
                save_video(
                    video=video,
                    fps=cfg.fps,
                    H=cfg.height,
                    W=cfg.width,
                    video_save_quality=5,
                    video_save_path=video_save_path,
                )

                # Save prompt to text file alongside video
                with open(prompt_save_path, "wb") as f:
                    f.write(prompt.encode("utf-8"))

                log.info(f"Saved video to {video_save_path}")
                log.info(f"Saved prompt to {prompt_save_path}")

                # print(f"New video saved to Path: "+ video_save_path)
                file_list.append(video_save_path)
                if video_no < total_video:
                    yield  status
                else:
                    end_time = time.time()
                    yield f"Total Generation Time: {end_time-start_time:.1f}s"
            seed += 1

    if temp_filename!= None and  os.path.isfile(temp_filename):
        os.remove(temp_filename)


def create_demo():
    
    with gr.Blocks() as demo:
        version = "Video2World" if video2world else "Text2World"
        gr.Markdown(f"<div align=center><H1>Cosmos1<SUP>GP</SUP> - {version}</H3></div>")
        gr.Markdown("*Original model by **NVidia**, GPU Poor version by **DeepBeepMeep**. Now this great world generator can run smoothly on a 24 GB rig.*")
        if use_te:
            gr.Markdown("*NVidia Transformer Engine currently used*")

        gr.Markdown("Please be aware that the number of frames should be a multiple of 121 frames (5s)")
        gr.Markdown("In order to produce 242 frames (10s), it is likely that you will need to use the Transformer Engine (*gradio_server.py --use-te*)")
        gr.Markdown("In the worst case, one step should not take more than 2 minutes. If it is the case you may be running out of RAM / VRAM. Try to generate fewer images / lower res / a less demanding profile.")

        with gr.Accordion("Video Engine Configuration", open = False):
            gr.Markdown("For the changes to be effective you will need to restart the gradio_server")

            with gr.Column():
                index = transformer_choices.index(transformer_filename) 
                index = 0 if index ==0 else index

                gr.Markdown("Note that currently due somme issue in the original Cosmos repo 16 bits (non quantized) models may be veryslow in the Transformer Engine")
                transformer_choice = gr.Dropdown(
                    choices=[
                        ("Cosmos1 14B 16 bits - the best quality but takes the most time", 0),
                        ("Cosmos1 14B quantized to 8 bits - the default engine but quantized", 1),
                        ("Cosmos1 7B 16 bits - lower model", 2),
                        ("Cosmos1 7B quantized to 8 bits - worst quality", 3),
                    ],
                    value= index,
                    label="Transformer"
                 )
                index = text_encoder_choices.index(text_encoder_filename)
                index = 0 if index ==0 else index

                gr.Markdown("Note that even if you choose a 16 bits T5 model below, depending on the profile it may be automatically quantized to 8 bits on the fly")
                text_encoder_choice = gr.Dropdown(
                    choices=[
                        ("T5 XXL Encoder 16 bits - unquantized text encoder, better quality uses more RAM", 0),
                        ("T5 XXL Encoder quantized to 8 bits - quantized text encoder, worse quality but uses less RAM", 1),
                    ],
                    value= index,
                    label="Text Encoder"
                 )

                gr.Markdown("**When using the Transformer Engine, the attention mode is forced to Flash attention**")
                attention_choice = gr.Dropdown(
                    choices=[
                        ("Sdpa: default torch attention, compatible with Windows but requires more memory", "sdpa"),
                        ("Xformers: good quality - requires additional install", "xformers"),
                        ("Sage: 30% faster but slightly worse quality - requires additional install (usually complex to set up on Windows without WSL)", "sage"),
                    ],
                    value= attention_mode,
                    label="Attention Type"
                 )
                gr.Markdown("**Compilation is not relevant with the Transformer Engine since it has its own inductor / compiler**")
                gr.Markdown("Beware: when restarting the server or changing a resolution or video duration, the first step of generation for a duration / resolution may last a few minutes due to recompilation")
                compile_choice = gr.Dropdown(
                    choices=[
                        ("ON: works only on Linux / WSL", "transformer"),
                        ("OFF: no other choice if you have Windows without using WSL", "" ),
                    ],
                    value= compile,
                    label="Compile Transformer (up to 50% faster and 30% more frames (less VRAM consumption) but requires triton support"
                 )                
                profile_choice = gr.Dropdown(
                    choices=[
                ("HighRAM_HighVRAM, profile 1: at least 48 GB of RAM and 24 GB of VRAM, the fastest for shorter videos a RTX 3090 / RTX 4090", 1),
                ("HighRAM_LowVRAM, profile 2 (Recommended): at least 48 GB of RAM and 12 GB of VRAM, the most versatile profile with high RAM, better suited for RTX 3070/3080/4070/4080 or for RTX 3090 / RTX 4090 with large pictures batches or long videos", 2),
                ("LowRAM_HighVRAM, profile 3: at least 32 GB of RAM and 24 GB of VRAM, adapted for RTX 3090 / RTX 4090 with limited RAM for good speed short video",3),
                ("LowRAM_LowVRAM, profile 4 (Default): at least 32 GB of RAM and 12 GB of VRAM, if you have little VRAM or want to generate longer videos",4),
                ("VerylowRAM_LowVRAM, profile 5: (Fail safe): at least 16 GB of RAM and 10 GB of VRAM, if you don't have much it won't be fast but maybe it will work",5)
                    ],
                    value= profile,
                    label="Profile"
                 )

                msg = gr.Markdown()            
                apply_btn  = gr.Button("Apply Changes")

                apply_btn.click(
                        fn=apply_changes,
                        inputs=[
                            transformer_choice,
                            text_encoder_choice,
                            attention_choice,
                            compile_choice,                            
                            profile_choice,
                        ],
                        outputs= msg
                    )

        with gr.Row():
            with gr.Column():
                video_to_continue = gr.Video(label= "Video to continue", visible= video2world)
                image_to_continue = gr.Image(label= "or Image as a starting point for a new video", visible= video2world)
                prompt = gr.Textbox(label="Prompt", value="Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field.")
                with gr.Row():
                    resolution = gr.Dropdown(
                        choices=[
                            # 720p
                            ("1280x720 (16:9, 720p)", "1280x720"),
                            ("720x1280 (9:16, 720p)", "720x1280"), 
                            ("1104x832 (4:3, 720p)", "1104x832"),
                            ("832x1104 (3:4, 720p)", "832x1104"),
                            ("960x960 (1:1, 720p)", "960x960"),
                            # 540p
                            ("960x544 (16:9, 540p)", "960x544"),
                            ("848x480 (16:9, 540p)", "848x480"),
                            ("544x960 (9:16, 540p)", "544x960"),
                            ("832x624 (4:3, 540p)", "832x624"), 
                            ("624x832 (3:4, 540p)", "624x832"),
                            ("720x720 (1:1, 540p)", "720x720"),
                        ],
                        value="1280x720",
                        label="Resolution"
                    )

                # video_length = gr.Slider(5, 193, value=97, step=4, label="Number of frames (24 = 1s)")

                    video_length = gr.Dropdown(
                        label="Video Length",
                        choices=[
                            ("5s (121f)", 121),
                            ("10s (242f)", 242),
                        ],
                        value=121,
                    )
                num_inference_steps = gr.Slider(2, 100, value=25, step=1, label="Number of Inference Steps")
                max_frames = gr.Slider(1, 100, value=9, step=1, label="Number of input frames to use for Video2World prediction", visible=video2world)

                show_advanced = gr.Checkbox(label="Show Advanced Options", value=False)
                with gr.Row(visible=False) as advanced_row:
                    with gr.Column():
                        neg_prompt = gr.Textbox(label="Negative Prompt", lines = 3, value=cfg.negative_prompt)
                        seed = gr.Number(value=-1, label="Seed (-1 for random)")
                        # guidance_scale = gr.Slider(1.0, 20.0, value=1.0, step=0.5, label="Guidance Scale")
                        # flow_shift = gr.Slider(0.0, 25.0, value=7.0, step=0.1, label="Flow Shift") 
                        embedded_guidance_scale = gr.Slider(1.0, 20.0, value=7.0, step=0.5, label="Embedded Guidance Scale")

                        repeat_generation = gr.Slider(1, 25.0, value=1.0, step=1, label="Number of Generated Video per prompt") 
                        # tea_cache_setting = gr.Dropdown(
                        #     choices=[
                        #         ("Disabled", 0),
                        #         ("Fast (x1.6 speed up)", 0.1), 
                        #         ("Faster (x2.1 speed up)", 0.15), 
                        #     ],
                        #     value=0,
                        #     label="Tea Cache acceleration (the faster the acceleration the higher the degradation of the quality of the video)"
                        # )

                show_advanced.change(fn=lambda x: gr.Row(visible=x), inputs=[show_advanced], outputs=[advanced_row])
            
            with gr.Column():
                gen_status = gr.Text(label="Status", interactive= False) 
                output = gr.Gallery(
                        label="Generated videos", show_label=False, elem_id="gallery"
                    , columns=[3], rows=[1], object_fit="contain", height="auto", selected_index=0, interactive= False)
                state = gr.State({})
                generate_btn = gr.Button("Generate")
                abort_btn = gr.Button("Abort")

        gen_status.change(refresh_gallery, inputs = [state], outputs = output )

        abort_btn.click(abort_generation,state,abort_btn )
        output.select(select_video, state, None )

        generate_btn.click(
            fn=generate_video,
            inputs=[
                prompt,
                neg_prompt,
                resolution,
                video_length,
                seed,
                num_inference_steps,
                # guidance_scale,
                # flow_shift,
                embedded_guidance_scale,
                repeat_generation,
                # tea_cache_setting,
                image_to_continue,
                video_to_continue,
                max_frames,
                state
            ],
            outputs= [gen_status] #,state 

        ).then( 
            finalize_gallery,
            [state], 
            [output , abort_btn]
        )
    
    return demo


if __name__ == "__main__":
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    server_port = int(args.server_port)

    if server_port == 0:
        server_port = int(os.getenv("SERVER_PORT", "7860"))

    server_name = args.server_name
    server_name = "localhost"
    if len(server_name) == 0:
        server_name = os.getenv("SERVER_NAME", "0.0.0.0")

        
    demo = create_demo()
    if args.open_browser:
        import webbrowser 
        if server_name.startswith("http"):
            url = server_name 
        else:
            url = "http://" + server_name 
        webbrowser.open(url + ":" + str(server_port), new = 0, autoraise = True)

    demo.launch(server_name=server_name, server_port=server_port)


 