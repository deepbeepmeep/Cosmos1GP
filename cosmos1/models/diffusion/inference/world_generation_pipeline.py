# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import os
from typing import Any, Optional

import numpy as np
import torch

from cosmos1.models.common.base_world_generation_pipeline import BaseWorldGenerationPipeline
from cosmos1.models.diffusion.inference.inference_utils import (
    generate_world_from_text,
    generate_world_from_video,
    get_condition_latent,
    get_video_batch,
    load_model_by_config,
    load_network_model,
    load_tokenizer_model,
)
from cosmos1.models.diffusion.model.model_t2w import DiffusionT2WModel
from cosmos1.models.diffusion.model.model_v2w import DiffusionV2WModel
from cosmos1.models.diffusion.prompt_upsampler.text2world_prompt_upsampler_inference import (
    create_prompt_upsampler,
    run_chat_completion,
)
from cosmos1.models.diffusion.prompt_upsampler.video2world_prompt_upsampler_inference import (
    create_vlm_prompt_upsampler,
    prepare_dialog,
)
from cosmos1.models.diffusion.prompt_upsampler.video2world_prompt_upsampler_inference import (
    run_chat_completion as run_chat_completion_vlm,
)
from cosmos1.utils import log

MODEL_NAME_DICT = {
    "Cosmos-1.0-Diffusion-7B-Text2World": "Cosmos_1_0_Diffusion_Text2World_7B",
    "Cosmos-1.0-Diffusion-14B-Text2World": "Cosmos_1_0_Diffusion_Text2World_14B",
    "Cosmos-1.0-Diffusion-7B-Video2World": "Cosmos_1_0_Diffusion_Video2World_7B",
    "Cosmos-1.0-Diffusion-14B-Video2World": "Cosmos_1_0_Diffusion_Video2World_14B",
}


class DiffusionText2WorldGenerationPipeline(BaseWorldGenerationPipeline):
    def __init__(
        self,
        inference_type: str,
        checkpoint_dir: str,
        checkpoint_name: str,
        transformer_filename: str,
        text_encoder_filename: str,
        prompt_upsampler_dir: Optional[str] = None,
        enable_prompt_upsampler: bool = True,
        enable_text_guardrail: bool = True,
        enable_video_guardrail: bool = True,
        offload_network: bool = False,
        offload_tokenizer: bool = False,
        offload_text_encoder_model: bool = False,
        offload_prompt_upsampler: bool = False,
        offload_guardrail_models: bool = False,
        guidance: float = 7.0,
        num_steps: int = 35,
        height: int = 704,
        width: int = 1280,
        fps: int = 24,
        num_video_frames: int = 121,
        seed: int = 0,
    ):
        """Initialize the diffusion world generation pipeline.

        Args:
            inference_type: Type of world generation ('text2world' or 'video2world')
            checkpoint_dir: Base directory containing model checkpoints
            checkpoint_name: Name of the diffusion transformer checkpoint to use
            prompt_upsampler_dir: Directory containing prompt upsampler model weights
            enable_prompt_upsampler: Whether to use prompt upsampling
            enable_text_guardrail: Whether to enable text guardrail
            enable_video_guardrail: Whether to enable video guardrail
            offload_network: Whether to offload diffusion transformer after inference
            offload_tokenizer: Whether to offload tokenizer after inference
            offload_text_encoder_model: Whether to offload T5 model after inference
            offload_prompt_upsampler: Whether to offload prompt upsampler
            offload_guardrail_models: Whether to offload guardrail models
            guidance: Classifier-free guidance scale
            num_steps: Number of diffusion sampling steps
            height: Height of output video
            width: Width of output video
            fps: Frames per second of output video
            num_video_frames: Number of frames to generate
            seed: Random seed for sampling
        """
        assert inference_type in [
            "text2world",
            "video2world",
        ], "Invalid inference_type, must be 'text2world' or 'video2world'"

        self.model_name = MODEL_NAME_DICT[checkpoint_name]
        self.guidance = guidance
        self.num_steps = num_steps
        self.height = height
        self.width = width
        self.fps = fps
        self.num_video_frames = num_video_frames
        self.seed = seed

        super().__init__(
            inference_type=inference_type,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            transformer_filename = transformer_filename,
            text_encoder_filename = text_encoder_filename,
            enable_text_guardrail=enable_text_guardrail,
            enable_video_guardrail=enable_video_guardrail,
            offload_network=offload_network,
            offload_tokenizer=offload_tokenizer,
            offload_text_encoder_model=offload_text_encoder_model,
            offload_guardrail_models=offload_guardrail_models,
        )
        self.prompt_upsampler_dir = prompt_upsampler_dir
        self.enable_prompt_upsampler = enable_prompt_upsampler
        self.offload_prompt_upsampler = offload_prompt_upsampler

        self.prompt_upsampler = None
        if enable_prompt_upsampler and not offload_prompt_upsampler:
            self._load_prompt_upsampler_model()

    def _load_prompt_upsampler_model(self):
        self.prompt_upsampler = create_prompt_upsampler(
            checkpoint_dir=os.path.join(self.checkpoint_dir, self.prompt_upsampler_dir),
        )

    def _load_model(self):
        self.model = load_model_by_config(
            config_job_name=self.model_name,
            config_file="cosmos1/models/diffusion/config/config.py",
            model_class=DiffusionT2WModel,
        )

    def _load_network(self):
        load_network_model(self.model, f"{self.checkpoint_dir}/transformer/{self.transformer_filename}")

    def _load_tokenizer(self):
        load_tokenizer_model(self.model, f"{self.checkpoint_dir}/Cosmos-1.0-Tokenizer-CV8x8x8")

    def _offload_prompt_upsampler_model(self):
        """Move prompt enhancement model to CPU/disk.

        Offloads prompt upsampling model after processing input
        to reduce GPU memory usage.
        """
        if self.prompt_upsampler:
            del self.prompt_upsampler
            self.prompt_upsampler = None
            gc.collect()
            torch.cuda.empty_cache()

    def _run_prompt_upsampler_on_prompt(self, prompt: str) -> str:
        """Enhance the input prompt using the prompt upsampler model.

        Args:
            prompt: Raw text prompt to be enhanced

        Returns:
            str: Enhanced version of the input prompt with more descriptive details
        """
        upsampled_prompt = run_chat_completion(self.prompt_upsampler, prompt)
        log.info(f"Upsampled prompt: {upsampled_prompt}")
        return upsampled_prompt

    def _run_prompt_upsampler_on_prompt_with_offload(self, *args: Any, **kwargs: Any) -> str:
        """Enhance prompt with prompt upsampler model.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Enhanced prompt string
        """
        if self.offload_prompt_upsampler:
            self._load_prompt_upsampler_model()

        enhanced_prompt = self._run_prompt_upsampler_on_prompt(*args, **kwargs)

        if self.offload_prompt_upsampler:
            self._offload_prompt_upsampler_model()

        return enhanced_prompt

    def _run_tokenizer_decoding(self, sample: torch.Tensor) -> np.ndarray:
        """Decode latent samples to video frames using the tokenizer decoder.

        Args:
            sample: Latent tensor from diffusion model [B, C, T, H, W]

        Returns:
            np.ndarray: Decoded video frames as uint8 numpy array [T, H, W, C]
                        with values in range [0, 255]
        """
        # Decode video
        video = (1.0 + self.model.decode(sample)).clamp(0, 2) / 2  # [B, 3, T, H, W]
        video = (video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()

        return video

    def _run_model(
        self,
        embedding: torch.Tensor,
        negative_prompt_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate video latents using the diffusion model.

        Args:
            embedding: Text embedding tensor from text encoder
            negative_prompt_embedding: Optional embedding for negative prompt guidance

        Returns:
            torch.Tensor: Generated video latents before tokenizer decoding

        Note:
            The model and tokenizer are automatically offloaded after inference
            if offloading is enabled in the config.
        """
        # Get video batch and state shape
        data_batch, state_shape = get_video_batch(
            model=self.model,
            prompt_embedding=embedding,
            negative_prompt_embedding=negative_prompt_embedding,
            height=self.height,
            width=self.width,
            fps=self.fps,
            num_video_frames=self.num_video_frames,
        )

        # Generate video frames
        sample = generate_world_from_text(
            model=self.model,
            state_shape=state_shape,
            is_negative_prompt=True if negative_prompt_embedding is not None else False,
            data_batch=data_batch,
            guidance=self.guidance,
            num_steps=self.num_steps,
            seed=self.seed,
            callback = self._callback if hasattr(self, "_callback") else None
        )

        return sample

    def _run_model_with_offload(
        self, prompt_embedding: torch.Tensor, negative_prompt_embedding: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """Generate world representation with automatic model offloading.

        Wraps the core generation process with model loading/offloading logic
        to minimize GPU memory usage during inference.

        Args:
            *args: Positional arguments passed to _run_model
            **kwargs: Keyword arguments passed to _run_model

        Returns:
            np.ndarray: Generated world representation as numpy array
        """
        if self.offload_network:
            self._load_network()
        self.model.tokenizer.video_vae.cpu()

        # if self.offload_tokenizer: 
        #     self._load_tokenizer()

        sample = self._run_model(prompt_embedding, negative_prompt_embedding)
        # if True:
        #     sample = torch.load("lalala.pt", weights_only = True)
        # else:
        #     torch.save(sample, "lalala.pt")

        self._offload.unload_all()

        if self.offload_network:
            self._offload_network()

        if self.offload_tokenizer:
            self._load_tokenizer()

        self.model.tokenizer.video_vae.cuda() # manual loading / unloading as the vae is not a standard object
        sample = self._run_tokenizer_decoding(sample)
        self.model.tokenizer.video_vae.cpu()

        if self.offload_tokenizer:
            self._offload_tokenizer()
        return sample

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        word_limit_to_skip_upsampler: Optional[int] = None,
    ) -> tuple[np.ndarray, str] | None:
        """Generate video from text prompt with optional negative prompt guidance.

        Pipeline steps:
        1. Run safety checks on input prompt
        2. Enhance prompt using upsampler if enabled
        3. Run safety checks on upsampled prompt if applicable
        4. Convert prompt to embeddings
        5. Generate video frames using diffusion
        6. Run safety checks and apply face blur on generated video frames

        Args:
            prompt: Text description of desired video
            negative_prompt: Optional text to guide what not to generate
            word_limit_to_skip_upsampler: Skip prompt upsampler for better robustness if the number of words in the prompt is greater than this value
        Returns:
            tuple: (
                Generated video frames as uint8 np.ndarray [T, H, W, C],
                Final prompt used for generation (may be enhanced)
            ), or None if content fails guardrail safety checks
        """
        log.info(f"Run with prompt: {prompt}")
        log.info(f"Run with negative prompt: {negative_prompt}")
        log.info(f"Run with prompt upsampler: {self.enable_prompt_upsampler}")

        if self.enable_text_guardrail:
            log.info("Run guardrail on prompt")
            is_safe = self._run_guardrail_on_prompt_with_offload(prompt)
            if not is_safe:
                log.critical("Input text prompt is not safe")
                return None
            log.info("Pass guardrail on prompt")

        # Enhance prompt
        if self.enable_prompt_upsampler:
            word_count = len(prompt.split())
            if word_limit_to_skip_upsampler is None or word_count <= word_limit_to_skip_upsampler:
                log.info("Run prompt upsampler on prompt")
                prompt = self._run_prompt_upsampler_on_prompt_with_offload(prompt)
                if self.enable_text_guardrail:
                    log.info("Run guardrail on upsampled prompt")
                    is_safe = self._run_guardrail_on_prompt_with_offload(prompt=prompt)
                    if not is_safe:
                        log.critical("Upsampled text prompt is not safe")
                        return None
                    log.info("Pass guardrail on upsampled prompt")
            else:
                log.info(
                    f"Skip prompt upsampler for better robustness because the number of words ({word_count}) in the prompt is greater than {word_limit_to_skip_upsampler}"
                )

        log.info("Run text embedding on prompt")
        if negative_prompt:
            prompts = [prompt, negative_prompt]
        else:
            prompts = [prompt]
        prompt_embeddings, _ = self._run_text_embedding_on_prompt_with_offload(prompts)
        prompt_embedding = prompt_embeddings[0]
        negative_prompt_embedding = prompt_embeddings[1] if negative_prompt else None
        log.info("Finish text embedding on prompt")

        # Generate video
        log.info("Run generation")
        video = self._run_model_with_offload(
            prompt_embedding,
            negative_prompt_embedding=negative_prompt_embedding,
        )
        log.info("Finish generation")

        if self.enable_video_guardrail:
            log.info("Run guardrail on generated video")
            video = self._run_guardrail_on_video_with_offload(video)
            if video is None:
                log.critical("Generated video is not safe")
                return None
            log.info("Pass guardrail on generated video")

        return video, prompt


class DiffusionVideo2WorldGenerationPipeline(DiffusionText2WorldGenerationPipeline):
    def __init__(
        self,
        inference_type: str,
        checkpoint_dir: str,
        checkpoint_name: str,
        transformer_filename: str,
        text_encoder_filename: str,

        prompt_upsampler_dir: Optional[str] = None,
        enable_prompt_upsampler: bool = True,
        enable_text_guardrail: bool = True,
        enable_video_guardrail: bool = True,
        offload_network: bool = False,
        offload_tokenizer: bool = False,
        offload_text_encoder_model: bool = False,
        offload_prompt_upsampler: bool = False,
        offload_guardrail_models: bool = False,
        guidance: float = 7.0,
        num_steps: int = 35,
        height: int = 704,
        width: int = 1280,
        fps: int = 24,
        num_video_frames: int = 121,
        seed: int = 0,
        num_input_frames: int = 1,
    ):
        """Initialize diffusion world generation pipeline.

        Args:
            inference_type: Type of world generation ('text2world' or 'video2world')
            checkpoint_dir: Base directory containing model checkpoints
            checkpoint_name: Name of the diffusion transformer checkpoint to use
            prompt_upsampler_dir: Directory containing prompt upsampler model weights
            enable_prompt_upsampler: Whether to use prompt upsampling
            enable_text_guardrail: Whether to enable text guardrail
            enable_video_guardrail: Whether to enable video guardrail
            offload_network: Whether to offload diffusion transformer after inference
            offload_tokenizer: Whether to offload tokenizer after inference
            offload_text_encoder_model: Whether to offload T5 model after inference
            offload_prompt_upsampler: Whether to offload prompt upsampler
            offload_guardrail_models: Whether to offload guardrail models
            guidance: Classifier-free guidance scale
            num_steps: Number of diffusion sampling steps
            height: Height of output video
            width: Width of output video
            fps: Frames per second of output video
            num_video_frames: Number of frames to generate
            seed: Random seed for sampling
            num_input_frames: Number of latent conditions
        """
        self.num_input_frames = num_input_frames
        super().__init__(
            inference_type=inference_type,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            transformer_filename = transformer_filename,
            text_encoder_filename = text_encoder_filename,
            prompt_upsampler_dir=prompt_upsampler_dir,
            enable_prompt_upsampler=enable_prompt_upsampler,
            enable_text_guardrail=enable_text_guardrail,
            enable_video_guardrail=enable_video_guardrail,
            offload_network=offload_network,
            offload_tokenizer=offload_tokenizer,
            offload_text_encoder_model=offload_text_encoder_model,
            offload_prompt_upsampler=offload_prompt_upsampler,
            offload_guardrail_models=offload_guardrail_models,
            guidance=guidance,
            num_steps=num_steps,
            height=height,
            width=width,
            fps=fps,
            num_video_frames=num_video_frames,
            seed=seed,
        )

    def _run_prompt_upsampler_on_prompt(self, image_or_video_path: str) -> str:
        """Enhance the input prompt using visual context from the conditioning image.

        Args:
            image_or_video_path: Path to conditioning image or video used for visual context

        Returns:
            str: Enhanced prompt incorporating visual details from the image
        """
        dialog = prepare_dialog(image_or_video_path)
        upsampled_prompt = run_chat_completion_vlm(
            self.prompt_upsampler, dialog, max_gen_len=400, temperature=0.01, top_p=0.9, logprobs=False
        )
        log.info(f"Upsampled prompt: {upsampled_prompt}")
        return upsampled_prompt

    def _load_prompt_upsampler_model(self):
        self.prompt_upsampler = create_vlm_prompt_upsampler(
            checkpoint_dir=os.path.join(self.checkpoint_dir, self.prompt_upsampler_dir),
        )

    def _load_model(self):
        self.model = load_model_by_config(
            config_job_name=self.model_name,
            config_file="cosmos1/models/diffusion/config/config.py",
            model_class=DiffusionV2WModel,
        )

    def _run_model(
        self,
        embedding: torch.Tensor,
        condition_latent: torch.Tensor,
        negative_prompt_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate video frames using the diffusion model.

        Args:
            embedding: Text embedding tensor from T5 encoder
            condition_latent: Latent tensor from conditioning image or video
            negative_prompt_embedding: Optional embedding for negative prompt guidance

        Returns:
            Tensor of generated video frames

        Note:
            Model and tokenizer are automatically offloaded after inference
            if offloading is enabled.
        """
        # Get video batch and state shape
        data_batch, state_shape = get_video_batch(
            model=self.model,
            prompt_embedding=embedding,
            negative_prompt_embedding=negative_prompt_embedding,
            height=self.height,
            width=self.width,
            fps=self.fps,
            num_video_frames=self.num_video_frames,
        )

        # Generate video frames
        video = generate_world_from_video(
            model=self.model,
            state_shape=self.model.state_shape,
            is_negative_prompt=True,
            data_batch=data_batch,
            guidance=self.guidance,
            num_steps=self.num_steps,
            seed=self.seed,
            condition_latent=condition_latent,
            num_input_frames=self.num_input_frames,
            callback = self._callback if hasattr(self, "_callback") else None
        )

        return video

    def _run_tokenizer_encoding(self, image_or_video_path: str) -> torch.Tensor:
        """
        Encode image to latent space

        Args:
            image_or_video_path: Path to conditioning image

        Returns:
            torch.Tensor: Latent tensor from tokenizer encoding
        """
        condition_latent = get_condition_latent(
            model=self.model,
            input_image_or_video_path=image_or_video_path,
            num_input_frames=self.num_input_frames,
            state_shape=self.model.state_shape,
            max_frames = self.max_frames  if  hasattr(self,"max_frames") else -1
        )

        return condition_latent

    def _run_model_with_offload(
        self,
        prompt_embedding: torch.Tensor,
        image_or_video_path: str,
        negative_prompt_embedding: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Generate world representation with automatic model offloading.

        Wraps the core generation process with model loading/offloading logic
        to minimize GPU memory usage during inference.

        Args:
            prompt_embedding: Text embedding tensor from T5 encoder
            image_or_video_path: Path to conditioning image or video
            negative_prompt_embedding: Optional embedding for negative prompt guidance

        Returns:
            np.ndarray: Generated world representation as numpy array
        """
        if self.offload_tokenizer:
            self._load_tokenizer()


        self.model.tokenizer.video_vae.cuda() # manual loading / unloading as the vae is not a standard object
        condition_latent = self._run_tokenizer_encoding(image_or_video_path)
        self.model.tokenizer.video_vae.cpu()
        torch.cuda.empty_cache()

        if self.offload_network:
            self._load_network()

        sample = self._run_model(prompt_embedding, condition_latent, negative_prompt_embedding)

        if self.offload_network:
            self._offload_network()

        self.model.tokenizer.video_vae.cuda() # manual loading / unloading as the vae is not a standard object
        sample = self._run_tokenizer_decoding(sample)
        self.model.tokenizer.video_vae.cpu()
        torch.cuda.empty_cache()

        if self.offload_tokenizer:
            self._offload_tokenizer()

        return sample

    def generate(
        self,
        prompt: str,
        image_or_video_path: str,
        negative_prompt: Optional[str] = None,
    ) -> tuple[np.ndarray, str] | None:
        """Generate video from text prompt and optional image.

        Pipeline steps:
        1. Run safety checks on input prompt
        2. Enhance prompt using upsampler if enabled
        3. Run safety checks on upsampled prompt if applicable
        4. Convert prompt to embeddings
        5. Generate video frames using diffusion
        6. Run safety checks and apply face blur on generated video frames

        Args:
            prompt: Text description of desired video
            image_or_video_path: Path to conditioning image or video
            negative_prompt: Optional text to guide what not to generate

        Returns:
            tuple: (
                Generated video frames as uint8 np.ndarray [T, H, W, C],
                Final prompt used for generation (may be enhanced)
            ), or None if content fails guardrail safety checks
        """
        log.info(f"Run with prompt: {prompt}")
        log.info(f"Run with image or video path: {image_or_video_path}")
        log.info(f"Run with negative prompt: {negative_prompt}")
        log.info(f"Run with prompt upsampler: {self.enable_prompt_upsampler}")

        if self.enable_text_guardrail and not self.enable_prompt_upsampler:
            log.info("Run guardrail on prompt")
            is_safe = self._run_guardrail_on_prompt_with_offload(prompt)
            if not is_safe:
                log.critical("Input text prompt is not safe")
                return None
            log.info("Pass guardrail on prompt")

        # Enhance prompt
        if self.enable_prompt_upsampler:
            log.info("Run prompt upsampler on image or video, input prompt is not used")
            prompt = self._run_prompt_upsampler_on_prompt_with_offload(image_or_video_path=image_or_video_path)
            if self.enable_text_guardrail:
                log.info("Run guardrail on upsampled prompt")
                is_safe = self._run_guardrail_on_prompt_with_offload(prompt)
                if not is_safe:
                    log.critical("Upsampled text prompt is not safe")
                    return None
                log.info("Pass guardrail on upsampled prompt")

        log.info("Run text embedding on prompt")
        if negative_prompt:
            prompts = [prompt, negative_prompt]
        else:
            prompts = [prompt]
        prompt_embeddings, _ = self._run_text_embedding_on_prompt_with_offload(prompts)
        prompt_embedding = prompt_embeddings[0]
        negative_prompt_embedding = prompt_embeddings[1] if negative_prompt else None
        log.info("Finish text embedding on prompt")

        # Generate video
        log.info("Run generation")
        video = self._run_model_with_offload(
            prompt_embedding,
            negative_prompt_embedding=negative_prompt_embedding,
            image_or_video_path=image_or_video_path,
        )
        log.info("Finish generation")

        if self.enable_video_guardrail:
            log.info("Run guardrail on generated video")
            video = self._run_guardrail_on_video_with_offload(video)
            if video is None:
                log.critical("Generated video is not safe")
                return None
            log.info("Pass guardrail on generated video")

        return video, prompt
