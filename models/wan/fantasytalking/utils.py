# Copyright Alibaba Inc. All Rights Reserved.

import librosa
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from core.io.media import VideoSaveConfig, write_video


def resize_image_by_longest_edge(image_path, target_size):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    scale = target_size / max(width, height)
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.LANCZOS)


def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    target = Path(save_path)
    container = target.suffix.lstrip(".") or "mp4"
    extra_params: Dict[str, Any] = {}
    if quality is not None:
        extra_params["quality"] = quality
    if ffmpeg_params:
        extra_params["ffmpeg_params"] = list(ffmpeg_params)

    config = VideoSaveConfig(
        fps=int(round(fps)),
        container=container,
        extra_params=extra_params,
    )
    result = write_video(frames, str(target), config=config)
    if result is None:
        raise RuntimeError(f"Failed to write video to {target}")


def get_audio_features(wav2vec, audio_processor, audio_path, fps, start_frame, num_frames):
    sr = 16000
    audio_input, sample_rate = librosa.load(audio_path, sr=sr)  # 采样率为 16kHz    start_time = 0
    if start_frame  < 0:
        pad = int(abs(start_frame)/ fps * sr)
        audio_input = np.concatenate([np.zeros(pad), audio_input])
        end_frame = num_frames
    else:
        end_frame = start_frame + num_frames

    start_time = start_frame / fps
    end_time = end_frame / fps

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    try:
        audio_segment = audio_input[start_sample:end_sample]
    except:
        audio_segment = audio_input

    input_values = audio_processor(
        audio_segment, sampling_rate=sample_rate, return_tensors="pt"
    ).input_values.to("cuda")

    with torch.no_grad():
        fea = wav2vec(input_values).last_hidden_state

    return fea
