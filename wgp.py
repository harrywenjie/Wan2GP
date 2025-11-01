import os, sys
# # os.environ.pop("TORCH_LOGS", None)  # make sure no env var is suppressing/overriding
# os.environ["TORCH_LOGS"]= "recompiles"
import torch._logging as tlog
# tlog.set_logs(recompiles=True, guards=True, graph_breaks=True)    
p = os.path.dirname(os.path.abspath(__file__))
if p not in sys.path:
    sys.path.insert(0, p)
import asyncio
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from shared.asyncio_utils import silence_proactor_connection_reset
silence_proactor_connection_reset()
import time
import threading
import argparse
import warnings
warnings.filterwarnings('ignore', message='Failed to find.*', module='triton')
from mmgp import offload, safetensors2, profile_type 
try:
    import triton
except ImportError:
    pass
from pathlib import Path
from datetime import datetime
import random
import json
import numpy as np
import importlib
from shared.bootstrap_defaults import DEFAULT_BOOTSTRAP_VALUES, GENERATION_FALLBACKS
from core.progress import clear_status, format_duration, get_latest_status, merge_status_context, update_status
from shared.utils.notifications import (
    notify_debug,
    notify_info,
    notify_warning,
    notify_error,
    get_notifications_logger,
)
from shared.utils import notification_sound
from shared.utils.loras_mutipliers import preparse_loras_multipliers, parse_loras_multipliers
from shared.utils.utils import convert_tensor_to_image, save_image, get_video_info, get_file_creation_date, convert_image_to_video, calculate_new_dimensions, convert_image_to_tensor, calculate_dimensions_and_resize_image, rescale_and_crop, get_video_frame, resize_and_remove_background, rgb_bw_to_rgba_mask
from shared.utils.utils import calculate_new_dimensions, get_outpainting_frame_location, get_outpainting_full_area_dimensions
from shared.utils.utils import has_video_file_extension, has_image_file_extension, has_audio_file_extension
from shared.utils.audio_video import (
    extract_audio_tracks,
    combine_video_with_audio_tracks,
    combine_and_concatenate_video_with_audio_tracks,
    cleanup_temp_audio_files,
    save_video as _save_video,
    save_image as _save_image,
)
from shared.notifications import create_legacy_notifier
from shared.utils.audio_video import read_image_metadata
from core.preview import prepare_preview_inputs
from core.task_inputs import SaveInputsPayload, SaveInputsRequest, TaskInputManager
from shared.utils.audio_metadata import read_audio_metadata
from shared.match_archi import match_nvidia_architecture
from shared.attention import get_attention_modes, get_supported_attention_modes
from shared.utils.utils import truncate_for_filesystem, sanitize_file_name, process_images_multithread, get_default_workers
from shared.utils.process_locks import acquire_GPU_ressources, get_gen_info, release_GPU_ressources, gen_lock
from core.io import get_available_filename
from core.io.media import MetadataSaveConfig, build_metadata_config, write_metadata_bundle
from huggingface_hub import hf_hub_download, snapshot_download
from shared.utils import files_locator as fl 
import torch
import gc
import traceback
import math 
import typing
from typing import Dict, Optional
import inspect
from shared.utils import prompt_parser
import base64
from PIL import Image


def save_video(*args, **kwargs):
    """
    Wrapper around ``shared.utils.audio_video.save_video`` that ensures logging flows
    through the notifications logger configured by the CLI.
    """

    if kwargs.get("logger") is None:
        kwargs["logger"] = get_notifications_logger()
    return _save_video(*args, **kwargs)


def save_image(*args, **kwargs):
    """
    Wrapper around ``shared.utils.audio_video.save_image`` with logger injection.
    """

    if kwargs.get("logger") is None:
        kwargs["logger"] = get_notifications_logger()
    return _save_image(*args, **kwargs)


metadata_configs: Dict[str, MetadataSaveConfig] = {}
metadata_choice: Optional[str] = None


def _resolve_metadata_config(
    kind: str,
    *,
    embedded_images: Optional[Dict[str, str]] = None,
) -> MetadataSaveConfig:
    configs = metadata_configs if isinstance(metadata_configs, dict) and metadata_configs else None
    config = build_metadata_config(kind, templates=configs)
    if kind == "video" and embedded_images:
        video_options = dict(config.extra_options.get("video", {}))
        video_options["source_images"] = embedded_images
        extra_options = dict(config.extra_options)
        extra_options["video"] = video_options
        config.extra_options = extra_options
    return config
import zipfile
import atexit
import shutil
import glob
import cv2
import html
from transformers.utils import logging
logging.set_verbosity_error
from tqdm import tqdm
import requests
# import torch._dynamo as dynamo
# dynamo.config.recompile_limit = 2000   # default is 256
# dynamo.config.accumulated_recompile_limit = 2000  # or whatever limit you want
from cli.queue_utils import (
    clear_queue_action as cli_clear_queue_action,
    generate_queue_summary as cli_generate_queue_summary,
    update_queue_data as cli_update_queue_data,
    get_preview_images,
    update_task_thumbnails,
)

target_mmgp_version = "3.6.7"
WanGP_version = "9.21"
settings_version = 2.39
max_source_video_frames = 3000
prompt_enhancer_image_caption_model, prompt_enhancer_image_caption_processor, prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer = None, None, None, None
image_names_list = ["image_start", "image_end", "image_refs"]

class GenerationError(RuntimeError):
    """Raised when a generation request cannot be satisfied."""

def _clone_default_value(value):
    if isinstance(value, list):
        return [_clone_default_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _clone_default_value(v) for k, v in value.items()}
    return value

def assemble_generation_params(params=None, *, model_type=None, state=None, include_model_defaults=True):
    """
    Merge CLI/UI overrides with canonical defaults so downstream generation paths
    receive a fully-populated parameter dictionary.
    """
    params = {} if params is None else params.copy()
    resolved_model_type = model_type or params.get("model_type")
    if resolved_model_type is None and state is not None:
        resolved_model_type = state.get("model_type")
    if resolved_model_type is None:
        raise ValueError("assemble_generation_params requires a model_type.")

    assembled = {}
    for key, value in GENERATION_FALLBACKS.items():
        assembled[key] = _clone_default_value(value)

    if include_model_defaults:
        model_defaults = get_default_settings(resolved_model_type)
        for key, value in model_defaults.items():
            assembled[key] = _clone_default_value(value)

    for key, value in params.items():
        assembled[key] = value

    assembled["model_type"] = resolved_model_type

    if state is not None:
        assembled.setdefault("state", state)
        assembled.setdefault("model_filename", state.get("model_filename"))

    if assembled.get("model_filename") is None:
        assembled["model_filename"] = get_model_filename(
            resolved_model_type,
            transformer_quantization,
            transformer_dtype_policy,
        )

    return assembled

from importlib.metadata import version
mmgp_version = version("mmgp")
if mmgp_version != target_mmgp_version:
    print(f"Incorrect version of mmgp ({mmgp_version}), version {target_mmgp_version} is needed. Please upgrade with the command 'pip install -r requirements.txt'")
    exit()
lock = threading.Lock()
current_task_id = None
task_id = 0
unique_id = 0
unique_id_lock = threading.Lock()
offloadobj = enhancer_offloadobj = wan_model = None
reload_needed = True
_task_inputs_manager: TaskInputManager | None = None

def set_wgp_global(variable_name: str, new_value: any) -> str:
    if variable_name not in globals():
        error_msg = f"Plugin tried to modify a non-existent global: '{variable_name}'."
        print(f"ERROR: {error_msg}")
        notify_warning(error_msg)
        return f"Error: Global variable '{variable_name}' does not exist."

    try:
        globals()[variable_name] = new_value
    except Exception as e:
        error_msg = f"Error while setting global '{variable_name}': {e}"
        print(f"ERROR: {error_msg}")
        return error_msg

def clear_gen_cache():
    if "_cache" in offload.shared_state:
        del offload.shared_state["_cache"]

def release_model():
    global wan_model, offloadobj, reload_needed
    wan_model = None    
    clear_gen_cache()
    offload.shared_state
    if offloadobj is not None:
        offloadobj.release()
        offloadobj = None
        torch.cuda.empty_cache()
        gc.collect()
        try:
            torch._C._host_emptyCache()
        except:
            pass
        reload_needed = True
    else:
        gc.collect()

def get_unique_id():
    global unique_id  
    with unique_id_lock:
        unique_id += 1
    return str(time.time()+unique_id)

def download_ffmpeg():
    if os.name != 'nt': return
    exes = ['ffmpeg.exe', 'ffprobe.exe', 'ffplay.exe']
    if all(os.path.exists(e) for e in exes): return
    api_url = 'https://api.github.com/repos/GyanD/codexffmpeg/releases/latest'
    r = requests.get(api_url, headers={'Accept': 'application/vnd.github+json'})
    assets = r.json().get('assets', [])
    zip_asset = next((a for a in assets if 'essentials_build.zip' in a['name']), None)
    if not zip_asset: return
    zip_url = zip_asset['browser_download_url']
    zip_name = zip_asset['name']
    with requests.get(zip_url, stream=True) as resp:
        total = int(resp.headers.get('Content-Length', 0))
        with open(zip_name, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    with zipfile.ZipFile(zip_name) as z:
        for f in z.namelist():
            if f.endswith(tuple(exes)) and '/bin/' in f:
                z.extract(f)
                os.rename(f, os.path.basename(f))
    os.remove(zip_name)

def format_generation_time(seconds):
    """Format generation time showing raw seconds with human-readable time in parentheses when over 60s"""
    raw_seconds = f"{int(seconds)}s"
    
    if seconds < 60:
        return raw_seconds
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        human_readable = f"{hours}h {minutes}m {secs}s"
    else:
        human_readable = f"{minutes}m {secs}s"
    
    return f"{raw_seconds} ({human_readable})"

def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()

def compute_sliding_window_no(current_video_length, sliding_window_size, discard_last_frames, reuse_frames):
    left_after_first_window = current_video_length - sliding_window_size + discard_last_frames
    return 1 + math.ceil(left_after_first_window / (sliding_window_size - discard_last_frames - reuse_frames))

def add_video_task(**inputs):
    global task_id
    state = inputs["state"]
    gen = get_gen_info(state)
    queue = gen["queue"]
    task_id += 1
    current_task_id = task_id

    preview = get_preview_images(inputs)
    preview_payload = preview.build_payload()
    plugin_data = inputs.pop('plugin_data', {})
    
    queue_entry = {
        "id": current_task_id,
        "params": inputs.copy(),
        "plugin_data": plugin_data,
        "repeats": inputs.get("repeat_generation",1),
        "length": inputs.get("video_length",0) or 0, 
        "steps": inputs.get("num_inference_steps",0) or 0,
        "prompt": inputs.get("prompt", ""),
        "start_image_data": preview.start_data,
        "end_image_data": preview.end_data,
    }
    queue_entry.update(preview_payload)
    queue.append(queue_entry)

def _interrupt_active_model():
    global wan_model
    if wan_model is not None and hasattr(wan_model, "_interrupt"):
        wan_model._interrupt = True


def clear_queue_action(state):
    """
    Legacy wrapper preserved for compatibility with scripts that still import
    `wgp.clear_queue_action`. Internally delegates to the CLI helper so the
    queue clearing behaviour stays in sync with the headless controller.
    """

    result = cli_clear_queue_action(
        state,
        lock=lock,
        interrupt_callback=_interrupt_active_model,
    )
    return {k: result[k] for k in result if k in {"queue_summary", "queue_length"}}


def generate_queue_summary(queue):
    return cli_generate_queue_summary(queue)


def update_queue_data(queue):
    result = cli_update_queue_data(queue)
    return {k: result[k] for k in result if k in {"queue_summary", "queue_length"}}


def _load_server_config():
    global server_config_filename
    server_config_filename = "wgp_config.json"

    if not os.path.isdir("settings"):
        os.mkdir("settings")
    if os.path.isfile("t2v_settings.json"):
        for f in glob.glob(os.path.join(".", "*_settings.json*")):
            target_file = os.path.join("settings", Path(f).parts[-1])
            shutil.move(f, target_file)

    if not os.path.isfile(server_config_filename) and os.path.isfile("gradio_config.json"):
        shutil.move("gradio_config.json", server_config_filename)

    if not Path(server_config_filename).is_file():
        server_config = {
            "attention_mode": "auto",
            "transformer_types": [],
            "transformer_quantization": "int8",
            "text_encoder_quantization": "int8",
            "save_path": "outputs",
            "image_save_path": "outputs",
            "compile": "",
            "metadata_type": "metadata",
            "boost": 1,
            "clear_file_list": 5,
            "vae_config": 0,
            "profile": profile_type.LowRAM_LowVRAM,
            "preload_model_policy": [],
            "checkpoints_paths": fl.default_checkpoints_paths,
            "model_hierarchy_type": 1,
            "check_loras": False,
            "save_masks": False,
            "save_quantized": False,
            "save_speakers": False,
            "verbose_level": 1,
        }

        with open(server_config_filename, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(server_config))
    else:
        with open(server_config_filename, "r", encoding="utf-8") as reader:
            text = reader.read()
        server_config = json.loads(text)

    return server_config


def _build_default_args(server_config):
    defaults = dict(DEFAULT_BOOTSTRAP_VALUES)
    defaults["advanced"] = bool(server_config.get("last_advanced_choice", defaults["advanced"]))
    defaults["compile"] = bool(server_config.get("compile", defaults["compile"]))
    defaults["check_loras"] = bool(server_config.get("check_loras", defaults["check_loras"]))
    defaults["gpu"] = server_config.get("default_gpu_device") or defaults["gpu"]
    defaults["lora_dir"] = server_config.get("wan_loras_dir") or defaults["lora_dir"]
    defaults["lora_dir_i2v"] = server_config.get("wan_i2v_loras_dir") or defaults["lora_dir_i2v"]
    defaults["lora_dir_ltxv"] = server_config.get("ltxv_loras_dir") or defaults["lora_dir_ltxv"]
    defaults["lora_dir_flux"] = server_config.get("flux_loras_dir") or defaults["lora_dir_flux"]
    defaults["lora_dir_qwen"] = server_config.get("qwen_loras_dir") or defaults["lora_dir_qwen"]
    defaults["lora_dir_tts"] = server_config.get("tts_loras_dir") or defaults["lora_dir_tts"]

    perc_reserved = server_config.get("perc_reserved_mem_max")
    if perc_reserved is not None:
        try:
            defaults["perc_reserved_mem_max"] = float(perc_reserved)
        except (TypeError, ValueError):
            defaults["perc_reserved_mem_max"] = DEFAULT_BOOTSTRAP_VALUES["perc_reserved_mem_max"]

    preload_mb = server_config.get("preload_mb")
    if preload_mb is not None:
        try:
            defaults["preload"] = int(preload_mb)
        except (TypeError, ValueError):
            defaults["preload"] = DEFAULT_BOOTSTRAP_VALUES["preload"]
    else:
        defaults["preload"] = DEFAULT_BOOTSTRAP_VALUES["preload"]

    profile_value = server_config.get("profile")
    if profile_value is not None:
        try:
            defaults["profile"] = int(profile_value)
        except (TypeError, ValueError):
            defaults["profile"] = DEFAULT_BOOTSTRAP_VALUES["profile"]
    else:
        defaults["profile"] = DEFAULT_BOOTSTRAP_VALUES["profile"]

    defaults["settings"] = server_config.get("settings_path") or defaults["settings"]
    defaults["save_masks"] = bool(server_config.get("save_masks", defaults["save_masks"]))
    defaults["save_quantized"] = bool(server_config.get("save_quantized", defaults["save_quantized"]))
    defaults["save_speakers"] = bool(server_config.get("save_speakers", defaults["save_speakers"]))
    raw_verbose = server_config.get("verbose_level", defaults["verbose"])
    try:
        defaults["verbose"] = int(raw_verbose)
    except (TypeError, ValueError):
        defaults["verbose"] = DEFAULT_BOOTSTRAP_VALUES["verbose"]
    vram_coeff = server_config.get("vram_safety_coefficient", defaults["vram_safety_coefficient"])
    try:
        defaults["vram_safety_coefficient"] = float(vram_coeff)
    except (TypeError, ValueError):
        defaults["vram_safety_coefficient"] = DEFAULT_BOOTSTRAP_VALUES["vram_safety_coefficient"]
    return argparse.Namespace(**defaults)


_RUNTIME_INITIALIZED = False

def get_lora_dir(model_type):
    model_family = get_model_family(model_type)
    base_model_type = get_base_model_type(model_type)
    i2v = test_class_i2v(model_type) and not  base_model_type in ["i2v_2_2", "i2v_2_2_multitalk"]
    if model_family == "wan":
        lora_dir =args.lora_dir
        if i2v and len(lora_dir)==0:
            lora_dir =args.lora_dir_i2v
        if len(lora_dir) > 0:
            return lora_dir
        root_lora_dir = "loras_i2v" if i2v else "loras"

        if  "1.3B" in model_type :
            lora_dir_1_3B = os.path.join(root_lora_dir, "1.3B")
            if os.path.isdir(lora_dir_1_3B ):
                return lora_dir_1_3B
        elif base_model_type in ["ti2v_2_2", "ovi"]:
            lora_dir_5B = os.path.join(root_lora_dir, "5B")
            if os.path.isdir(lora_dir_5B ):
                return lora_dir_5B
        else:
            lora_dir_14B = os.path.join(root_lora_dir, "14B")
            if os.path.isdir(lora_dir_14B ):
                return lora_dir_14B
        return root_lora_dir    
    elif model_family == "ltxv":
            return args.lora_dir_ltxv
    elif model_family == "flux":
            return args.lora_dir_flux
    elif model_family =="qwen":
            return args.lora_dir_qwen
    elif model_family =="tts":
            return args.lora_dir_tts
    else:
        raise Exception("loras unknown")


attention_modes_installed = get_attention_modes()
attention_modes_supported = get_supported_attention_modes()

args = None
gpu_major = 0
gpu_minor = 0
bfloat16_supported = True
processing_device = "cuda"
lock_ui_attention = False
lock_ui_transformer = False
lock_ui_compile = False
force_profile_no = -1
verbose_level = 1
check_loras = False
server_config_filename = "wgp_config.json"
server_config = {}
checkpoints_paths = []
three_levels_hierarchy = True
transformer_types = []
transformer_type = "t2v"
advanced = False
last_resolution = None
transformer_quantization = "int8"
transformer_dtype_policy = ""
text_encoder_quantization = "int8"
attention_mode = "auto"
default_profile = profile_type.LowRAM_LowVRAM
loaded_profile = -1
compile = ""
boost = 1
vae_config = 0
save_path = os.path.join(os.getcwd(), "outputs")
image_save_path = os.path.join(os.getcwd(), "outputs")
preload_model_policy = []
lora_preselected_preset = ""
lora_preset_model = "t2v"


def _apply_runtime_args(runtime_args, preloaded_config=None):
    global args, gpu_major, gpu_minor, bfloat16_supported, processing_device
    global lock_ui_attention, lock_ui_transformer, lock_ui_compile
    global force_profile_no, verbose_level, check_loras
    global server_config, checkpoints_paths, three_levels_hierarchy
    global transformer_types, transformer_type, advanced, last_resolution
    global transformer_quantization, transformer_dtype_policy, text_encoder_quantization
    global attention_mode, default_profile, loaded_profile, compile, boost, vae_config
    global reload_needed, save_path, image_save_path, preload_model_policy
    global lora_preselected_preset, lora_preset_model

    args = runtime_args
    args.flow_reverse = True

    gpu_target = getattr(args, "gpu", "")
    processing_device = gpu_target if len(gpu_target) > 0 else "cuda"

    try:
        gpu_major, gpu_minor = torch.cuda.get_device_capability(gpu_target or None)
    except Exception:
        gpu_major, gpu_minor = (0, 0)

    if gpu_major < 8:
        print("Switching to FP16 models when possible as GPU architecture doesn't support optimed BF16 Kernels")
        bfloat16_supported = False
    else:
        bfloat16_supported = True

    lock_ui_attention = False
    lock_ui_transformer = False
    lock_ui_compile = False

    try:
        force_profile_no = int(getattr(args, "profile", -1))
    except (TypeError, ValueError):
        force_profile_no = -1

    try:
        verbose_level = int(getattr(args, "verbose", 1))
    except (TypeError, ValueError):
        verbose_level = 1

    check_loras = bool(getattr(args, "check_loras", False))

    if preloaded_config is None:
        server_config = _load_server_config()
    else:
        server_config = preloaded_config

    src_move = [
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14-bf16.safetensors",
        "models_t5_umt5-xxl-enc-bf16.safetensors",
        "models_t5_umt5-xxl-enc-quanto_int8.safetensors",
    ]
    tgt_move = ["xlm-roberta-large", "umt5-xxl", "umt5-xxl"]
    for src, tgt in zip(src_move, tgt_move):
        src_path = fl.locate_file(src, error_if_none=False)
        tgt_path = fl.get_download_location(tgt)
        if src_path is not None:
            try:
                if os.path.isfile(tgt_path):
                    os.remove(src_path)
                else:
                    os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
                    shutil.move(src_path, tgt_path)
            except Exception:
                pass

    checkpoints_paths = server_config.get("checkpoints_paths")
    if checkpoints_paths is None:
        checkpoints_paths = server_config["checkpoints_paths"] = fl.default_checkpoints_paths
    fl.set_checkpoints_paths(checkpoints_paths)
    three_levels_hierarchy = server_config.get("model_hierarchy_type", 1) == 1

    for path in [
        "wan2.1_Vace_1.3B_preview_bf16.safetensors",
        "sky_reels2_diffusion_forcing_1.3B_bf16.safetensors",
        "sky_reels2_diffusion_forcing_720p_14B_bf16.safetensors",
        "sky_reels2_diffusion_forcing_720p_14B_quanto_int8.safetensors",
        "sky_reels2_diffusion_forcing_720p_14B_quanto_fp16_int8.safetensors",
        "wan2.1_image2video_480p_14B_bf16.safetensors",
        "wan2.1_image2video_480p_14B_quanto_int8.safetensors",
        "wan2.1_image2video_720p_14B_quanto_int8.safetensors",
        "wan2.1_image2video_720p_14B_quanto_fp16_int8.safetensors",
        "wan2.1_image2video_720p_14B_bf16.safetensors",
        "wan2.1_text2video_14B_bf16.safetensors",
        "wan2.1_text2video_14B_quanto_int8.safetensors",
        "wan2.1_Vace_14B_mbf16.safetensors",
        "wan2.1_Vace_14B_quanto_mbf16_int8.safetensors",
        "wan2.1_FLF2V_720p_14B_quanto_int8.safetensors",
        "wan2.1_FLF2V_720p_14B_bf16.safetensors",
        "wan2.1_FLF2V_720p_14B_fp16.safetensors",
        "wan2.1_Vace_1.3B_mbf16.safetensors",
        "wan2.1_text2video_1.3B_bf16.safetensors",
        "ltxv_0.9.7_13B_dev_bf16.safetensors",
    ]:
        located = fl.locate_file(path, error_if_none=False)
        if located is not None:
            print(
                f"Removing old version of model '{path}'. A new version of this model will be downloaded next time you use it."
            )
            os.remove(located)

    for file_path, expected_size in [(fl.locate_file("Florence2/modeling_florence2.py", error_if_none=False), 127287)]:
        try:
            if os.path.isfile(file_path) and os.path.getsize(file_path) == expected_size:
                print(
                    f"Removing old version of model '{file_path}'. A new version of this model will be downloaded next time you use it."
                )
                os.remove(file_path)
        except Exception:
            pass

    lora_preselected_preset = getattr(args, "lora_preset", "")

    transformer_quantization = server_config.get("transformer_quantization", "int8")
    transformer_dtype_policy = server_config.get("transformer_dtype_policy", "")
    if getattr(args, "fp16", False):
        transformer_dtype_policy = "fp16"
    if getattr(args, "bf16", False):
        transformer_dtype_policy = "bf16"

    text_encoder_quantization = server_config.get("text_encoder_quantization", "int8")
    attention_mode = server_config.get("attention_mode", "auto")
    attention_override = getattr(args, "attention", "")
    if attention_override:
        if attention_override in ["auto", "sdpa", "sage", "sage2", "flash", "xformers"]:
            attention_mode = attention_override
            lock_ui_attention = True
        else:
            raise Exception(f"Unknown attention mode '{attention_override}'")

    default_profile = force_profile_no if force_profile_no >= 0 else server_config["profile"]
    loaded_profile = -1
    compile = server_config.get("compile", "")
    boost = server_config.get("boost", 1)
    vae_config = server_config.get("vae_config", 0)
    vae_override = getattr(args, "vae_config", "")
    if vae_override:
        vae_config = int(vae_override)

    save_path = server_config.get("save_path", os.path.join(os.getcwd(), "outputs"))
    image_save_path = server_config.get("image_save_path", os.path.join(os.getcwd(), "outputs"))
    if "video_output_codec" not in server_config:
        server_config["video_output_codec"] = "libx264_8"
    if "video_container" not in server_config:
        server_config["video_container"] = "mp4"
    if "embed_source_images" not in server_config:
        server_config["embed_source_images"] = False
    if "image_output_codec" not in server_config:
        server_config["image_output_codec"] = "jpeg_95"

    preload_model_policy = server_config.get("preload_model_policy", [])

    transformer_types = server_config.get("transformer_types", [])
    new_transformer_types = []
    for model_type in transformer_types:
        if get_model_def(model_type) is None:
            print(
                f"Model '{model_type}' is missing. Either install it in the finetune folder or remove this model from key 'transformer_types' in wgp_config.json"
            )
        else:
            new_transformer_types.append(model_type)
    transformer_types = new_transformer_types

    transformer_type = server_config.get("last_model_type")
    advanced = server_config.get("last_advanced_choice", False)
    last_resolution = server_config.get("last_resolution_choice", None)
    if getattr(args, "advanced", False):
        advanced = True

    if transformer_type is not None and transformer_type not in model_types and transformer_type not in models_def:
        transformer_type = None
    if transformer_type is None:
        transformer_type = transformer_types[0] if transformer_types else "t2v"

    if getattr(args, "t2v_14B", False) or getattr(args, "t2v", False):
        transformer_type = "t2v"
    if getattr(args, "i2v_14B", False) or getattr(args, "i2v", False):
        transformer_type = "i2v"
    if getattr(args, "t2v_1_3B", False):
        transformer_type = "t2v_1.3B"
    if getattr(args, "i2v_1_3B", False):
        transformer_type = "fun_inp_1.3B"
    if getattr(args, "vace_1_3B", False):
        transformer_type = "vace_1.3B"

    lora_preset_model = transformer_type

    if getattr(args, "compile", False):
        compile = "transformer"
        lock_ui_compile = True

    reload_needed = False


def initialize_runtime(namespace=None, overrides=None, force=False):
    global _RUNTIME_INITIALIZED
    if _RUNTIME_INITIALIZED and not force:
        return args

    preloaded_config = None
    if namespace is None:
        preloaded_config = _load_server_config()
        runtime_args = _build_default_args(preloaded_config)
    else:
        runtime_args = namespace

    if overrides:
        for key, value in overrides.items():
            setattr(runtime_args, key, value)

    _apply_runtime_args(runtime_args, preloaded_config=preloaded_config)
    _RUNTIME_INITIALIZED = True
    return args


def ensure_runtime_initialized():
    return initialize_runtime()


models_def = {}
family_handlers = ["models.wan.wan_handler", "models.wan.ovi_handler", "models.wan.df_handler", "models.ltx_video.ltxv_handler", "models.flux.flux_handler", "models.qwen.qwen_handler", "models.chatterbox.chatterbox_handler"]

model_signatures = {"t2v": "text2video_14B", "t2v_1.3B" : "text2video_1.3B",   "fun_inp_1.3B" : "Fun_InP_1.3B",  "fun_inp" :  "Fun_InP_14B", 
                    "i2v" : "image2video_480p", "i2v_720p" : "image2video_720p" , "vace_1.3B" : "Vace_1.3B", "vace_14B": "Vace_14B", "recam_1.3B": "recammaster_1.3B", 
                    "sky_df_1.3B" : "sky_reels2_diffusion_forcing_1.3B", "sky_df_14B" : "sky_reels2_diffusion_forcing_14B", 
                    "sky_df_720p_14B" : "sky_reels2_diffusion_forcing_720p_14B",
                    "phantom_1.3B" : "phantom_1.3B", "phantom_14B" : "phantom_14B", "ltxv_13B" : "ltxv_0.9.7_13B_dev", "ltxv_13B_distilled" : "ltxv_0.9.7_13B_distilled"  }


def map_family_handlers(family_handlers):
    base_types_handlers, families_infos, models_eqv_map, models_comp_map = {}, {"unknown": (100, "Unknown")}, {}, {}
    for path in family_handlers:
        handler = importlib.import_module(path).family_handler
        for model_type in handler.query_supported_types():
            if model_type in base_types_handlers:
                prev = base_types_handlers[model_type].__name__
                raise Exception(f"Model type {model_type} supported by {prev} and {handler.__name__}")
            base_types_handlers[model_type] = handler
        families_infos.update(handler.query_family_infos())
        eq_map, comp_map = handler.query_family_maps()
        models_eqv_map.update(eq_map); models_comp_map.update(comp_map)
    return base_types_handlers, families_infos, models_eqv_map, models_comp_map

model_types_handlers, families_infos,  models_eqv_map, models_comp_map = map_family_handlers(family_handlers)

def get_base_model_type(model_type):
    model_def = get_model_def(model_type)
    if model_def == None:
        return model_type if model_type in model_types_handlers else None 
        # return model_type
    else:
        return model_def["architecture"]

def get_parent_model_type(model_type):
    base_model_type =  get_base_model_type(model_type)
    if base_model_type is None: return None
    model_def = get_model_def(base_model_type)
    return model_def.get("parent_model_type", base_model_type)
"vace_14B"
def get_model_handler(model_type):
    base_model_type = get_base_model_type(model_type)
    if base_model_type is None:
        raise Exception(f"Unknown model type {model_type}")
    model_handler = model_types_handlers.get(base_model_type, None)
    if model_handler is None:
        raise Exception(f"No model handler found for base model type {base_model_type}")
    return model_handler

def are_model_types_compatible(imported_model_type, current_model_type):
    imported_base_model_type = get_base_model_type(imported_model_type)
    curent_base_model_type = get_base_model_type(current_model_type)
    if imported_base_model_type == curent_base_model_type:
        return True

    if imported_base_model_type in models_eqv_map:
        imported_base_model_type = models_eqv_map[imported_base_model_type]

    comp_list=  models_comp_map.get(imported_base_model_type, None)
    if comp_list == None: return False
    return curent_base_model_type in comp_list 

def get_model_def(model_type):
    return models_def.get(model_type, None )



def get_model_type(model_filename):
    for model_type, signature in model_signatures.items():
        if signature in model_filename:
            return model_type
    return None
    # raise Exception("Unknown model:" + model_filename)

def get_model_family(model_type, for_ui = False):
    base_model_type = get_base_model_type(model_type)
    if base_model_type is None:
        return "unknown"
    
    if for_ui : 
        model_def = get_model_def(model_type)
        model_family = model_def.get("group", None)
        if model_family is not None and model_family in families_infos:
            return model_family
    handler = model_types_handlers.get(base_model_type, None)
    if handler is None: 
        return "unknown"
    return handler.query_model_family()

def test_class_i2v(model_type):    
    model_def = get_model_def(model_type)
    return model_def.get("i2v_class", False)

def test_vace_module(model_type):
    model_def = get_model_def(model_type)
    return model_def.get("vace_class", False)

def test_class_t2v(model_type):
    model_def = get_model_def(model_type)
    return model_def.get("t2v_class", False)

def test_any_sliding_window(model_type):
    model_def = get_model_def(model_type)
    return model_def.get("sliding_window", False)

def get_model_min_frames_and_step(model_type):
    mode_def = get_model_def(model_type)
    frames_minimum = mode_def.get("frames_minimum", 5)
    frames_steps = mode_def.get("frames_steps", 4)
    latent_size = mode_def.get("latent_size", frames_steps)
    return frames_minimum, frames_steps, latent_size 
    
def get_model_fps(model_type):
    mode_def = get_model_def(model_type)
    fps= mode_def.get("fps", 16)
    return fps

def get_computed_fps(force_fps, base_model_type , video_guide, video_source ):
    if force_fps == "auto":
        if video_source != None:
            fps,  _, _, _ = get_video_info(video_source)
        elif video_guide != None:
            fps,  _, _, _ = get_video_info(video_guide)
        else:
            fps = get_model_fps(base_model_type)
    elif force_fps == "control" and video_guide != None:
        fps,  _, _, _ = get_video_info(video_guide)
    elif force_fps == "source" and video_source != None:
        fps,  _, _, _ = get_video_info(video_source)
    elif len(force_fps) > 0 and is_integer(force_fps) :
        fps = int(force_fps)
    else:
        fps = get_model_fps(base_model_type)
    return fps

def get_model_name(model_type, description_container = [""]):
    model_def = get_model_def(model_type)
    if model_def == None: 
        return f"Unknown model {model_type}"
    model_name = model_def["name"]
    description = model_def["description"]
    description_container[0] = description
    return model_name

def get_model_record(model_name):
    return f"WanGP v{WanGP_version} by DeepBeepMeep - " +  model_name

def get_model_recursive_prop(model_type, prop = "URLs", sub_prop_name = None, return_list = True,  stack= []):
    model_def = models_def.get(model_type, None)
    if model_def != None: 
        prop_value = model_def.get(prop, None)
        if prop_value == None:
            return []
        if sub_prop_name is not None:
            if sub_prop_name == "_list":
                if not isinstance(prop_value,list) or len(prop_value) != 1:
                    raise Exception(f"Sub property value for property {prop} of model type {model_type} should be a list of size 1")
                prop_value = prop_value[0]
            else:
                if not isinstance(prop_value,dict) and not sub_prop_name in prop_value:
                    raise Exception(f"Invalid sub property value {sub_prop_name} for property {prop} of model type {model_type}")
                prop_value = prop_value[sub_prop_name]
        if isinstance(prop_value, str):
            if len(stack) > 10: raise Exception(f"Circular Reference in Model {prop} dependencies: {stack}")
            return get_model_recursive_prop(prop_value, prop = prop, sub_prop_name =sub_prop_name, stack = stack + [prop_value] )
        else:
            return prop_value
    else:
        if model_type in model_types:
            return [] if return_list else model_type 
        else:
            raise Exception(f"Unknown model type '{model_type}'")
        

def get_model_filename(model_type, quantization ="int8", dtype_policy = "", module_type = None, submodel_no = 1, URLs = None, stack=[]):
    if URLs is not None:
        pass
    elif module_type is not None:
        base_model_type = get_base_model_type(model_type) 
        # model_type_handler = model_types_handlers[base_model_type]
        # modules_files = model_type_handler.query_modules_files() if hasattr(model_type_handler, "query_modules_files") else {}
        if isinstance(module_type, list):
            URLs = module_type
        else:
            if "#" not in module_type:
                sub_prop_name = "_list"
            else:
                pos = module_type.rfind("#")
                sub_prop_name =  module_type[pos+1:]
                module_type = module_type[:pos]  
            URLs = get_model_recursive_prop(module_type, "modules", sub_prop_name =sub_prop_name, return_list= False)

        # choices = modules_files.get(module_type, None)
        # if choices == None: raise Exception(f"Invalid Module Id '{module_type}'")
    else:
        key_name = "URLs" if submodel_no  <= 1 else f"URLs{submodel_no}"

        model_def = models_def.get(model_type, None)
        if model_def == None: return ""
        URLs = model_def.get(key_name, [])
        if isinstance(URLs, str):
            if len(stack) > 10: raise Exception(f"Circular Reference in Model {key_name} dependencies: {stack}")
            return get_model_filename(URLs, quantization=quantization, dtype_policy=dtype_policy, submodel_no = submodel_no, stack = stack + [URLs])

    choices = URLs if isinstance(URLs, list) else [URLs]
    if len(choices) == 0:
        return ""
    if len(quantization) == 0:
        quantization = "bf16"

    model_family =  get_model_family(model_type) 
    dtype = get_transformer_dtype(model_family, dtype_policy)
    if len(choices) <= 1:
        raw_filename = choices[0]
    else:
        if quantization in ("int8", "fp8"):
            sub_choices = [ name for name in choices if quantization in os.path.basename(name) or quantization.upper() in os.path.basename(name)]
        else:
            sub_choices = [ name for name in choices if "quanto" not in os.path.basename(name)]

        if len(sub_choices) > 0:
            dtype_str = "fp16" if dtype == torch.float16 else "bf16"
            new_sub_choices = [ name for name in sub_choices if dtype_str in os.path.basename(name) or dtype_str.upper() in os.path.basename(name)]
            sub_choices = new_sub_choices if len(new_sub_choices) > 0 else sub_choices
            raw_filename = sub_choices[0]
        else:
            raw_filename = choices[0]

    return raw_filename

def get_transformer_dtype(model_family, transformer_dtype_policy):
    if not isinstance(transformer_dtype_policy, str):
        return transformer_dtype_policy
    if len(transformer_dtype_policy) == 0:
        if not bfloat16_supported:
            return torch.float16
        else:
            if model_family == "wan"and False:
                return torch.float16
            else: 
                return torch.bfloat16
        return transformer_dtype
    elif transformer_dtype_policy =="fp16":
        return torch.float16
    else:
        return torch.bfloat16

def get_settings_file_name(model_type):
    return  os.path.join(args.settings, model_type + "_settings.json")

def fix_settings(model_type, ui_defaults, min_settings_version = 0):
    if model_type is None: return

    settings_version =  max(min_settings_version, ui_defaults.get("settings_version", 0))
    model_def = get_model_def(model_type)
    base_model_type = get_base_model_type(model_type)

    prompts = ui_defaults.get("prompts", "")
    if len(prompts) > 0:
        ui_defaults["prompt"] = prompts
    image_prompt_type = ui_defaults.get("image_prompt_type", None)
    if image_prompt_type != None :
        if not isinstance(image_prompt_type, str):
            image_prompt_type = "S" if image_prompt_type  == 0 else "SE"
        if settings_version <= 2:
            image_prompt_type = image_prompt_type.replace("G","")
        ui_defaults["image_prompt_type"] = image_prompt_type

    if "lset_name" in ui_defaults: del ui_defaults["lset_name"]

    audio_prompt_type = ui_defaults.get("audio_prompt_type", None)
    if settings_version < 2.2: 
        if not base_model_type in ["vace_1.3B","vace_14B", "sky_df_1.3B", "sky_df_14B", "ltxv_13B"]:
            for p in  ["sliding_window_size", "sliding_window_overlap", "sliding_window_overlap_noise", "sliding_window_discard_last_frames"]:
                if p in ui_defaults: del ui_defaults[p]

        if audio_prompt_type == None :
            if any_audio_track(base_model_type):
                audio_prompt_type ="A"
                ui_defaults["audio_prompt_type"] = audio_prompt_type

    if settings_version < 2.35 and any_audio_track(base_model_type): 
        audio_prompt_type = audio_prompt_type or ""
        audio_prompt_type += "V"
        ui_defaults["audio_prompt_type"] = audio_prompt_type

    video_prompt_type = ui_defaults.get("video_prompt_type", "")

    if base_model_type in ["flux"] and settings_version < 2.23:
        video_prompt_type = video_prompt_type.replace("K", "").replace("I", "KI")

    remove_background_images_ref = ui_defaults.get("remove_background_images_ref", None)
    if settings_version < 2.22:
        if "I" in video_prompt_type:
            if remove_background_images_ref == 2:
                video_prompt_type = video_prompt_type.replace("I", "KI")
        if remove_background_images_ref != 0:
            remove_background_images_ref = 1
    if remove_background_images_ref is not None:
        ui_defaults["remove_background_images_ref"] = remove_background_images_ref

    ui_defaults["video_prompt_type"] = video_prompt_type

    tea_cache_setting = ui_defaults.get("tea_cache_setting", None)
    tea_cache_start_step_perc = ui_defaults.get("tea_cache_start_step_perc", None)

    if tea_cache_setting != None:
        del ui_defaults["tea_cache_setting"]
        if tea_cache_setting > 0:
            ui_defaults["skip_steps_multiplier"] = tea_cache_setting
            ui_defaults["skip_steps_cache_type"] = "tea"
        else:
            ui_defaults["skip_steps_multiplier"] = 1.75
            ui_defaults["skip_steps_cache_type"] = ""

    if tea_cache_start_step_perc != None:
        del ui_defaults["tea_cache_start_step_perc"]
        ui_defaults["skip_steps_start_step_perc"] = tea_cache_start_step_perc

    image_prompt_type = ui_defaults.get("image_prompt_type", "")
    if len(image_prompt_type) > 0:
        image_prompt_types_allowed = model_def.get("image_prompt_types_allowed","")
        image_prompt_type = filter_letters(image_prompt_type, image_prompt_types_allowed)
    ui_defaults["image_prompt_type"] = image_prompt_type

    video_prompt_type = ui_defaults.get("video_prompt_type", "")
    image_ref_choices_list = model_def.get("image_ref_choices", {}).get("choices", [])
    if model_def.get("guide_custom_choices", None) is  None:
        if len(image_ref_choices_list)==0:
            video_prompt_type = del_in_sequence(video_prompt_type, "IK")
        else:
            first_choice = image_ref_choices_list[0][1]
            if "I" in first_choice and not "I" in video_prompt_type: video_prompt_type += "I"
            if len(image_ref_choices_list)==1 and "K" in first_choice and not "K" in video_prompt_type: video_prompt_type += "K"
        ui_defaults["video_prompt_type"] = video_prompt_type

    model_handler = get_model_handler(base_model_type)
    if hasattr(model_handler, "fix_settings"):
            model_handler.fix_settings(base_model_type, settings_version, model_def, ui_defaults)


def get_default_settings(model_type):
    def get_default_prompt(i2v):
        if i2v:
            return "Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field."
        else:
            return "A large orange octopus is seen resting on the bottom of the ocean floor, blending in with the sandy and rocky terrain. Its tentacles are spread out around its body, and its eyes are closed. The octopus is unaware of a king crab that is crawling towards it from behind a rock, its claws raised and ready to attack. The crab is brown and spiny, with long legs and antennae. The scene is captured from a wide angle, showing the vastness and depth of the ocean. The water is clear and blue, with rays of sunlight filtering through. The shot is sharp and crisp, with a high dynamic range. The octopus and the crab are in focus, while the background is slightly blurred, creating a depth of field effect."
    i2v = test_class_i2v(model_type)
    defaults_filename = get_settings_file_name(model_type)
    if not Path(defaults_filename).is_file():
        model_def = get_model_def(model_type)
        base_model_type = get_base_model_type(model_type)
        ui_defaults = {
            "settings_version" : settings_version,
            "prompt": get_default_prompt(i2v),
            "resolution": "1280x720" if "720" in base_model_type else "832x480",
            "video_length": 81,
            "num_inference_steps": 30,
            "seed": -1,
            "repeat_generation": 1,
            "multi_images_gen_type": 0,        
            "guidance_scale": 5.0,
            "flow_shift": 7.0 if not "720" in base_model_type and i2v else 5.0, 
            "negative_prompt": "",
            "activated_loras": [],
            "loras_multipliers": "",
            "skip_steps_multiplier": 1.5,
            "skip_steps_start_step_perc": 20,
            "RIFLEx_setting": 0,
            "slg_switch": 0,
            "slg_layers": [9],
            "slg_start_perc": 10,
            "slg_end_perc": 90,
            "audio_prompt_type": "V",
        }
        model_handler = get_model_handler(model_type)
        model_handler.update_default_settings(base_model_type, model_def, ui_defaults)

        ui_defaults_update = model_def.get("settings", None) 
        if ui_defaults_update is not None: ui_defaults.update(ui_defaults_update)

        if len(ui_defaults.get("prompt","")) == 0:
            ui_defaults["prompt"]= get_default_prompt(i2v)

        with open(defaults_filename, "w", encoding="utf-8") as f:
            json.dump(ui_defaults, f, indent=4)
    else:
        with open(defaults_filename, "r", encoding="utf-8") as f:
            ui_defaults = json.load(f)
        fix_settings(model_type, ui_defaults)            
    
    default_seed = args.seed
    if default_seed > -1:
        ui_defaults["seed"] = default_seed
    default_number_frames = args.frames
    if default_number_frames > 0:
        ui_defaults["video_length"] = default_number_frames
    default_number_steps = args.steps
    if default_number_steps > 0:
        ui_defaults["num_inference_steps"] = default_number_steps
    return ui_defaults


def init_model_def(model_type, model_def):
    base_model_type = get_base_model_type(model_type)
    family_handler = model_types_handlers.get(base_model_type, None)
    if family_handler is None:
        raise Exception(f"Unknown model type {base_model_type}")
    default_model_def = family_handler.query_model_def(base_model_type, model_def)
    if default_model_def is None: return model_def
    default_model_def.update(model_def)
    return default_model_def


models_def_paths =  glob.glob( os.path.join("defaults", "*.json") ) + glob.glob( os.path.join("finetunes", "*.json") ) 
models_def_paths.sort()
for file_path in models_def_paths:
    model_type = os.path.basename(file_path)[:-5]
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            json_def = json.load(f)
        except Exception as e:
            raise Exception(f"Error while parsing Model Definition File '{file_path}': {str(e)}")
    model_def = json_def["model"]
    model_def["path"] = file_path
    del json_def["model"]      
    settings = json_def   
    existing_model_def = models_def.get(model_type, None) 
    if existing_model_def is not None:
        existing_settings = models_def.get("settings", None)
        if existing_settings != None:
            existing_settings.update(settings)
        existing_model_def.update(model_def)
    else:
        models_def[model_type] = model_def # partial def
        model_def= init_model_def(model_type, model_def)
        models_def[model_type] = model_def # replace with full def
        model_def["settings"] = settings

model_types = models_def.keys()
displayed_model_types= []
for model_type in model_types:
    model_def = get_model_def(model_type)
    if not model_def is None and model_def.get("visible", True): 
        displayed_model_types.append(model_type)


def save_model(model, model_type, dtype,  config_file,  submodel_no = 1,  is_module = False, filter = None, no_fp16_main_model = True, module_source_no = 1):
    model_def = get_model_def(model_type)
    # To save module and quantized modules
    # 1) set Transformer Model Quantization Type to 16 bits
    # 2) insert in def module_source : path and "model_fp16.safetensors in URLs"
    # 3) Generate (only quantized fp16 will be created)
    # 4) replace in def module_source : path and "model_bf16.safetensors in URLs"
    # 5) Generate (both bf16 and quantized bf16 will be created)
    if model_def == None: return
    if is_module:
        url_key = "modules"
        source_key = "module_source" if module_source_no <=1 else "module_source2"
    else:
        url_key = "URLs" if submodel_no <=1 else "URLs" + str(submodel_no)
        source_key = "source" if submodel_no <=1 else "source2"
    URLs= model_def.get(url_key, None)
    if URLs is None: return
    if isinstance(URLs, str):
        print("Unable to save model for a finetune that references external files")
        return
    from mmgp import offload    
    dtypestr= "bf16" if dtype == torch.bfloat16 else "fp16"
    if no_fp16_main_model: dtypestr = dtypestr.replace("fp16", "bf16")
    model_filename = None
    if is_module:
        if not isinstance(URLs,list) or len(URLs) != 1:
            print("Target Module files are missing")
            return 
        URLs= URLs[0]
    if isinstance(URLs, dict):
        url_dict_key = "URLs" if module_source_no ==1 else "URLs2"
        URLs = URLs[url_dict_key]
    for url in URLs:
        if "quanto" not in url and dtypestr in url:
            model_filename = os.path.basename(url)
            break
    if model_filename is None:
        print(f"No target filename with bf16 or fp16 in its name is mentioned in {url_key}")
        return

    finetune_file = os.path.join(os.path.dirname(model_def["path"]) , model_type + ".json")
    with open(finetune_file, 'r', encoding='utf-8') as reader:
        saved_finetune_def = json.load(reader)

    update_model_def = False
    model_filename_path = os.path.join(fl.get_download_location(), model_filename)
    quanto_dtypestr= "bf16" if dtype == torch.bfloat16 else "fp16"
    if ("m" + dtypestr) in model_filename: 
        dtypestr = "m" + dtypestr 
        quanto_dtypestr = "m" + quanto_dtypestr 
    if fl.locate_file(model_filename) is None and (not no_fp16_main_model or dtype == torch.bfloat16):
        offload.save_model(model, model_filename_path, config_file_path=config_file, filter_sd=filter)
        print(f"New model file '{model_filename}' had been created for finetune Id '{model_type}'.")
        del saved_finetune_def["model"][source_key]
        del model_def[source_key]
        print(f"The 'source' entry has been removed in the '{finetune_file}' definition file.")
        update_model_def = True

    if is_module:
        quanto_filename = model_filename.replace(dtypestr, "quanto_" + quanto_dtypestr + "_int8" )
        quanto_filename_path = os.path.join(fl.get_download_folder() , quanto_filename)
        if hasattr(model, "_quanto_map"):
            print("unable to generate quantized module, the main model should at full 16 bits before quantization can be done")
        elif fl.locate_file(quanto_filename) is None:
            offload.save_model(model, quanto_filename_path, config_file_path=config_file, do_quantize= True, filter_sd=filter)
            print(f"New quantized file '{quanto_filename}' had been created for finetune Id '{model_type}'.")
            if isinstance(model_def[url_key][0],dict): 
                model_def[url_key][0][url_dict_key].append(quanto_filename) 
                saved_finetune_def["model"][url_key][0][url_dict_key].append(quanto_filename)
            else: 
                model_def[url_key][0].append(quanto_filename) 
                saved_finetune_def["model"][url_key][0].append(quanto_filename)
            update_model_def = True
    if update_model_def:
        with open(finetune_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(saved_finetune_def, indent=4))

def save_quantized_model(model, model_type, model_filename, dtype,  config_file, submodel_no = 1):
    if "quanto" in model_filename: return
    model_def = get_model_def(model_type)
    if model_def == None: return
    url_key = "URLs" if submodel_no <=1 else "URLs" + str(submodel_no)
    URLs= model_def.get(url_key, None)
    if URLs is None: return
    if isinstance(URLs, str):
        print("Unable to create a quantized model for a finetune that references external files")
        return
    from mmgp import offload
    if dtype == torch.bfloat16:
         model_filename =  model_filename.replace("fp16", "bf16").replace("FP16", "bf16")
    elif dtype == torch.float16:
         model_filename =  model_filename.replace("bf16", "fp16").replace("BF16", "bf16")

    for rep in ["mfp16", "fp16", "mbf16", "bf16"]:
        if "_" + rep in model_filename:
            model_filename = model_filename.replace("_" + rep, "_quanto_" + rep + "_int8")
            break
    if not "quanto" in model_filename:
        pos = model_filename.rfind(".")
        model_filename =  model_filename[:pos] + "_quanto_int8" + model_filename[pos+1:] 
    
    if fl.locate_file(model_filename) is not None:
        print(f"There isn't any model to quantize as quantized model '{model_filename}' aready exists")
    else:
        model_filename_path = os.path.join(fl.get_download_folder(), model_filename)
        offload.save_model(model, model_filename_path, do_quantize= True, config_file_path=config_file)
        print(f"New quantized file '{model_filename}' had been created for finetune Id '{model_type}'.")
        if not model_filename in URLs:
            URLs.append(model_filename)
            finetune_file = os.path.join(os.path.dirname(model_def["path"]) , model_type + ".json")
            with open(finetune_file, 'r', encoding='utf-8') as reader:
                saved_finetune_def = json.load(reader)
            saved_finetune_def["model"][url_key] = URLs
            with open(finetune_file, "w", encoding="utf-8") as writer:
                writer.write(json.dumps(saved_finetune_def, indent=4))
            print(f"The '{finetune_file}' definition file has been automatically updated with the local path to the new quantized model.")

def get_loras_preprocessor(transformer, model_type):
    preprocessor =  getattr(transformer, "preprocess_loras", None)
    if preprocessor == None:
        return None
    
    def preprocessor_wrapper(sd):
        return preprocessor(model_type, sd)

    return preprocessor_wrapper

def get_local_model_filename(model_filename):
    if model_filename.startswith("http"):
        local_model_filename =os.path.basename(model_filename)
    else:
        local_model_filename = model_filename
    local_model_filename = fl.locate_file(local_model_filename, error_if_none= False)
    return local_model_filename
    


def process_files_def(repoId = None, sourceFolderList = None, fileList = None, targetFolderList = None):
    original_targetRoot = fl.get_download_location()
    if targetFolderList is None:
        targetFolderList = [None] * len(sourceFolderList)
    for targetFolder, sourceFolder, files in zip(targetFolderList, sourceFolderList,fileList ):
        if targetFolder is not None and len(targetFolder) == 0:  targetFolder = None
        targetRoot = os.path.join(original_targetRoot, targetFolder) if targetFolder is not None else original_targetRoot            
        if len(files)==0:
            if fl.locate_folder(sourceFolder if targetFolder is None else os.path.join(targetFolder, sourceFolder), error_if_none= False ) is None:
                snapshot_download(repo_id=repoId,  allow_patterns=sourceFolder +"/*", local_dir= targetRoot)
        else:
            for onefile in files:     
                if len(sourceFolder) > 0: 
                    if fl.locate_file( (sourceFolder + "/" + onefile)  if targetFolder is None else os.path.join(targetFolder, sourceFolder, onefile), error_if_none= False) is None:   
                        hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = targetRoot, subfolder=sourceFolder)
                else:
                    if fl.locate_file(onefile if targetFolder is None else os.path.join(targetFolder, onefile), error_if_none= False) is None:          
                        hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = targetRoot)

def download_mmaudio():
    if server_config.get("mmaudio_enabled", 0) != 0:
        enhancer_def = {
            "repoId" : "DeepBeepMeep/Wan2.1",
            "sourceFolderList" : [ "mmaudio", "DFN5B-CLIP-ViT-H-14-378"  ],
            "fileList" : [ ["mmaudio_large_44k_v2.pth", "synchformer_state_dict.pth", "v1-44.pth"],["open_clip_config.json", "open_clip_pytorch_model.bin"]]
        }
        process_files_def(**enhancer_def)


def download_file(url,filename):
    if url.startswith("https://huggingface.co/") and "/resolve/main/" in url:
        base_dir = os.path.dirname(filename)
        url = url[len("https://huggingface.co/"):]
        url_parts = url.split("/resolve/main/")
        repoId = url_parts[0]
        onefile = os.path.basename(url_parts[-1])
        sourceFolder = os.path.dirname(url_parts[-1])
        if len(sourceFolder) == 0:
            hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = fl.get_download_location() if len(base_dir)==0 else base_dir)
        else:
            temp_dir_path = os.path.join(fl.get_download_location(), "temp")
            target_path = os.path.join(temp_dir_path, sourceFolder)
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = temp_dir_path, subfolder=sourceFolder)
            shutil.move(os.path.join( target_path, onefile), fl.get_download_location() if len(base_dir)==0 else base_dir)
            shutil.rmtree(temp_dir_path)
    else:
        from urllib.request import urlretrieve
        from shared.utils.download import create_progress_hook
        urlretrieve(url,filename, create_progress_hook(filename))

download_shared_done = False
def download_models(model_filename = None, model_type= None, module_type = False, submodel_no = 1):
    def computeList(filename):
        if filename == None:
            return []
        pos = filename.rfind("/")
        filename = filename[pos+1:]
        return [filename]        




    shared_def = {
        "repoId" : "DeepBeepMeep/Wan2.1",
        "sourceFolderList" : [ "pose", "scribble", "flow", "depth", "mask", "wav2vec", "chinese-wav2vec2-base", "roformer", "pyannote", "det_align", "" ],
        "fileList" : [ ["dw-ll_ucoco_384.onnx", "yolox_l.onnx"],["netG_A_latest.pth"],  ["raft-things.pth"], 
                      ["depth_anything_v2_vitl.pth","depth_anything_v2_vitb.pth"], ["sam_vit_h_4b8939_fp16.safetensors", "model.safetensors", "config.json"], 
                      ["config.json", "feature_extractor_config.json", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.json"],
                      ["config.json", "pytorch_model.bin", "preprocessor_config.json"],
                      ["model_bs_roformer_ep_317_sdr_12.9755.ckpt", "model_bs_roformer_ep_317_sdr_12.9755.yaml", "download_checks.json"],
                      ["pyannote_model_wespeaker-voxceleb-resnet34-LM.bin", "pytorch_model_segmentation-3.0.bin"], ["detface.pt"], [ "flownet.pkl" ] ]
    }
    process_files_def(**shared_def)


    if server_config.get("enhancer_enabled", 0) == 1:
        enhancer_def = {
            "repoId" : "DeepBeepMeep/LTX_Video",
            "sourceFolderList" : [ "Florence2", "Llama3_2"  ],
            "fileList" : [ ["config.json", "configuration_florence2.py", "model.safetensors", "modeling_florence2.py", "preprocessor_config.json", "processing_florence2.py", "tokenizer.json", "tokenizer_config.json"],["config.json", "generation_config.json", "Llama3_2_quanto_bf16_int8.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]  ]
        }
        process_files_def(**enhancer_def)

    elif server_config.get("enhancer_enabled", 0) == 2:
        enhancer_def = {
            "repoId" : "DeepBeepMeep/LTX_Video",
            "sourceFolderList" : [ "Florence2", "llama-joycaption-beta-one-hf-llava"  ],
            "fileList" : [ ["config.json", "configuration_florence2.py", "model.safetensors", "modeling_florence2.py", "preprocessor_config.json", "processing_florence2.py", "tokenizer.json", "tokenizer_config.json"],["config.json", "llama_joycaption_quanto_bf16_int8.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]  ]
        }
        process_files_def(**enhancer_def)

    download_mmaudio()
    global download_shared_done
    download_shared_done = True

    if model_filename is None: return

    base_model_type = get_base_model_type(model_type)
    model_def = get_model_def(model_type)
    
    any_source = ("source2" if submodel_no ==2 else "source") in model_def
    any_module_source = ("module_source2" if submodel_no ==2 else "module_source") in model_def 
    model_type_handler = model_types_handlers[base_model_type]
 
    if any_source and not module_type or any_module_source and module_type:
        model_filename = None
    else:
        local_model_filename = get_local_model_filename(model_filename)
        if local_model_filename is None and len(model_filename) > 0:
            local_model_filename = fl.get_download_location(os.path.basename(model_filename))
            url = model_filename

            if not url.startswith("http"):
                raise Exception(f"Model '{model_filename}' was not found locally and no URL was provided to download it. Please add an URL in the model definition file.")
            try:
                download_file(url, local_model_filename)
            except Exception as e:
                if os.path.isfile(local_model_filename): os.remove(local_model_filename) 
                raise Exception(f"'{url}' is invalid for Model '{model_type}' : {str(e)}'")
            if module_type: return
        model_filename = None

    for prop in ["preload_URLs", "text_encoder_URLs"]:
        preload_URLs = get_model_recursive_prop(model_type, prop, return_list= True)
        if prop in  ["text_encoder_URLs"]:
            preload_URLs = [get_model_filename(model_type=model_type, quantization= text_encoder_quantization, dtype_policy = transformer_dtype_policy, URLs=preload_URLs)] if len(preload_URLs) > 0 else []

        for url in preload_URLs:
            filename = fl.locate_file(os.path.basename(url), error_if_none= False)
            if filename is None: 
                filename = fl.get_download_location(os.path.basename(url))
                if not url.startswith("http"):
                    raise Exception(f"{prop}{filename}' was not found locally and no URL was provided to download it. Please add an URL in the model definition file.")
                try:
                    download_file(url, filename)
                except Exception as e:
                    if os.path.isfile(filename): os.remove(filename) 
                    raise Exception(f"{prop} '{url}' is invalid: {str(e)}'")

    model_loras = get_model_recursive_prop(model_type, "loras", return_list= True)
    for url in model_loras:
        filename = os.path.join(get_lora_dir(model_type), url.split("/")[-1])
        if not os.path.isfile(filename ): 
            if not url.startswith("http"):
                raise Exception(f"Lora '{filename}' was not found in the Loras Folder and no URL was provided to download it. Please add an URL in the model definition file.")
            try:
                download_file(url, filename)
            except Exception as e:
                if os.path.isfile(filename): os.remove(filename) 
                raise Exception(f"Lora URL '{url}' is invalid: {str(e)}'")
            
    if module_type: return            
    model_files = model_type_handler.query_model_files(computeList, base_model_type, model_filename, text_encoder_quantization)
    if not isinstance(model_files, list): model_files = [model_files]
    for one_repo in model_files:
        process_files_def(**one_repo)

offload.default_verboseLevel = verbose_level



def check_loras_exist(model_type, loras_choices_files, download = False, send_cmd = None):
    lora_dir = get_lora_dir(model_type)
    manager = _get_task_inputs_manager()
    cache = manager.get_loras_url_cache()
    missing_local_loras = []
    missing_remote_loras = []
    for lora_file in loras_choices_files:
        local_path = os.path.join(lora_dir, os.path.basename(lora_file))
        if not os.path.isfile(local_path):
            url = cache.get(local_path)
            if url is not None:
                if download:
                    if send_cmd is not None:
                        send_cmd("status", f'Downloading Lora {os.path.basename(lora_file)}...')
                    try:
                        download_file(url, local_path)
                    except Exception:
                        missing_remote_loras.append(lora_file)
            else:
                missing_local_loras.append(lora_file)

    error = ""
    if len(missing_local_loras) > 0:
        error += f"The following Loras files are missing or invalid: {missing_local_loras}."
    if len(missing_remote_loras) > 0:
        error += f"The following Loras files could not be downloaded: {missing_remote_loras}."
    
    return error

def extract_preset(model_type, lset_name, loras):
    loras_choices = []
    loras_choices_files = []
    loras_mult_choices = ""
    prompt =""
    full_prompt =""
    lset_name = sanitize_file_name(lset_name)
    lora_dir = get_lora_dir(model_type)
    if not lset_name.endswith(".lset"):
        lset_name_filename = os.path.join(lora_dir, lset_name + ".lset" ) 
    else:
        lset_name_filename = os.path.join(lora_dir, lset_name ) 
    error = ""
    if not os.path.isfile(lset_name_filename):
        error = f"Preset '{lset_name}' not found "
    else:

        with open(lset_name_filename, "r", encoding="utf-8") as reader:
            text = reader.read()
        lset = json.loads(text)

        loras_choices = lset["loras"]
        loras_mult_choices = lset["loras_mult"]
        prompt = lset.get("prompt", "")
        full_prompt = lset.get("full_prompt", False)
    return loras_choices, loras_mult_choices, prompt, full_prompt, error


def setup_loras(model_type, transformer,  lora_dir, lora_preselected_preset, split_linear_modules_map = None):
    loras =[]
    default_loras_choices = []
    default_loras_multis_str = ""
    loras_presets = []
    default_lora_preset = ""
    default_lora_preset_prompt = ""

    from pathlib import Path

    lora_dir = get_lora_dir(model_type)
    if lora_dir != None :
        if not os.path.isdir(lora_dir):
            raise Exception("--lora-dir should be a path to a directory that contains Loras")


    if lora_dir != None:
        dir_loras =  glob.glob( os.path.join(lora_dir , "*.sft") ) + glob.glob( os.path.join(lora_dir , "*.safetensors") ) 
        dir_loras.sort()
        loras += [element for element in dir_loras if element not in loras ]

        dir_presets_settings = glob.glob( os.path.join(lora_dir , "*.json") ) 
        dir_presets_settings.sort()
        dir_presets =   glob.glob( os.path.join(lora_dir , "*.lset") ) 
        dir_presets.sort()
        # loras_presets = [ Path(Path(file_path).parts[-1]).stem for file_path in dir_presets_settings + dir_presets]
        loras_presets = [ Path(file_path).parts[-1] for file_path in dir_presets_settings + dir_presets]

    if transformer !=None:
        loras = offload.load_loras_into_model(transformer, loras,  activate_all_loras=False, check_only= True, preprocess_sd=get_loras_preprocessor(transformer, model_type), split_linear_modules_map = split_linear_modules_map) #lora_multiplier,

    if len(loras) > 0:
        loras = [ os.path.basename(lora) for lora in loras  ]

    if len(lora_preselected_preset) > 0:
        if not os.path.isfile(os.path.join(lora_dir, lora_preselected_preset + ".lset")):
            raise Exception(f"Unknown preset '{lora_preselected_preset}'")
        default_lora_preset = lora_preselected_preset
        default_loras_choices, default_loras_multis_str, default_lora_preset_prompt, _ , error = extract_preset(model_type, default_lora_preset, loras)
        if len(error) > 0:
            print(error[:200])
    return loras, loras_presets, default_loras_choices, default_loras_multis_str, default_lora_preset_prompt, default_lora_preset

def get_transformer_model(model, submodel_no = 1):
    if submodel_no > 1:
        model_key = f"model{submodel_no}"
        if not hasattr(model, model_key): return None

    if hasattr(model, "model"):
        if submodel_no > 1:
            return getattr(model, f"model{submodel_no}")
        else:
            return model.model
    elif hasattr(model, "transformer"):
        return model.transformer
    else:
        raise Exception("no transformer found")

def init_pipe(pipe, kwargs, override_profile):
    preload =int(args.preload)
    if preload == 0:
        preload = server_config.get("preload_in_VRAM", 0)

    kwargs["extraModelsToQuantize"]=  None
    profile = override_profile if override_profile != -1 else default_profile
    if profile in (2, 4, 5):
        budgets = { "transformer" : 100 if preload  == 0 else preload, "text_encoder" : 100 if preload  == 0 else preload, "*" : max(1000 if profile==5 else 3000 , preload) }
        if "transformer2" in pipe:
            budgets["transformer2"] = 100 if preload  == 0 else preload
        kwargs["budgets"] = budgets
    elif profile == 3:
        kwargs["budgets"] = { "*" : "70%" }

    if "transformer2" in pipe:
        if profile in [3,4]:
            kwargs["pinnedMemory"] = ["transformer", "transformer2"]

    return profile

def reset_prompt_enhancer():
    global prompt_enhancer_image_caption_model, prompt_enhancer_image_caption_processor, prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer, enhancer_offloadobj
    prompt_enhancer_image_caption_model = None
    prompt_enhancer_image_caption_processor = None
    prompt_enhancer_llm_model = None
    prompt_enhancer_llm_tokenizer = None
    if enhancer_offloadobj is not None:
        enhancer_offloadobj.release()
        enhancer_offloadobj = None

def setup_prompt_enhancer(pipe, kwargs):
    global prompt_enhancer_image_caption_model, prompt_enhancer_image_caption_processor, prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer
    model_no = server_config.get("enhancer_enabled", 0) 
    if model_no != 0:
        from transformers import ( AutoModelForCausalLM, AutoProcessor, AutoTokenizer, LlamaForCausalLM )
        prompt_enhancer_image_caption_model = AutoModelForCausalLM.from_pretrained(fl.locate_folder("Florence2"), trust_remote_code=True)
        prompt_enhancer_image_caption_processor = AutoProcessor.from_pretrained(fl.locate_folder("Florence2"), trust_remote_code=True)
        pipe["prompt_enhancer_image_caption_model"] = prompt_enhancer_image_caption_model
        prompt_enhancer_image_caption_model._model_dtype = torch.float
        # def preprocess_sd(sd, map):
        #     new_sd ={}
        #     for k, v in sd.items():
        #         k = "model." + k.replace(".model.", ".")
        #         if "lm_head.weight" in k: k = "lm_head.weight"
        #         new_sd[k] = v
        #     return new_sd, map
        # prompt_enhancer_llm_model = offload.fast_load_transformers_model("c:/temp/joy/model-00001-of-00004.safetensors", modelClass= LlavaForConditionalGeneration, defaultConfigPath="ckpts/llama-joycaption-beta-one-hf-llava/config.json", preprocess_sd=preprocess_sd)
        # offload.save_model(prompt_enhancer_llm_model, "joy_llava_quanto_int8.safetensors", do_quantize= True)

        if model_no == 1:
            budget = 5000
            prompt_enhancer_llm_model = offload.fast_load_transformers_model( fl.locate_file("Llama3_2/Llama3_2_quanto_bf16_int8.safetensors"))
            prompt_enhancer_llm_tokenizer = AutoTokenizer.from_pretrained(fl.locate_folder("Llama3_2"))
        else:
            budget = 10000
            prompt_enhancer_llm_model = offload.fast_load_transformers_model(fl.locate_file("llama-joycaption-beta-one-hf-llava/llama_joycaption_quanto_bf16_int8.safetensors"))
            prompt_enhancer_llm_tokenizer = AutoTokenizer.from_pretrained(fl.locate_folder("llama-joycaption-beta-one-hf-llava"))
        pipe["prompt_enhancer_llm_model"] = prompt_enhancer_llm_model
        if not "budgets" in kwargs: kwargs["budgets"] = {}
        kwargs["budgets"]["prompt_enhancer_llm_model"] = budget 
    else:
        reset_prompt_enhancer()



def load_models(model_type, override_profile = -1):
    global transformer_type, loaded_profile
    base_model_type = get_base_model_type(model_type)
    model_def = get_model_def(model_type)
    save_quantized = args.save_quantized and model_def != None
    model_filename = get_model_filename(model_type=model_type, quantization= "" if save_quantized else transformer_quantization, dtype_policy = transformer_dtype_policy) 
    if "URLs2" in model_def:
        model_filename2 = get_model_filename(model_type=model_type, quantization= "" if save_quantized else transformer_quantization, dtype_policy = transformer_dtype_policy, submodel_no=2) # !!!!
    else:
        model_filename2 = None
    modules = get_model_recursive_prop(model_type, "modules",  return_list= True)
    modules = [get_model_recursive_prop(module, "modules", sub_prop_name  ="_list",  return_list= True) if isinstance(module, str) else module for module in modules ]
    if save_quantized and "quanto" in model_filename:
        save_quantized = False
        print("Need to provide a non quantized model to create a quantized model to be saved") 
    if save_quantized and len(modules) > 0:
        print(f"Unable to create a finetune quantized model as some modules are declared in the finetune definition. If your finetune includes already the module weights you can remove the 'modules' entry and try again. If not you will need also to change temporarly the model 'architecture' to an architecture that wont require the modules part ({modules}) to quantize and then add back the original 'modules' and 'architecture' entries.")
        save_quantized = False
    quantizeTransformer = not save_quantized and model_def !=None and transformer_quantization in ("int8", "fp8") and model_def.get("auto_quantize", False) and not "quanto" in model_filename
    if quantizeTransformer and len(modules) > 0:
        print(f"Autoquantize is not yet supported if some modules are declared")
        quantizeTransformer = False
    model_family = get_model_family(model_type)
    transformer_dtype = get_transformer_dtype(model_family, transformer_dtype_policy)
    if quantizeTransformer or "quanto" in model_filename:
        transformer_dtype = torch.bfloat16 if "bf16" in model_filename or "BF16" in model_filename else transformer_dtype
        transformer_dtype = torch.float16 if "fp16" in model_filename or"FP16" in model_filename else transformer_dtype
    perc_reserved_mem_max = args.perc_reserved_mem_max
    vram_safety_coefficient = args.vram_safety_coefficient 
    model_file_list = [model_filename]
    model_type_list = [model_type]
    module_type_list = [None]
    model_submodel_no_list = [1]
    if model_filename2 != None:
        model_file_list += [model_filename2]
        model_type_list += [model_type]
        module_type_list += [None]
        model_submodel_no_list += [2]
    for module_type in modules:
        if isinstance(module_type,dict):
            URLs1 = module_type.get("URLs", None)
            if URLs1 is None: raise Exception(f"No URLs defined for Module {module_type}")
            model_file_list.append(get_model_filename(model_type, transformer_quantization, transformer_dtype, URLs = URLs1))
            URLs2 = module_type.get("URLs2", None)
            if URLs2 is None: raise Exception(f"No URL2s defined for Module {module_type}")
            model_file_list.append(get_model_filename(model_type, transformer_quantization, transformer_dtype, URLs = URLs2))
            model_type_list += [model_type] * 2
            module_type_list += [True] * 2
            model_submodel_no_list += [1,2]
        else:
            model_file_list.append(get_model_filename(model_type, transformer_quantization, transformer_dtype, module_type= module_type))
            model_type_list.append(model_type)
            module_type_list.append(True)
            model_submodel_no_list.append(0) 
    local_model_file_list= []
    for filename, file_model_type, file_module_type, submodel_no in zip(model_file_list, model_type_list, module_type_list, model_submodel_no_list):
        if len(filename) == 0: continue 
        download_models(filename, file_model_type, file_module_type, submodel_no)
        local_model_file_list.append( get_local_model_filename(filename) )
    if len(local_model_file_list) == 0:
        download_models("", model_type, "", -1)

    VAE_dtype = torch.float16 if server_config.get("vae_precision","16") == "16" else torch.float
    mixed_precision_transformer =  server_config.get("mixed_precision","0") == "1"
    transformer_type = None
    for module_type, filename in zip(module_type_list, local_model_file_list):
        if module_type is None:  
            print(f"Loading Model '{filename}' ...")
        else: 
            print(f"Loading Module '{filename}' ...")


    override_text_encoder = get_model_recursive_prop(model_type, "text_encoder_URLs", return_list= True)
    if len( override_text_encoder) > 0:
        override_text_encoder = get_model_filename(model_type=model_type, quantization= text_encoder_quantization, dtype_policy = transformer_dtype_policy, URLs=override_text_encoder) if len(override_text_encoder) > 0 else None
        if override_text_encoder is not None:
            override_text_encoder =  get_local_model_filename(override_text_encoder)
            if override_text_encoder is not None:
                print(f"Loading Text Encoder '{override_text_encoder}' ...")
    else:
        override_text_encoder = None

    wan_model, pipe = model_types_handlers[base_model_type].load_model(
                local_model_file_list, model_type, base_model_type, model_def, quantizeTransformer = quantizeTransformer, text_encoder_quantization = text_encoder_quantization,
                dtype = transformer_dtype, VAE_dtype = VAE_dtype, mixed_precision_transformer = mixed_precision_transformer, save_quantized = save_quantized, submodel_no_list   = model_submodel_no_list, override_text_encoder = override_text_encoder )

    kwargs = {}
    if "pipe" in pipe:
        kwargs = pipe
        pipe = kwargs.pop("pipe")
    if "coTenantsMap" not in kwargs: kwargs["coTenantsMap"] = {}

    profile = init_pipe(pipe, kwargs, override_profile)
    if server_config.get("enhancer_mode", 0) == 0:
        setup_prompt_enhancer(pipe, kwargs)
    loras_transformer = []
    if "transformer" in pipe:
        loras_transformer += ["transformer"]        
    if "transformer2" in pipe:
        loras_transformer += ["transformer2"]
    if len(compile) > 0 and hasattr(wan_model, "custom_compile"):
        wan_model.custom_compile(backend= "inductor", mode ="default")
    compile_modules = model_def.get("compile", compile) if len(compile) > 0 else ""
    offloadobj = offload.profile(pipe, profile_no= profile, compile = compile_modules, quantizeTransformer = False, loras = loras_transformer, perc_reserved_mem_max = perc_reserved_mem_max , vram_safety_coefficient = vram_safety_coefficient , convertWeightsFloatTo = transformer_dtype, **kwargs)  
    if len(args.gpu) > 0:
        torch.set_default_device(args.gpu)
    transformer_type = model_type
    loaded_profile = profile
    return wan_model, offloadobj 

if not "P" in preload_model_policy:
    wan_model, offloadobj, transformer = None, None, None
    reload_needed = True
else:
    wan_model, offloadobj = load_models(transformer_type)
    if check_loras:
        transformer = get_transformer_model(wan_model)
        setup_loras(transformer_type, transformer,  get_lora_dir(transformer_type), "", None)
        exit()

gen_in_progress = False

def is_generation_in_progress():
    global gen_in_progress
    return gen_in_progress

def get_auto_attention():
    for attn in ["sage2","sage","sdpa"]:
        if attn in attention_modes_supported:
            return attn
    return "sdpa"

def generate_header(model_type, compile, attention_mode):

    description_container = [""]
    get_model_name(model_type, description_container)
    model_filename = os.path.basename(get_model_filename(model_type, transformer_quantization, transformer_dtype_policy)) or "" 
    description  = description_container[0]
    header = f"<DIV style=height:40px>{description}</DIV>"
    overridden_attention = get_overridden_attention(model_type)
    attn_mode = attention_mode if overridden_attention == None else overridden_attention 
    header += "<DIV style='align:right;width:100%'><FONT SIZE=3>Attention mode <B>" + (attn_mode if attn_mode!="auto" else "auto/" + get_auto_attention() )
    if attention_mode not in attention_modes_installed:
        header += " -NOT INSTALLED-"
    elif attention_mode not in attention_modes_supported:
        header += " -NOT SUPPORTED-"
    elif overridden_attention is not None and attention_mode != overridden_attention:
        header += " -MODEL SPECIFIC-"
    header += "</B>"

    if compile:
        header += ", Pytorch compilation <B>ON</B>"
    if "fp16" in model_filename:
        header += ", Data Type <B>FP16</B>"
    else:
        header += ", Data Type <B>BF16</B>"

    if "int8" in model_filename:
        header += ", Quantization <B>Scaled Int8</B>"
    header += "<FONT></DIV>"

    return header

def release_RAM():
    if gen_in_progress:
        notify_info("Unable to release RAM when a Generation is in Progress")
    else:
        release_model()
        notify_info("Models stored in RAM have been released")

def pack_audio_gallery_state(*_args, **_kwargs):
    raise RuntimeError('Audio gallery state management was removed in the headless build.')

def unpack_audio_list(*_args, **_kwargs):
    raise RuntimeError('Audio gallery state management was removed in the headless build.')

def refresh_gallery(*_args, **_kwargs):
    raise RuntimeError('Interactive galleries were removed in the headless build.')

def finalize_generation(*_args, **_kwargs):
    raise RuntimeError('Interactive galleries were removed in the headless build.')

def get_default_video_info():
    return "Please Select an Video / Image"    


def get_file_list(state, input_file_list, audio_files = False):
    manager = _get_task_inputs_manager()
    return manager.get_file_list(state, input_file_list, audio_files=audio_files)

def set_file_choice(gen, file_list, choice, audio_files = False):
    TaskInputManager.set_file_choice(gen, file_list, choice, audio_files=audio_files)

def select_audio(state, audio_files_paths, audio_file_selected):
    gen = get_gen_info(state)
    audio_file_list, audio_file_settings_list = get_file_list(state, unpack_audio_list(audio_files_paths))

    if audio_file_selected >= 0:
        choice = audio_file_selected
    else:
        choice = min(len(audio_file_list)-1, gen.get("audio_selected",0)) if len(audio_file_list) > 0 else -1
    set_file_choice(gen,  audio_file_list, choice, audio_files=True )
def select_video(*_args, **_kwargs):
    raise RuntimeError('Interactive galleries were removed in the headless build.')

def convert_image(image):

    from PIL import ImageOps
    from typing import cast
    if isinstance(image, str):
        image = Image.open(image)
    image = image.convert('RGB')
    return cast(Image, ImageOps.exif_transpose(image))

def get_resampled_video(video_in, start_frame, max_frames, target_fps, bridge='torch'):
    if isinstance(video_in, str) and has_image_file_extension(video_in):
        video_in = Image.open(video_in)
    if isinstance(video_in, Image.Image):
        return torch.from_numpy(np.array(video_in).astype(np.uint8)).unsqueeze(0)
    
    from shared.utils.utils import resample

    import decord
    decord.bridge.set_bridge(bridge)
    reader = decord.VideoReader(video_in)
    fps = round(reader.get_avg_fps())
    if max_frames < 0:
        max_frames = max(len(reader)/ fps * target_fps + max_frames, 0)


    frame_nos = resample(fps, len(reader), max_target_frames_count= max_frames, target_fps=target_fps, start_target_frame= start_frame)
    frames_list = reader.get_batch(frame_nos)
    # print(f"frame nos: {frame_nos}")
    return frames_list

# def get_resampled_video(video_in, start_frame, max_frames, target_fps):
#     from torchvision.io import VideoReader
#     import torch
#     from shared.utils.utils import resample

#     vr = VideoReader(video_in, "video")
#     meta = vr.get_metadata()["video"]

#     fps = round(float(meta["fps"][0]))
#     duration_s = float(meta["duration"][0])
#     num_src_frames = int(round(duration_s * fps))  # robust length estimate

#     if max_frames < 0:
#         max_frames = max(int(num_src_frames / fps * target_fps + max_frames), 0)

#     frame_nos = resample(
#         fps, num_src_frames,
#         max_target_frames_count=max_frames,
#         target_fps=target_fps,
#         start_target_frame=start_frame
#     )
#     if len(frame_nos) == 0:
#         return torch.empty((0,))  # nothing to return

#     target_ts = [i / fps for i in frame_nos]

#     # Read forward once, grabbing frames when we pass each target timestamp
#     frames = []
#     vr.seek(target_ts[0])
#     idx = 0
#     tol = 0.5 / fps  # half-frame tolerance
#     for frame in vr:
#         t = float(frame["pts"])       # seconds
#         if idx < len(target_ts) and t + tol >= target_ts[idx]:
#             frames.append(frame["data"].permute(1,2,0))  # Tensor [H, W, C]
#             idx += 1
#             if idx >= len(target_ts):
#                 break

#     return frames


def get_preprocessor(process_type, inpaint_color):
    if process_type=="pose":
        from preprocessing.dwpose.pose import PoseBodyFaceVideoAnnotator
        cfg_dict = {
            "DETECTION_MODEL": fl.locate_file("pose/yolox_l.onnx"),
            "POSE_MODEL": fl.locate_file("pose/dw-ll_ucoco_384.onnx"),
            "RESIZE_SIZE": 1024
        }
        anno_ins = lambda img: PoseBodyFaceVideoAnnotator(cfg_dict).forward(img)
    elif process_type=="depth":

        from preprocessing.depth_anything_v2.depth import DepthV2VideoAnnotator

        if server_config.get("depth_anything_v2_variant", "vitl") == "vitl":
            cfg_dict = {
                "PRETRAINED_MODEL": fl.locate_file("depth/depth_anything_v2_vitl.pth"),
                'MODEL_VARIANT': 'vitl'
            }
        else:
            cfg_dict = {
                "PRETRAINED_MODEL": fl.locate_file("depth/depth_anything_v2_vitb.pth"),
                'MODEL_VARIANT': 'vitb',
            }

        anno_ins = lambda img: DepthV2VideoAnnotator(cfg_dict).forward(img)
    elif process_type=="gray":
        from preprocessing.gray import GrayVideoAnnotator
        cfg_dict = {}
        anno_ins = lambda img: GrayVideoAnnotator(cfg_dict).forward(img)
    elif process_type=="canny":
        from preprocessing.canny import CannyVideoAnnotator
        cfg_dict = {
                "PRETRAINED_MODEL": fl.locate_file("scribble/netG_A_latest.pth")
            }
        anno_ins = lambda img: CannyVideoAnnotator(cfg_dict).forward(img)
    elif process_type=="scribble":
        from preprocessing.scribble import ScribbleVideoAnnotator
        cfg_dict = {
                "PRETRAINED_MODEL": fl.locate_file("scribble/netG_A_latest.pth")
            }
        anno_ins = lambda img: ScribbleVideoAnnotator(cfg_dict).forward(img)
    elif process_type=="flow":
        from preprocessing.flow import FlowVisAnnotator
        cfg_dict = {
                "PRETRAINED_MODEL": fl.locate_file("flow/raft-things.pth")
            }
        anno_ins = lambda img: FlowVisAnnotator(cfg_dict).forward(img)
    elif process_type=="inpaint":
        anno_ins = lambda img :  len(img) * [inpaint_color]
    elif process_type == None or process_type in ["raw", "identity"]:
        anno_ins = lambda img : img
    else:
        raise Exception(f"process type '{process_type}' non supported")
    return anno_ins




def extract_faces_from_video_with_mask(*_args, **_kwargs):
    """
    Placeholder for the legacy face-cropping routine used by Lynx/Stand-in flows.

    Historically, the implementation:
    - Consumed `input_video_path` plus an optional `input_mask_path`, `max_frames`,
      `start_frame`, `target_fps`, and a `size` parameter (default 512).
    - Resampled frames to `target_fps`, optionally applied the binary mask per frame,
      and relied on `AlignImage` (InsightFace-based) to detect and crop the dominant face.
    - Returned a tensor shaped `[3, num_frames, size, size]` in the `[-1, 1]` range,
      padding/triming to match the requested window length. When `args.save_masks`
      was set it also dumped a diagnostic MP4 of the cropped faces.

    Any replacement should mirror that contract so Wan’s Lynx/Stand-in modules can
    reuse the downstream VAE / rotary embedding logic without further changes.
    """
    raise RuntimeError(
        "Face extraction is temporarily disabled; Lynx/Stand-in workflows require the upcoming CLI face cropper."
    )


def preprocess_video_with_mask(input_video_path, input_mask_path, height, width,  max_frames, start_frame=0, fit_canvas = None, fit_crop = False, target_fps = 16, block_size= 16, expand_scale = 2, process_type = "inpaint", process_type2 = None, to_bbox = False, RGB_Mask = False, negate_mask = False, process_outside_mask = None, inpaint_color = 127, outpainting_dims = None, proc_no = 1):

    def mask_to_xyxy_box(mask):
        rows, cols = np.where(mask == 255)
        xmin = min(cols)
        xmax = max(cols) + 1
        ymin = min(rows)
        ymax = max(rows) + 1
        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(xmax, mask.shape[1])
        ymax = min(ymax, mask.shape[0])
        box = [xmin, ymin, xmax, ymax]
        box = [int(x) for x in box]
        return box
    inpaint_color = int(inpaint_color)
    pad_frames = 0
    if start_frame < 0:
        pad_frames= -start_frame
        max_frames += start_frame
        start_frame = 0

    if not input_video_path or max_frames <= 0:
        return None, None
    any_mask = input_mask_path != None
    pose_special = "pose" in process_type
    any_identity_mask = False
    if process_type == "identity":
        any_identity_mask = True
        negate_mask = False
        process_outside_mask = None
    preproc = get_preprocessor(process_type, inpaint_color)
    preproc2 = None
    if process_type2 != None:
        preproc2 = get_preprocessor(process_type2, inpaint_color) if process_type != process_type2 else preproc
    if process_outside_mask == process_type :
        preproc_outside = preproc
    elif preproc2 != None and process_outside_mask == process_type2 :
        preproc_outside = preproc2
    else:
        preproc_outside = get_preprocessor(process_outside_mask, inpaint_color)
    video = get_resampled_video(input_video_path, start_frame, max_frames, target_fps)
    if any_mask:
        mask_video = get_resampled_video(input_mask_path, start_frame, max_frames, target_fps)

    if len(video) == 0 or any_mask and len(mask_video) == 0:
        return None, None
    if fit_crop and outpainting_dims != None:
        fit_crop = False
        fit_canvas = 0 if fit_canvas is not None else None

    frame_height, frame_width, _ = video[0].shape

    if outpainting_dims != None:
        if fit_canvas != None:
            frame_height, frame_width = get_outpainting_full_area_dimensions(frame_height,frame_width, outpainting_dims)
        else:
            frame_height, frame_width = height, width

    if fit_canvas != None:
        height, width = calculate_new_dimensions(height, width, frame_height, frame_width, fit_into_canvas = fit_canvas, block_size = block_size)

    if outpainting_dims != None:
        final_height, final_width = height, width
        height, width, margin_top, margin_left =  get_outpainting_frame_location(final_height, final_width,  outpainting_dims, 1)        

    if any_mask:
        num_frames = min(len(video), len(mask_video))
    else:
        num_frames = len(video)

    if any_identity_mask:
        any_mask = True

    proc_list =[]
    proc_list_outside =[]
    proc_mask = []

    # for frame_idx in range(num_frames):
    def prep_prephase(frame_idx):
        frame = Image.fromarray(video[frame_idx].cpu().numpy()) #.asnumpy()
        if fit_crop:
            frame = rescale_and_crop(frame, width, height)
        else:
            frame = frame.resize((width, height), resample=Image.Resampling.LANCZOS) 
        frame = np.array(frame) 
        if any_mask:
            if any_identity_mask:
                mask = np.full( (height, width, 3), 0, dtype= np.uint8)
            else:
                mask = Image.fromarray(mask_video[frame_idx].cpu().numpy()) #.asnumpy()
                if fit_crop:
                    mask = rescale_and_crop(mask, width, height)
                else:
                    mask = mask.resize((width, height), resample=Image.Resampling.LANCZOS) 
                mask = np.array(mask)

            if len(mask.shape) == 3 and mask.shape[2] == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)
            original_mask = mask.copy()
            if expand_scale != 0:
                kernel_size = abs(expand_scale)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                op_expand = cv2.dilate if expand_scale > 0 else cv2.erode
                mask = op_expand(mask, kernel, iterations=3)

            if to_bbox and np.sum(mask == 255) > 0 : #or True 
                x0, y0, x1, y1 = mask_to_xyxy_box(mask)
                mask = mask * 0
                mask[y0:y1, x0:x1] = 255
            if negate_mask:
                mask = 255 - mask
                if pose_special:
                    original_mask = 255 - original_mask

        if pose_special and any_mask:            
            target_frame = np.where(original_mask[..., None], frame, 0) 
        else:
            target_frame = frame 

        if any_mask:
            return (target_frame, frame, mask) 
        else:
            return (target_frame, None, None)
    max_workers = get_default_workers()
    proc_lists = process_images_multithread(prep_prephase, [frame_idx for frame_idx in range(num_frames)], "prephase", wrap_in_list= False, max_workers=max_workers, in_place= True)
    proc_list, proc_list_outside, proc_mask = [None] * len(proc_lists), [None] * len(proc_lists), [None] * len(proc_lists)
    for frame_idx, frame_group in enumerate(proc_lists): 
        proc_list[frame_idx], proc_list_outside[frame_idx], proc_mask[frame_idx] = frame_group
    prep_prephase = None
    video = None
    mask_video = None

    if preproc2 != None:
        proc_list2 = process_images_multithread(preproc2, proc_list, process_type2, max_workers=max_workers)
        #### to be finished ...or not
    proc_list = process_images_multithread(preproc, proc_list, process_type, max_workers=max_workers)
    if any_mask:
        proc_list_outside = process_images_multithread(preproc_outside, proc_list_outside, process_outside_mask, max_workers=max_workers)
    else:
        proc_list_outside = proc_mask = len(proc_list) * [None]

    masked_frames = []
    masks = []
    for frame_no, (processed_img, processed_img_outside, mask) in enumerate(zip(proc_list, proc_list_outside, proc_mask)):
        if any_mask :
            masked_frame = np.where(mask[..., None], processed_img, processed_img_outside)
            if process_outside_mask != None:
                mask = np.full_like(mask, 255)
            mask = torch.from_numpy(mask)
            if RGB_Mask:
                mask =  mask.unsqueeze(-1).repeat(1,1,3)
            if outpainting_dims != None:
                full_frame= torch.full( (final_height, final_width, mask.shape[-1]), 255, dtype= torch.uint8, device= mask.device)
                full_frame[margin_top:margin_top+height, margin_left:margin_left+width] = mask
                mask = full_frame 
            masks.append(mask[:, :, 0:1].clone())
        else:
            masked_frame = processed_img

        if isinstance(masked_frame, int):
            masked_frame= np.full( (height, width, 3), inpaint_color, dtype= np.uint8)

        masked_frame = torch.from_numpy(masked_frame)
        if masked_frame.shape[-1] == 1:
            masked_frame =  masked_frame.repeat(1,1,3).to(torch.uint8)

        if outpainting_dims != None:
            full_frame= torch.full( (final_height, final_width, masked_frame.shape[-1]),  inpaint_color, dtype= torch.uint8, device= masked_frame.device)
            full_frame[margin_top:margin_top+height, margin_left:margin_left+width] = masked_frame
            masked_frame = full_frame 

        masked_frames.append(masked_frame)
        proc_list[frame_no] = proc_list_outside[frame_no] = proc_mask[frame_no] = None


    # if args.save_masks:
    #     from preprocessing.dwpose.pose import save_one_video
    #     saved_masked_frames = [mask.cpu().numpy() for mask in masked_frames ]
    #     save_one_video(f"masked_frames{'' if proc_no==1 else str(proc_no)}.mp4", saved_masked_frames, fps=target_fps, quality=8, macro_block_size=None)
    #     if any_mask:
    #         saved_masks = [mask.cpu().numpy() for mask in masks ]
    #         save_one_video("masks.mp4", saved_masks, fps=target_fps, quality=8, macro_block_size=None)
    preproc = None
    preproc_outside = None
    gc.collect()
    torch.cuda.empty_cache()
    if pad_frames > 0:
        masked_frames = masked_frames[0] * pad_frames + masked_frames
        if any_mask: masked_frames = masks[0] * pad_frames + masks
    masked_frames = torch.stack(masked_frames).permute(-1,0,1,2).float().div_(127.5).sub_(1.)
    masks = torch.stack(masks).permute(-1,0,1,2).float().div_(255) if any_mask else None

    return masked_frames, masks

def preprocess_video(height, width, video_in, max_frames, start_frame=0, fit_canvas = None, fit_crop = False, target_fps = 16, block_size = 16):

    frames_list = get_resampled_video(video_in, start_frame, max_frames, target_fps)

    if len(frames_list) == 0:
        return None

    if fit_canvas == None or fit_crop:
        new_height = height
        new_width = width
    else:
        frame_height, frame_width, _ = frames_list[0].shape
        if fit_canvas :
            scale1  = min(height / frame_height, width /  frame_width)
            scale2  = min(height / frame_width, width /  frame_height)
            scale = max(scale1, scale2)
        else:
            scale =   ((height * width ) /  (frame_height * frame_width))**(1/2)

        new_height = (int(frame_height * scale) // block_size) * block_size
        new_width = (int(frame_width * scale) // block_size) * block_size

    processed_frames_list = []
    for frame in frames_list:
        frame = Image.fromarray(np.clip(frame.cpu().numpy(), 0, 255).astype(np.uint8))
        if fit_crop:
            frame  = rescale_and_crop(frame, new_width, new_height)
        else:
            frame = frame.resize((new_width,new_height), resample=Image.Resampling.LANCZOS) 
        processed_frames_list.append(frame)

    np_frames = [np.array(frame) for frame in processed_frames_list]

    # from preprocessing.dwpose.pose import save_one_video
    # save_one_video("test.mp4", np_frames, fps=8, quality=8, macro_block_size=None)

    torch_frames = []
    for np_frame in np_frames:
        torch_frame = torch.from_numpy(np_frame)
        torch_frames.append(torch_frame)

    return torch.stack(torch_frames) 

 
def parse_keep_frames_video_guide(keep_frames, video_length):
        
    def absolute(n):
        if n==0:
            return 0
        elif n < 0:
            return max(0, video_length + n)
        else:
            return min(n-1, video_length-1)
    keep_frames = keep_frames.strip()
    if len(keep_frames) == 0:
        return [True] *video_length, "" 
    frames =[False] *video_length
    error = ""
    sections = keep_frames.split(" ")
    for section in sections:
        section = section.strip()
        if ":" in section:
            parts = section.split(":")
            if not is_integer(parts[0]):
                error =f"Invalid integer {parts[0]}"
                break
            start_range = absolute(int(parts[0]))
            if not is_integer(parts[1]):
                error =f"Invalid integer {parts[1]}"
                break
            end_range = absolute(int(parts[1]))
            for i in range(start_range, end_range + 1):
                frames[i] = True
        else:
            if not is_integer(section) or int(section) == 0:
                error =f"Invalid integer {section}"
                break
            index = absolute(int(section))
            frames[index] = True

    if len(error ) > 0:
        return [], error
    for i in range(len(frames)-1, 0, -1):
        if frames[i]:
            break
    frames= frames[0: i+1]
    return  frames, error


def perform_temporal_upsampling(sample, previous_last_frame, temporal_upsampling, fps):
    exp = 0
    if temporal_upsampling == "rife2":
        exp = 1
    elif temporal_upsampling == "rife4":
        exp = 2
    output_fps = fps
    if exp > 0: 
        from postprocessing.rife.inference import temporal_interpolation
        if previous_last_frame != None:
            sample = torch.cat([previous_last_frame, sample], dim=1)
            previous_last_frame = sample[:, -1:].clone()
            sample = temporal_interpolation( fl.locate_file("flownet.pkl"), sample, exp, device=processing_device)
            sample = sample[:, 1:]
        else:
            sample = temporal_interpolation( fl.locate_file("flownet.pkl"), sample, exp, device=processing_device)
            previous_last_frame = sample[:, -1:].clone()

        output_fps = output_fps * 2**exp
    return sample, previous_last_frame, output_fps 


def perform_spatial_upsampling(sample, spatial_upsampling):
    from shared.utils.utils import resize_lanczos 
    if spatial_upsampling == "lanczos1.5":
        scale = 1.5
    else:
        scale = 2
    h, w = sample.shape[-2:]
    h *= scale
    h = round(h/16) * 16
    w *= scale
    w = round(w/16) * 16
    h = int(h)
    w = int(w)
    frames_to_upsample = [sample[:, i] for i in range( sample.shape[1]) ] 
    def upsample_frames(frame):
        return resize_lanczos(frame, h, w).unsqueeze(1)
    sample = torch.cat(process_images_multithread(upsample_frames, frames_to_upsample, "upsample", wrap_in_list = False, max_workers=get_default_workers(), in_place=True), dim=1)
    frames_to_upsample = None
    return sample 

def any_audio_track(model_type):
    base_model_type = get_base_model_type(model_type)
    if base_model_type in ["fantasy", "chatterbox"]:
        return True
    model_def = get_model_def(model_type)
    if not model_def:
        return False
    if model_def.get("returns_audio", False):
        return True
    return model_def.get("multitalk_class", False)

def set_seed(seed):
    import random
    seed = random.randint(0, 99999999) if seed == None or seed < 0 else seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return seed

def edit_video(
                send_cmd,
                state,
                mode,
                video_source,
                seed,   
                temporal_upsampling,
                spatial_upsampling,
                film_grain_intensity,
                film_grain_saturation,
                MMAudio_setting,
                MMAudio_prompt,
                MMAudio_neg_prompt,
                repeat_generation,
                audio_source,
                **kwargs
                ):



    gen = get_gen_info(state)

    if gen.get("abort", False): return 
    abort = False
		
		
    MMAudio_setting = MMAudio_setting or 0
    task_inputs_manager = _get_task_inputs_manager()
    configs, _, _ = task_inputs_manager.load_settings_from_file(state, video_source, False, False, False)
    if configs == None: configs = { "type" : get_model_record("Post Processing") }

    has_already_audio = False
    audio_tracks = []
    if MMAudio_setting == 0:
        audio_tracks, audio_metadata  = extract_audio_tracks(video_source)
        has_already_audio = len(audio_tracks) > 0
    
    if audio_source is not None:
        audio_tracks = [audio_source]

    with lock:
        file_list = gen["file_list"]
        file_settings_list = gen["file_settings_list"]



    seed = set_seed(seed)

    from shared.utils.utils import get_video_info
    fps, width, height, frames_count = get_video_info(video_source)        
    frames_count = min(frames_count, max_source_video_frames)
    sample = None

    if mode == "edit_postprocessing":
        if len(temporal_upsampling) > 0 or len(spatial_upsampling) > 0 or film_grain_intensity > 0:                
            send_cmd("progress", [0, get_latest_status(state,"Upsampling" if len(temporal_upsampling) > 0 or len(spatial_upsampling) > 0 else "Adding Film Grain"  )])
            sample = get_resampled_video(video_source, 0, max_source_video_frames, fps)
            sample = sample.float().div_(127.5).sub_(1.).permute(-1,0,1,2)
            frames_count = sample.shape[1] 

        output_fps  = round(fps)
        if len(temporal_upsampling) > 0:
            sample, previous_last_frame, output_fps = perform_temporal_upsampling(sample, None, temporal_upsampling, fps)
            configs["temporal_upsampling"] = temporal_upsampling
            frames_count = sample.shape[1] 


        if len(spatial_upsampling) > 0:
            sample = perform_spatial_upsampling(sample, spatial_upsampling )
            configs["spatial_upsampling"] = spatial_upsampling

        if film_grain_intensity > 0:
            from postprocessing.film_grain import add_film_grain
            sample = add_film_grain(sample, film_grain_intensity, film_grain_saturation) 
            configs["film_grain_intensity"] = film_grain_intensity
            configs["film_grain_saturation"] = film_grain_saturation
    else:
        output_fps  = round(fps)

    any_mmaudio = MMAudio_setting != 0 and server_config.get("mmaudio_enabled", 0) != 0 and frames_count >=output_fps
    if any_mmaudio: download_mmaudio()

    tmp_path = None
    any_change = False
    if sample != None:
        video_path =get_available_filename(save_path, video_source, "_tmp") if any_mmaudio or has_already_audio else get_available_filename(save_path, video_source, "_post")  
        save_video( tensor=sample[None], save_file=video_path, fps=output_fps, nrow=1, normalize=True, value_range=(-1, 1), codec_type= server_config.get("video_output_codec", None), container=server_config.get("video_container", "mp4"))

        if any_mmaudio or has_already_audio: tmp_path = video_path
        any_change = True
    else:
        video_path = video_source

    repeat_no = 0
    extra_generation = 0
    initial_total_windows = 0
    any_change_initial = any_change
    while not gen.get("abort", False): 
        any_change = any_change_initial
        extra_generation += gen.get("extra_orders",0)
        gen["extra_orders"] = 0
        total_generation = repeat_generation + extra_generation
        gen["total_generation"] = total_generation         
        if repeat_no >= total_generation: break
        repeat_no +=1
        gen["repeat_no"] = repeat_no
        suffix =  "" if "_post" in video_source else "_post"

        if audio_source is not None:
            audio_prompt_type = configs.get("audio_prompt_type", "")
            if not "T" in audio_prompt_type:audio_prompt_type += "T"
            configs["audio_prompt_type"] = audio_prompt_type
            any_change = True

        if any_mmaudio:
            send_cmd("progress", [0, get_latest_status(state,"MMAudio Soundtrack Generation")])
            from postprocessing.mmaudio.mmaudio import video_to_audio
            new_video_path = get_available_filename(save_path, video_source, suffix)
            video_to_audio(video_path, prompt = MMAudio_prompt, negative_prompt = MMAudio_neg_prompt, seed = seed, num_steps = 25, cfg_strength = 4.5, duration= frames_count /output_fps, save_path = new_video_path , persistent_models = server_config.get("mmaudio_enabled", 0) == 2, verboseLevel = verbose_level)
            configs["MMAudio_setting"] = MMAudio_setting
            configs["MMAudio_prompt"] = MMAudio_prompt
            configs["MMAudio_neg_prompt"] = MMAudio_neg_prompt
            configs["MMAudio_seed"] = seed
            any_change = True
        elif len(audio_tracks) > 0:
            # combine audio files and new video file
            new_video_path = get_available_filename(save_path, video_source, suffix)
            combine_video_with_audio_tracks(video_path, audio_tracks, new_video_path, audio_metadata=audio_metadata)
        else:
            new_video_path = video_path
        if tmp_path != None:
            os.remove(tmp_path)

        if any_change:
            if mode == "edit_remux":
                print(f"Remuxed Video saved to Path: "+ new_video_path)
            else:
                print(f"Postprocessed video saved to Path: "+ new_video_path)
            with lock:
                file_list.append(new_video_path)
                file_settings_list.append(configs)

            if configs is not None:
                metadata_mode = metadata_choice if metadata_choice is not None else server_config.get("metadata_type", "metadata")
                metadata_logger = get_notifications_logger() if metadata_mode == "metadata" else None
                temp_images_path: Optional[str] = None
                embedded_images = None
                if metadata_mode == "metadata" and server_config.get("embed_source_images", False):
                    from shared.utils.video_metadata import extract_source_images

                    temp_images_path = get_available_filename(save_path, video_source, force_extension=".temp")
                    embedded_images = extract_source_images(video_source, temp_images_path)
                if metadata_mode == "json":
                    json_path = Path(new_video_path).with_suffix(".json")
                    with open(json_path, "w", encoding="utf-8") as writer:
                        json.dump(configs, writer, indent=4)
                elif metadata_mode == "metadata":
                    metadata_config = _resolve_metadata_config("video", embedded_images=embedded_images)
                    write_metadata_bundle(
                        new_video_path,
                        configs,
                        config=metadata_config,
                        logger=metadata_logger,
                    )
                if temp_images_path and os.path.isdir(temp_images_path):
                    shutil.rmtree(temp_images_path, ignore_errors=True)
            send_cmd("output")
            seed = set_seed(-1)
    if has_already_audio:
        cleanup_temp_audio_files(audio_tracks)
    notifier.reset_progress(state)

def get_overridden_attention(model_type):
    model_def = get_model_def(model_type)
    override_attention = model_def.get("attention", None)
    if override_attention is None: return None
    gpu_version = gpu_major * 10 + gpu_minor
    attention_list = match_nvidia_architecture(override_attention, gpu_version) 
    if len(attention_list ) == 0: return None
    override_attention = attention_list[0]
    if override_attention is not None and override_attention not in attention_modes_supported: return None
    return override_attention

def get_transformer_loras(model_type):
    model_def = get_model_def(model_type)
    transformer_loras_filenames = get_model_recursive_prop(model_type, "loras", return_list=True)
    lora_dir = get_lora_dir(model_type)
    transformer_loras_filenames = [ os.path.join(lora_dir, os.path.basename(filename)) for filename in transformer_loras_filenames]
    transformer_loras_multipliers = get_model_recursive_prop(model_type, "loras_multipliers", return_list=True) + [1.] * len(transformer_loras_filenames)
    transformer_loras_multipliers = transformer_loras_multipliers[:len(transformer_loras_filenames)]
    return transformer_loras_filenames, transformer_loras_multipliers

class DynamicClass:
    def __init__(self, **kwargs):
        self._data = {}
        # Preassign default properties from kwargs
        for key, value in kwargs.items():
            self._data[key] = value
    
    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            if not hasattr(self, '_data'):
                super().__setattr__('_data', {})
            self._data[name] = value
    
    def assign(self, **kwargs):
        """Assign multiple properties at once"""
        for key, value in kwargs.items():
            self._data[key] = value
        return self  # For method chaining
    
    def update(self, dict):
        """Alias for assign() - more dict-like"""
        return self.assign(**dict)


def process_prompt_enhancer(prompt_enhancer, original_prompts,  image_start, original_image_refs, is_image, audio_only, seed ):

    text_encoder_max_tokens = 256
    from models.ltx_video.utils.prompt_enhance_utils import generate_cinematic_prompt
    prompt_images = []
    if "I" in prompt_enhancer:
        if image_start != None:
            if not isinstance(image_start, list): image_start= [image_start] 
            prompt_images += image_start
        if original_image_refs != None:
            prompt_images += original_image_refs[:1]
    prompt_images = [Image.open(img) if isinstance(img,str) else img for img in prompt_images]
    if len(original_prompts) == 0 and not "T" in prompt_enhancer:
        return None
    else:
        from shared.utils.utils import seed_everything
        seed = seed_everything(seed)
        # for i, original_prompt in enumerate(original_prompts):
        prompts = generate_cinematic_prompt(
            prompt_enhancer_image_caption_model,
            prompt_enhancer_image_caption_processor,
            prompt_enhancer_llm_model,
            prompt_enhancer_llm_tokenizer,
            original_prompts if "T" in prompt_enhancer else ["an image"],
            prompt_images if len(prompt_images) > 0 else None,
            video_prompt = not is_image,
            text_prompt = audio_only,
            max_new_tokens=text_encoder_max_tokens,
        )
        return prompts

def get_outpainting_dims(video_guide_outpainting):
    return None if video_guide_outpainting== None or len(video_guide_outpainting) == 0 or video_guide_outpainting == "0 0 0 0" or video_guide_outpainting.startswith("#") else [int(v) for v in video_guide_outpainting.split(" ")] 

def generate_video(
    task,
    send_cmd,
    image_mode,
    prompt,
    negative_prompt,    
    resolution,
    video_length,
    batch_size,
    seed,
    force_fps,
    num_inference_steps,
    guidance_scale,
    guidance2_scale,
    guidance3_scale,
    switch_threshold,
    switch_threshold2,
    guidance_phases,
    model_switch_phase,
    audio_guidance_scale,
    flow_shift,
    sample_solver,
    embedded_guidance_scale,
    repeat_generation,
    multi_prompts_gen_type,
    multi_images_gen_type,
    skip_steps_cache_type,
    skip_steps_multiplier,
    skip_steps_start_step_perc,    
    activated_loras,
    loras_multipliers,
    image_prompt_type,
    image_start,
    image_end,
    model_mode,
    video_source,
    keep_frames_video_source,
    video_prompt_type,
    image_refs,
    frames_positions,
    video_guide,
    image_guide,
    keep_frames_video_guide,
    denoising_strength,
    video_guide_outpainting,
    video_mask,
    image_mask,
    control_net_weight,
    control_net_weight2,
    control_net_weight_alt,
    mask_expand,
    audio_guide,
    audio_guide2,
    audio_source,
    audio_prompt_type,
    speakers_locations,
    sliding_window_size,
    sliding_window_overlap,
    sliding_window_color_correction_strength,
    sliding_window_overlap_noise,
    sliding_window_discard_last_frames,
    image_refs_relative_size,
    remove_background_images_ref,
    temporal_upsampling,
    spatial_upsampling,
    film_grain_intensity,
    film_grain_saturation,
    MMAudio_setting,
    MMAudio_prompt,
    MMAudio_neg_prompt,    
    RIFLEx_setting,
    NAG_scale,
    NAG_tau,
    NAG_alpha,
    slg_switch,
    slg_layers,    
    slg_start_perc,
    slg_end_perc,
    apg_switch,
    cfg_star_switch,
    cfg_zero_step,
    prompt_enhancer,
    min_frames_if_references,
    override_profile,
    pace,
    exaggeration,
    temperature,
    state,
    model_type,
    model_filename,
    mode,
    plugin_data=None,
    notifier=None,
    callback_builder=None,
):



    def remove_temp_filenames(temp_filenames_list):
        for temp_filename in temp_filenames_list: 
            if temp_filename!= None and os.path.isfile(temp_filename):
                os.remove(temp_filename)

    process_map_outside_mask = { "Y" : "depth", "W": "scribble", "X": "inpaint", "Z": "flow"}
    process_map_video_guide = { "P": "pose", "D" : "depth", "S": "scribble", "E": "canny", "L": "flow", "C": "gray", "M": "inpaint", "U": "identity"}
    processes_names = { "pose": "Open Pose", "depth": "Depth Mask", "scribble" : "Shapes", "flow" : "Flow Map", "gray" : "Gray Levels", "inpaint" : "Inpaint Mask", "identity": "Identity Mask", "raw" : "Raw Format", "canny" : "Canny Edges"}

    global wan_model, offloadobj, reload_needed
    gen = get_gen_info(state)
    torch.set_grad_enabled(False) 
    if mode.startswith("edit_"):
        edit_video(send_cmd, state, mode, video_source, seed, temporal_upsampling, spatial_upsampling, film_grain_intensity, film_grain_saturation, MMAudio_setting, MMAudio_prompt, MMAudio_neg_prompt, repeat_generation, audio_source)
        return
    with lock:
        file_list = gen["file_list"]
        file_settings_list = gen["file_settings_list"]
        audio_file_list = gen["audio_file_list"]
        audio_file_settings_list = gen["audio_file_settings_list"]

    if notifier is None:
        notifier = create_legacy_notifier(
            clear_status_fn=clear_status,
            update_task_thumbnails_fn=update_task_thumbnails,
            notification_sound_module=notification_sound,
            server_config=server_config,
        )

    model_def = get_model_def(model_type) 
    is_image = image_mode > 0
    audio_only = model_def.get("audio_only", False)

    set_video_prompt_type = model_def.get("set_video_prompt_type", None)
    if set_video_prompt_type is not None:
        video_prompt_type = add_to_sequence(video_prompt_type, set_video_prompt_type)
    if is_image:
        if min_frames_if_references >= 1000:
            video_length = min_frames_if_references - 1000
        else:
            video_length = min_frames_if_references if "I" in video_prompt_type or "V" in video_prompt_type else 1 
    else:
        batch_size = 1
    temp_filenames_list = []

    if image_guide is not None and isinstance(image_guide, Image.Image):
        video_guide = image_guide
        image_guide = None

    if image_mask is not None and isinstance(image_mask, Image.Image):
        video_mask = image_mask
        image_mask = None

    if model_def.get("no_background_removal", False): remove_background_images_ref = 0
    
    base_model_type = get_base_model_type(model_type)
    model_handler = get_model_handler(base_model_type)
    block_size = model_handler.get_vae_block_size(base_model_type) if hasattr(model_handler, "get_vae_block_size") else 16

    if "P" in preload_model_policy and not "U" in preload_model_policy:
        while wan_model == None:
            time.sleep(1)
        
    if model_type !=  transformer_type or reload_needed or override_profile>0 and override_profile != loaded_profile or override_profile<0 and default_profile != loaded_profile:
        wan_model = None
        release_model()
        send_cmd("status", f"Loading model {get_model_name(model_type)}...")
        wan_model, offloadobj = load_models(model_type, override_profile)
        send_cmd("status", "Model loaded")
        reload_needed=  False
    overridden_attention = get_overridden_attention(model_type)
    # if overridden_attention is not None and overridden_attention !=  attention_mode: print(f"Attention mode has been overriden to {overridden_attention} for model type '{model_type}'")
    attn = overridden_attention if overridden_attention is not None else attention_mode
    if attn == "auto":
        attn = get_auto_attention()
    elif not attn in attention_modes_supported:
        send_cmd("info", f"You have selected attention mode '{attention_mode}'. However it is not installed or supported on your system. You should either install it or switch to the default 'sdpa' attention.")
        send_cmd("exit")
        return
    
    width, height = resolution.split("x")
    width, height = int(width) // block_size *  block_size, int(height) // block_size *  block_size
    default_image_size = (height, width)

    if slg_switch == 0:
        slg_layers = None

    offload.shared_state["_attention"] =  attn
    device_mem_capacity = torch.cuda.get_device_properties(0).total_memory / 1048576
    if  hasattr(wan_model, "vae") and hasattr(wan_model.vae, "get_VAE_tile_size"):
        VAE_tile_size = wan_model.vae.get_VAE_tile_size(vae_config, device_mem_capacity, server_config.get("vae_precision", "16") == "32")
    else:
        VAE_tile_size = None

    trans = get_transformer_model(wan_model)
    trans2 = get_transformer_model(wan_model, 2)
    audio_sampling_rate = 16000

    prompts = prompt.split("\n")
    prompts = [part for part in prompts if len(prompt)>0]
    parsed_keep_frames_video_source= max_source_video_frames if len(keep_frames_video_source) ==0 else int(keep_frames_video_source) 
    transformer_loras_filenames, transformer_loras_multipliers  = get_transformer_loras(model_type)
    if guidance_phases < 1: guidance_phases = 1
    if transformer_loras_filenames != None:
        loras_list_mult_choices_nums, loras_slists, errors =  parse_loras_multipliers(transformer_loras_multipliers, len(transformer_loras_filenames), num_inference_steps, nb_phases = guidance_phases )
        if len(errors) > 0: raise Exception(f"Error parsing Transformer Loras: {errors}")
        loras_selected = transformer_loras_filenames 

    if hasattr(wan_model, "get_loras_transformer"):
        extra_loras_transformers, extra_loras_multipliers = wan_model.get_loras_transformer(get_model_recursive_prop, **locals())
        loras_list_mult_choices_nums, loras_slists, errors =  parse_loras_multipliers(extra_loras_multipliers, len(extra_loras_transformers), num_inference_steps, nb_phases = guidance_phases, merge_slist= loras_slists )
        if len(errors) > 0: raise Exception(f"Error parsing Extra Transformer Loras: {errors}")
        loras_selected += extra_loras_transformers 

    loras = state["loras"]
    if len(loras) > 0:
        loras_list_mult_choices_nums, loras_slists, errors =  parse_loras_multipliers(loras_multipliers, len(activated_loras), num_inference_steps, nb_phases = guidance_phases, merge_slist= loras_slists )
        if len(errors) > 0: raise Exception(f"Error parsing Loras: {errors}")
        lora_dir = get_lora_dir(model_type)
        errors = check_loras_exist(model_type, activated_loras, True, send_cmd)
        if len(errors) > 0 : raise GenerationError(errors)
        loras_selected += [ os.path.join(lora_dir, os.path.basename(lora)) for lora in activated_loras]

    if hasattr(wan_model, "get_trans_lora"):
        trans_lora, trans2_lora = wan_model.get_trans_lora()
    else:     
        trans_lora, trans2_lora = trans, trans2

    if len(loras_selected) > 0:
        pinnedLora = loaded_profile !=5  # and transformer_loras_filenames == None False # # # 
        split_linear_modules_map = getattr(trans,"split_linear_modules_map", None)
        offload.load_loras_into_model(trans_lora, loras_selected, loras_list_mult_choices_nums, activate_all_loras=True, preprocess_sd=get_loras_preprocessor(trans, base_model_type), pinnedLora=pinnedLora, split_linear_modules_map = split_linear_modules_map) 
        errors = trans_lora._loras_errors
        if len(errors) > 0:
            error_files = [msg for _ ,  msg  in errors]
            raise GenerationError("Error while loading Loras: " + ", ".join(error_files))
        if trans2_lora is not None: 
            offload.sync_models_loras(trans_lora, trans2_lora)
        
    seed = None if seed == -1 else seed
    # negative_prompt = "" # not applicable in the inference
    original_filename = model_filename 
    model_filename = get_model_filename(base_model_type)  

    _, _, latent_size = get_model_min_frames_and_step(model_type)  
    video_length = (video_length -1) // latent_size * latent_size + 1
    if sliding_window_size !=0:
        sliding_window_size = (sliding_window_size -1) // latent_size * latent_size + 1
    if sliding_window_overlap !=0:
        sliding_window_overlap = (sliding_window_overlap -1) // latent_size * latent_size + 1
    if sliding_window_discard_last_frames !=0:
        sliding_window_discard_last_frames = sliding_window_discard_last_frames // latent_size * latent_size 

    current_video_length = video_length
    # VAE Tiling
    device_mem_capacity = torch.cuda.get_device_properties(None).total_memory / 1048576
    guide_inpaint_color = model_def.get("guide_inpaint_color", 127.5)
    extract_guide_from_window_start = model_def.get("extract_guide_from_window_start", False) 
    fantasy = base_model_type in ["fantasy"]
    multitalk = model_def.get("multitalk_class", False)

    if "B" in audio_prompt_type or "X" in audio_prompt_type:
        from models.wan.multitalk.multitalk import parse_speakers_locations
        speakers_bboxes, error = parse_speakers_locations(speakers_locations)
    else:
        speakers_bboxes = None        
    if "L" in image_prompt_type:
        if len(file_list)>0:
            video_source = file_list[-1]
        else:
            mp4_files = glob.glob(os.path.join(save_path, "*.mp4"))
            video_source = max(mp4_files, key=os.path.getmtime) if mp4_files else None                            
    fps = 1 if is_image else get_computed_fps(force_fps, base_model_type , video_guide, video_source )
    control_audio_tracks = source_audio_tracks = source_audio_metadata = []
    if "R" in audio_prompt_type and video_guide is not None and MMAudio_setting == 0 and not any_letters(audio_prompt_type, "ABX"):
        control_audio_tracks, _  = extract_audio_tracks(video_guide)
    if video_source is not None:
        source_audio_tracks, source_audio_metadata = extract_audio_tracks(video_source)
        video_fps, _, _, video_frames_count = get_video_info(video_source)
        video_source_duration = video_frames_count / video_fps
    else:
        video_source_duration = 0

    reset_control_aligment = "T" in video_prompt_type

    if test_any_sliding_window(model_type) :
        if video_source is not None:
            current_video_length +=  sliding_window_overlap - 1
        sliding_window = current_video_length > sliding_window_size
        reuse_frames = min(sliding_window_size - latent_size, sliding_window_overlap) 
    else:
        sliding_window = False
        sliding_window_size = current_video_length
        reuse_frames = 0

    original_image_refs = image_refs
    image_refs = None if image_refs is None else ([] + image_refs) # work on a copy as it is going to be modified
    # image_refs = None
    # nb_frames_positions= 0
    # Output Video Ratio Priorities:
    # Source Video or Start Image > Control Video > Image Ref (background or positioned frames only) >  UI Width, Height
    # Image Ref (non background and non positioned frames) are boxed in a white canvas in order to keep their own width/height ratio
    frames_to_inject = []
    any_background_ref  = 0
    if "K" in video_prompt_type: 
        any_background_ref = 2 if model_def.get("all_image_refs_are_background_ref", False) else 1

    outpainting_dims = get_outpainting_dims(video_guide_outpainting)
    fit_canvas = server_config.get("fit_canvas", 0)
    fit_crop = fit_canvas == 2
    if fit_crop and outpainting_dims is not None:
        fit_crop = False
        fit_canvas = 0

    joint_pass = boost ==1 #and profile != 1 and profile != 3  
    
    skip_steps_cache = None if len(skip_steps_cache_type) == 0 else DynamicClass(cache_type = skip_steps_cache_type) 

    if skip_steps_cache != None:
        skip_steps_cache.update({     
        "multiplier" : skip_steps_multiplier,
        "start_step":  int(skip_steps_start_step_perc*num_inference_steps/100)
        })
        model_handler.set_cache_parameters(skip_steps_cache_type, base_model_type, model_def, locals(), skip_steps_cache)
        if skip_steps_cache_type == "mag":
            def_mag_ratios = model_def.get("magcache_ratios", None) if model_def != None else None
            if def_mag_ratios is not None: skip_steps_cache.def_mag_ratios = def_mag_ratios
        elif skip_steps_cache_type == "tea":
            def_tea_coefficients = model_def.get("teacache_coefficients", None) if model_def != None else None
            if def_tea_coefficients is not None: skip_steps_cache.coefficients = def_tea_coefficients
        else:
            raise Exception(f"unknown cache type {skip_steps_cache_type}")
    trans.cache = skip_steps_cache
    if trans2 is not None: trans2.cache = skip_steps_cache
    face_arc_embeds = None
    src_ref_images = src_ref_masks = None
    output_new_audio_data = None
    output_new_audio_filepath = None
    original_audio_guide = audio_guide
    original_audio_guide2 = audio_guide2
    audio_proj_split = None
    audio_proj_full = None
    audio_scale = None
    audio_context_lens = None
    if audio_guide != None:
        from models.wan.fantasytalking.infer import parse_audio
        from preprocessing.extract_vocals import get_vocals
        import librosa
        duration = librosa.get_duration(path=audio_guide)
        combination_type = "add"
        clean_audio_files = "V" in audio_prompt_type
        if audio_guide2 is not None:
            duration2 = librosa.get_duration(path=audio_guide2)
            if "C" in audio_prompt_type: duration += duration2
            else: duration = min(duration, duration2)
            combination_type = "para" if "P" in audio_prompt_type else "add" 
            if clean_audio_files:
                audio_guide = get_vocals(original_audio_guide, get_available_filename(save_path, audio_guide, "_clean", ".wav"))
                audio_guide2 = get_vocals(original_audio_guide2, get_available_filename(save_path, audio_guide2, "_clean2", ".wav"))
                temp_filenames_list += [audio_guide, audio_guide2]
        else:
            if "X" in audio_prompt_type: 
                # dual speaker, voice separation
                from preprocessing.speakers_separator import extract_dual_audio
                combination_type = "para"
                if args.save_speakers:
                    audio_guide, audio_guide2  = "speaker1.wav", "speaker2.wav"
                else:
                    audio_guide, audio_guide2  = get_available_filename(save_path, audio_guide, "_tmp1", ".wav"),  get_available_filename(save_path, audio_guide, "_tmp2", ".wav")
                    temp_filenames_list +=   [audio_guide, audio_guide2]                  
                if clean_audio_files:
                    clean_audio_guide = get_vocals(original_audio_guide, get_available_filename(save_path, original_audio_guide, "_clean", ".wav"))
                    temp_filenames_list += [clean_audio_guide]
                extract_dual_audio(clean_audio_guide if clean_audio_files else original_audio_guide, audio_guide, audio_guide2)

            elif clean_audio_files:
                # Single Speaker
                audio_guide = get_vocals(original_audio_guide, get_available_filename(save_path, audio_guide, "_clean", ".wav"))
                temp_filenames_list += [audio_guide]

            output_new_audio_filepath = original_audio_guide

        current_video_length = min(int(fps * duration //latent_size) * latent_size + latent_size + 1, current_video_length)
        if fantasy:
            # audio_proj_split_full, audio_context_lens_full = parse_audio(audio_guide, num_frames= max_source_video_frames, fps= fps,  padded_frames_for_embeddings= (reuse_frames if reset_control_aligment else 0), device= processing_device  )
            audio_scale = 1.0
        elif multitalk:
            from models.wan.multitalk.multitalk import get_full_audio_embeddings
            # pad audio_proj_full if aligned to beginning of window to simulate source window overlap
            min_audio_duration =  current_video_length/fps if reset_control_aligment else video_source_duration + current_video_length/fps
            audio_proj_full, output_new_audio_data = get_full_audio_embeddings(audio_guide1 = audio_guide, audio_guide2= audio_guide2, combination_type= combination_type , num_frames= max_source_video_frames, sr= audio_sampling_rate, fps =fps, padded_frames_for_embeddings = (reuse_frames if reset_control_aligment else 0), min_audio_duration = min_audio_duration) 
            if output_new_audio_data is not None: # not none if modified
                if clean_audio_files: # need to rebuild the sum of audios with original audio
                    _, output_new_audio_data = get_full_audio_embeddings(audio_guide1 = original_audio_guide, audio_guide2= original_audio_guide2, combination_type= combination_type , num_frames= max_source_video_frames, sr= audio_sampling_rate, fps =fps, padded_frames_for_embeddings = (reuse_frames if reset_control_aligment else 0), min_audio_duration = min_audio_duration, return_sum_only= True) 
                output_new_audio_filepath=  None # need to build original speaker track if it changed size (due to padding at the end) or if it has been combined

    seed = set_seed(seed)

    torch.set_grad_enabled(False) 
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(image_save_path, exist_ok=True)
    gc.collect()
    torch.cuda.empty_cache()
    wan_model._interrupt = False
    abort = False
    if gen.get("abort", False):
        return 
    # gen["abort"] = False
    gen["prompt"] = prompt    
    repeat_no = 0
    extra_generation = 0
    initial_total_windows = 0
    discard_last_frames = sliding_window_discard_last_frames
    default_requested_frames_to_generate = current_video_length
    nb_frames_positions = 0
    if sliding_window:
        initial_total_windows= compute_sliding_window_no(default_requested_frames_to_generate, sliding_window_size, discard_last_frames, reuse_frames) 
        current_video_length = sliding_window_size
    else:
        initial_total_windows = 1

    first_window_video_length = current_video_length
    original_prompts = prompts.copy()
    gen["sliding_window"] = sliding_window 
    while not abort: 
        extra_generation += gen.get("extra_orders",0)
        gen["extra_orders"] = 0
        total_generation = repeat_generation + extra_generation
        gen["total_generation"] = total_generation     
        gen["header_text"] = ""    
        if repeat_no >= total_generation: break
        repeat_no +=1
        gen["repeat_no"] = repeat_no
        src_video = src_video2 = src_mask = src_mask2 = src_faces = sparse_video_image = None
        prefix_video = pre_video_frame = None
        source_video_overlap_frames_count = 0 # number of frames overalapped in source video for first window
        source_video_frames_count = 0  # number of frames to use in source video (processing starts source_video_overlap_frames_count frames before )
        frames_already_processed = None
        overlapped_latents = None
        context_scale = None
        window_no = 0
        extra_windows = 0
        guide_start_frame = 0 # pos of of first control video frame of current window  (reuse_frames later than the first processed frame)
        keep_frames_parsed = [] # aligned to the first control frame of current window (therefore ignore previous reuse_frames)
        pre_video_guide = None # reuse_frames of previous window
        image_size = default_image_size #  default frame dimensions for budget until it is change due to a resize
        sample_fit_canvas = fit_canvas
        current_video_length = first_window_video_length
        gen["extra_windows"] = 0
        gen["total_windows"] = 1
        gen["window_no"] = 1
        num_frames_generated = 0 # num of new frames created (lower than the number of frames really processed due to overlaps and discards)
        requested_frames_to_generate = default_requested_frames_to_generate # num  of num frames to create (if any source window this num includes also the overlapped source window frames)
        start_time = time.time()
        if prompt_enhancer_image_caption_model != None and prompt_enhancer !=None and len(prompt_enhancer)>0 and server_config.get("enhancer_mode", 0) == 0:
            send_cmd("progress", [0, get_latest_status(state, "Enhancing Prompt")])
            enhanced_prompts = process_prompt_enhancer(prompt_enhancer, original_prompts,  image_start, original_image_refs, is_image, audio_only, seed )
            if enhanced_prompts is not None:
                print(f"Enhanced prompts: {enhanced_prompts}" )
                task["prompt"] = "\n".join(["!enhanced!"] + enhanced_prompts)
                send_cmd("output")
                prompt = enhanced_prompts[0]            
                abort = gen.get("abort", False)

        while not abort:
            enable_RIFLEx = RIFLEx_setting == 0 and current_video_length > (6* get_model_fps(base_model_type)+1) or RIFLEx_setting == 1
            prompt =  prompts[window_no] if window_no < len(prompts) else prompts[-1]
            new_extra_windows = gen.get("extra_windows",0)
            gen["extra_windows"] = 0
            extra_windows += new_extra_windows
            requested_frames_to_generate +=  new_extra_windows * (sliding_window_size - discard_last_frames - reuse_frames)
            sliding_window = sliding_window  or extra_windows > 0
            if sliding_window and window_no > 0:
                # num_frames_generated -= reuse_frames
                if (requested_frames_to_generate - num_frames_generated) <  latent_size:
                    break
                current_video_length = min(sliding_window_size, ((requested_frames_to_generate - num_frames_generated + reuse_frames + discard_last_frames) // latent_size) * latent_size + 1 )

            total_windows = initial_total_windows + extra_windows
            gen["total_windows"] = total_windows
            if window_no >= total_windows:
                break
            window_no += 1
            gen["window_no"] = window_no
            return_latent_slice = None 
            if reuse_frames > 0:                
                return_latent_slice = slice(-(reuse_frames - 1 + discard_last_frames ) // latent_size - 1, None if discard_last_frames == 0 else -(discard_last_frames // latent_size) )
            refresh_preview  = {"image_guide" : image_guide, "image_mask" : image_mask} if image_mode >= 1 else {}

            image_start_tensor = image_end_tensor = None
            if window_no == 1 and (video_source is not None or image_start is not None):
                if image_start is not None:
                    image_start_tensor, new_height, new_width = calculate_dimensions_and_resize_image(image_start, height, width, sample_fit_canvas, fit_crop, block_size = block_size)
                    if fit_crop: refresh_preview["image_start"] = image_start_tensor 
                    image_start_tensor = convert_image_to_tensor(image_start_tensor)
                    pre_video_guide =  prefix_video = image_start_tensor.unsqueeze(1)
                else:
                    prefix_video  = preprocess_video(width=width, height=height,video_in=video_source, max_frames= parsed_keep_frames_video_source , start_frame = 0, fit_canvas= sample_fit_canvas, fit_crop = fit_crop, target_fps = fps, block_size = block_size )
                    prefix_video  = prefix_video.permute(3, 0, 1, 2)
                    prefix_video  = prefix_video.float().div_(127.5).sub_(1.) # c, f, h, w
                    if fit_crop or "L" in image_prompt_type: refresh_preview["video_source"] = convert_tensor_to_image(prefix_video, 0) 

                    new_height, new_width = prefix_video.shape[-2:]                    
                    pre_video_guide =  prefix_video[:, -reuse_frames:]
                pre_video_frame = convert_tensor_to_image(prefix_video[:, -1])
                source_video_overlap_frames_count = pre_video_guide.shape[1]
                source_video_frames_count = prefix_video.shape[1]
                if sample_fit_canvas != None: 
                    image_size  = pre_video_guide.shape[-2:]
                    sample_fit_canvas = None
                guide_start_frame =  prefix_video.shape[1]
            if image_end is not None:
                image_end_list=  image_end if isinstance(image_end, list) else [image_end]
                if len(image_end_list) >= window_no:
                    new_height, new_width = image_size                    
                    image_end_tensor, _, _ = calculate_dimensions_and_resize_image(image_end_list[window_no-1], new_height, new_width, sample_fit_canvas, fit_crop, block_size = block_size)
                    # image_end_tensor =image_end_list[window_no-1].resize((new_width, new_height), resample=Image.Resampling.LANCZOS) 
                    refresh_preview["image_end"] = image_end_tensor 
                    image_end_tensor = convert_image_to_tensor(image_end_tensor)
                image_end_list= None
            window_start_frame = guide_start_frame - (reuse_frames if window_no > 1 else source_video_overlap_frames_count)
            guide_end_frame = guide_start_frame + current_video_length - (source_video_overlap_frames_count if window_no == 1 else reuse_frames)
            alignment_shift = source_video_frames_count if reset_control_aligment else 0
            aligned_guide_start_frame = guide_start_frame - alignment_shift
            aligned_guide_end_frame = guide_end_frame - alignment_shift
            aligned_window_start_frame = window_start_frame - alignment_shift  
            if fantasy and audio_guide is not None:
                audio_proj_split , audio_context_lens = parse_audio(audio_guide, start_frame = aligned_window_start_frame, num_frames= current_video_length, fps= fps,  device= processing_device  )
            if multitalk:
                from models.wan.multitalk.multitalk import get_window_audio_embeddings
                # special treatment for start frame pos when alignement to first frame requested as otherwise the start frame number will be negative due to overlapped frames (has been previously compensated later with padding)
                audio_proj_split = get_window_audio_embeddings(audio_proj_full, audio_start_idx= aligned_window_start_frame + (source_video_overlap_frames_count if reset_control_aligment else 0 ), clip_length = current_video_length)

            if repeat_no == 1 and window_no == 1 and image_refs is not None and len(image_refs) > 0:
                frames_positions_list = []
                if frames_positions is not None and len(frames_positions)> 0:
                    positions = frames_positions.replace(","," ").split(" ")
                    cur_end_pos =  -1 + (source_video_frames_count - source_video_overlap_frames_count)
                    last_frame_no = requested_frames_to_generate + source_video_frames_count - source_video_overlap_frames_count
                    joker_used = False
                    project_window_no = 1
                    for pos in positions :
                        if len(pos) > 0:
                            if pos in ["L", "l"]:
                                cur_end_pos += sliding_window_size if project_window_no > 1 else current_video_length 
                                if cur_end_pos >= last_frame_no-1 and not joker_used:
                                    joker_used = True
                                    cur_end_pos = last_frame_no -1
                                project_window_no += 1
                                frames_positions_list.append(cur_end_pos)
                                cur_end_pos -= sliding_window_discard_last_frames + reuse_frames
                            else:
                                frames_positions_list.append(int(pos)-1 + alignment_shift)
                    frames_positions_list = frames_positions_list[:len(image_refs)]
                nb_frames_positions = len(frames_positions_list) 
                if nb_frames_positions > 0:
                    frames_to_inject = [None] * (max(frames_positions_list) + 1)
                    for i, pos in enumerate(frames_positions_list):
                        frames_to_inject[pos] = image_refs[i] 


            video_guide_processed = video_mask_processed = video_guide_processed2 = video_mask_processed2 = sparse_video_image = None
            if video_guide is not None:
                keep_frames_parsed_full, error = parse_keep_frames_video_guide(keep_frames_video_guide, source_video_frames_count -source_video_overlap_frames_count + requested_frames_to_generate)
                if len(error) > 0:
                    raise GenerationError(f"invalid keep frames {keep_frames_video_guide}")
                guide_frames_extract_start = aligned_window_start_frame if extract_guide_from_window_start else aligned_guide_start_frame
                extra_control_frames = model_def.get("extra_control_frames", 0)
                if extra_control_frames > 0 and aligned_guide_start_frame >= extra_control_frames: guide_frames_extract_start -= extra_control_frames
                        
                keep_frames_parsed = [True] * -guide_frames_extract_start if guide_frames_extract_start  <0 else []
                keep_frames_parsed += keep_frames_parsed_full[max(0, guide_frames_extract_start): aligned_guide_end_frame ] 
                guide_frames_extract_count = len(keep_frames_parsed)

                # Extract Faces to video
                if "B" in video_prompt_type:
                    send_cmd("progress", [0, get_latest_status(state, "Extracting Face Movements")])
                    src_faces = extract_faces_from_video_with_mask(video_guide, video_mask, max_frames= guide_frames_extract_count, start_frame= guide_frames_extract_start, size= 512, target_fps = fps)
                    if src_faces is not None and src_faces.shape[1] < current_video_length:
                        src_faces = torch.cat([src_faces, torch.full( (3, current_video_length - src_faces.shape[1], 512, 512 ), -1, dtype = src_faces.dtype, device= src_faces.device) ], dim=1)

                # Sparse Video to Video
                sparse_video_image = None
                if "R" in video_prompt_type:
                    sparse_video_image = get_video_frame(video_guide, aligned_guide_start_frame, return_last_if_missing = True, target_fps = fps, return_PIL = True)

                # Generic Video Preprocessing
                process_outside_mask = process_map_outside_mask.get(filter_letters(video_prompt_type, "YWX"), None)
                preprocess_type, preprocess_type2 =  "raw", None 
                for process_num, process_letter in enumerate( filter_letters(video_prompt_type, video_guide_processes)):
                    if process_num == 0:
                        preprocess_type = process_map_video_guide.get(process_letter, "raw")
                    else:
                        preprocess_type2 = process_map_video_guide.get(process_letter, None)
                status_info = "Extracting " + processes_names[preprocess_type]
                extra_process_list = ([] if preprocess_type2==None else [preprocess_type2]) + ([] if process_outside_mask==None or process_outside_mask == preprocess_type else [process_outside_mask])
                if len(extra_process_list) == 1:
                    status_info += " and " + processes_names[extra_process_list[0]]
                elif len(extra_process_list) == 2:
                    status_info +=  ", " + processes_names[extra_process_list[0]] + " and " + processes_names[extra_process_list[1]]
                context_scale = [control_net_weight /2, control_net_weight2 /2] if preprocess_type2 is not None else [control_net_weight]
                if not (preprocess_type == "identity" and preprocess_type2 is None and video_mask is None):send_cmd("progress", [0, get_latest_status(state, status_info)])
                inpaint_color = 0 if preprocess_type=="pose" and process_outside_mask == "inpaint" else guide_inpaint_color
                video_guide_processed, video_mask_processed = preprocess_video_with_mask(video_guide if sparse_video_image is None else sparse_video_image, video_mask, height=image_size[0], width = image_size[1], max_frames= guide_frames_extract_count, start_frame = guide_frames_extract_start, fit_canvas = sample_fit_canvas, fit_crop = fit_crop, target_fps = fps,  process_type = preprocess_type, expand_scale = mask_expand, RGB_Mask = True, negate_mask = "N" in video_prompt_type, process_outside_mask = process_outside_mask, outpainting_dims = outpainting_dims, proc_no =1, inpaint_color =inpaint_color, block_size = block_size, to_bbox = "H" in video_prompt_type )
                if preprocess_type2 != None:
                    video_guide_processed2, video_mask_processed2 = preprocess_video_with_mask(video_guide, video_mask, height=image_size[0], width = image_size[1], max_frames= guide_frames_extract_count, start_frame = guide_frames_extract_start, fit_canvas = sample_fit_canvas, fit_crop = fit_crop, target_fps = fps,  process_type = preprocess_type2, expand_scale = mask_expand, RGB_Mask = True, negate_mask = "N" in video_prompt_type, process_outside_mask = process_outside_mask, outpainting_dims = outpainting_dims, proc_no =2, block_size = block_size, to_bbox = "H" in video_prompt_type  )

                if video_guide_processed is not None  and sample_fit_canvas is not None:
                    image_size = video_guide_processed.shape[-2:]
                    sample_fit_canvas = None

            if window_no == 1 and image_refs is not None and len(image_refs) > 0:
                if sample_fit_canvas is not None and (nb_frames_positions > 0 or "K" in video_prompt_type) :
                    from shared.utils.utils import get_outpainting_full_area_dimensions
                    w, h = image_refs[0].size
                    if outpainting_dims != None:
                        h, w = get_outpainting_full_area_dimensions(h,w, outpainting_dims)
                    image_size = calculate_new_dimensions(height, width, h, w, fit_canvas)
                sample_fit_canvas = None
                if repeat_no == 1:
                    if fit_crop:
                        if any_background_ref == 2:
                            end_ref_position = len(image_refs)
                        elif any_background_ref == 1:
                            end_ref_position = nb_frames_positions + 1
                        else:
                            end_ref_position = nb_frames_positions 
                        for i, img in enumerate(image_refs[:end_ref_position]):
                            image_refs[i] = rescale_and_crop(img, default_image_size[1], default_image_size[0])
                        refresh_preview["image_refs"] = image_refs

                    if len(image_refs) > nb_frames_positions:
                        src_ref_images = image_refs[nb_frames_positions:]
                        if "Q" in video_prompt_type:
                            from preprocessing.arc.face_encoder import FaceEncoderArcFace, get_landmarks_from_image
                            image_pil = src_ref_images[-1]
                            face_encoder = FaceEncoderArcFace()
                            face_encoder.init_encoder_model(processing_device)
                            face_arc_embeds = face_encoder(image_pil, need_proc=True, landmarks=get_landmarks_from_image(image_pil))
                            face_arc_embeds = face_arc_embeds.squeeze(0).cpu()
                            face_encoder = image_pil = None
                            gc.collect()
                            torch.cuda.empty_cache()

                        if remove_background_images_ref > 0:
                            send_cmd("progress", [0, get_latest_status(state, "Removing Images References Background")])

                        src_ref_images, src_ref_masks  = resize_and_remove_background(src_ref_images , image_size[1], image_size[0],
                                                                                        remove_background_images_ref > 0, any_background_ref, 
                                                                                        fit_into_canvas= model_def.get("fit_into_canvas_image_refs", 1),
                                                                                        block_size=block_size,
                                                                                        outpainting_dims =outpainting_dims,
                                                                                        background_ref_outpainted = model_def.get("background_ref_outpainted", True),
                                                                                        return_tensor= model_def.get("return_image_refs_tensor", False),
                                                                                        ignore_last_refs =model_def.get("no_processing_on_last_images_refs",0))

            frames_to_inject_parsed = frames_to_inject[ window_start_frame if extract_guide_from_window_start else guide_start_frame: guide_end_frame]
            if video_guide is not None or len(frames_to_inject_parsed) > 0 or model_def.get("forced_guide_mask_inputs", False): 
                any_mask = video_mask is not None or model_def.get("forced_guide_mask_inputs", False)
                any_guide_padding = model_def.get("pad_guide_video", False)
                from shared.utils.utils import prepare_video_guide_and_mask
                src_videos, src_masks = prepare_video_guide_and_mask(   [video_guide_processed] + ([] if video_guide_processed2 is None else [video_guide_processed2]), 
                                                                        [video_mask_processed] + ([] if video_guide_processed2 is None else [video_mask_processed2]),
                                                                        None if extract_guide_from_window_start or model_def.get("dont_cat_preguide", False) or sparse_video_image is not None else pre_video_guide, 
                                                                        image_size, current_video_length, latent_size,
                                                                        any_mask, any_guide_padding, guide_inpaint_color, 
                                                                        keep_frames_parsed, frames_to_inject_parsed , outpainting_dims)
                video_guide_processed = video_guide_processed2 = video_mask_processed = video_mask_processed2 = None
                if len(src_videos) == 1:
                    src_video, src_video2, src_mask, src_mask2 = src_videos[0], None, src_masks[0], None 
                else:
                    src_video, src_video2 = src_videos 
                    src_mask, src_mask2 = src_masks 
                src_videos = src_masks = None
                if src_video is None:
                    abort = True 
                    break
                if src_faces is not None:
                    if src_faces.shape[1] < src_video.shape[1]:
                        src_faces = torch.concat( [src_faces,  src_faces[:, -1:].repeat(1, src_video.shape[1] - src_faces.shape[1], 1,1)], dim =1)
                    else:
                        src_faces = src_faces[:, :src_video.shape[1]]
                if video_guide is not None or len(frames_to_inject_parsed) > 0:
                    if args.save_masks:
                        if src_video is not None: 
                            save_video( src_video, "masked_frames.mp4", fps)
                            if any_mask: save_video( src_mask, "masks.mp4", fps, value_range=(0, 1))
                        if src_video2 is not None: 
                            save_video( src_video2, "masked_frames2.mp4", fps)
                            if any_mask: save_video( src_mask2, "masks2.mp4", fps, value_range=(0, 1))
                if video_guide is not None:                        
                    preview_frame_no = 0 if extract_guide_from_window_start or model_def.get("dont_cat_preguide", False) or sparse_video_image is not None else (guide_start_frame - window_start_frame) 
                    preview_frame_no = min(src_video.shape[1] -1, preview_frame_no)
                    refresh_preview["video_guide"] = convert_tensor_to_image(src_video, preview_frame_no)
                    if src_video2 is not None:
                        refresh_preview["video_guide"] = [refresh_preview["video_guide"], convert_tensor_to_image(src_video2, preview_frame_no)] 
                    if src_mask is not None and video_mask is not None:                        
                        refresh_preview["video_mask"] = convert_tensor_to_image(src_mask, preview_frame_no, mask_levels = True)

            if src_ref_images is not None or nb_frames_positions:
                if len(frames_to_inject_parsed):
                    new_image_refs = [convert_tensor_to_image(src_video, frame_no + (0 if extract_guide_from_window_start else (aligned_guide_start_frame - aligned_window_start_frame)) ) for frame_no, inject in enumerate(frames_to_inject_parsed) if inject]
                else:
                    new_image_refs = []
                if src_ref_images is not None:
                    new_image_refs +=  [convert_tensor_to_image(img) if torch.is_tensor(img) else img for img in src_ref_images  ]
                refresh_preview["image_refs"] = new_image_refs
                new_image_refs = None

            if len(refresh_preview) > 0:
                preview_inputs = prepare_preview_inputs(
                    {
                        "image_start": image_start,
                        "video_source": video_source,
                        "image_end": image_end,
                        "video_guide": video_guide,
                        "image_guide": image_guide,
                        "video_mask": video_mask,
                        "image_mask": image_mask,
                        "image_refs": image_refs,
                    },
                    refresh_preview,
                )
                notifier.refresh_preview(task, preview_inputs)
                send_cmd("output")

            if window_no ==  1:                
                conditioning_latents_size = ( (source_video_overlap_frames_count-1) // latent_size) + 1 if source_video_overlap_frames_count > 0 else 0
            else:
                conditioning_latents_size = ( (reuse_frames-1) // latent_size) + 1

            status = get_latest_status(state)
            gen["progress_status"] = status
            progress_phase = "Generation Audio" if audio_only else "Encoding Prompt"
            gen["progress_phase"] = (progress_phase , -1 )
            callback = None
            if callback_builder is not None:
                try:
                    callback = callback_builder(state, send_cmd, status, num_inference_steps)
                except Exception as callback_exc:
                    notify_warning(f"Progress callback builder failed ({callback_exc}); disabling progress updates.")
                    callback = None
            progress_args = [0, merge_status_context(status, progress_phase )]
            send_cmd("progress", progress_args)

            if skip_steps_cache !=  None:
                skip_steps_cache.update({
                "num_steps" : num_inference_steps,                
                "skipped_steps" : 0,
                "previous_residual": None,
                "previous_modulated_input":  None,
                })
            # samples = torch.empty( (1,2)) #for testing
            # if False:
            def set_header_text(txt):
                gen["header_text"] = txt
                send_cmd("output")

            generated_audio = None
            try:
                samples = wan_model.generate(
                    input_prompt = prompt,
                    image_start = image_start_tensor,  
                    image_end = image_end_tensor,
                    input_frames = src_video,
                    input_frames2 = src_video2,
                    input_ref_images=  src_ref_images,
                    input_ref_masks = src_ref_masks,
                    input_masks = src_mask,
                    input_masks2 = src_mask2,
                    input_video= pre_video_guide,
                    input_faces = src_faces,
                    denoising_strength=denoising_strength,
                    prefix_frames_count = source_video_overlap_frames_count if window_no <= 1 else reuse_frames,
                    frame_num= (current_video_length // latent_size)* latent_size + 1,
                    batch_size = batch_size,
                    height = image_size[0],
                    width = image_size[1],
                    fit_into_canvas = fit_canvas,
                    shift=flow_shift,
                    sample_solver=sample_solver,
                    sampling_steps=num_inference_steps,
                    guide_scale=guidance_scale,
                    guide2_scale = guidance2_scale,
                    guide3_scale = guidance3_scale,
                    switch_threshold = switch_threshold, 
                    switch2_threshold = switch_threshold2,
                    guide_phases= guidance_phases,
                    model_switch_phase = model_switch_phase,
                    embedded_guidance_scale=embedded_guidance_scale,
                    n_prompt=negative_prompt,
                    seed=seed,
                    callback=callback,
                    enable_RIFLEx = enable_RIFLEx,
                    VAE_tile_size = VAE_tile_size,
                    joint_pass = joint_pass,
                    slg_layers = slg_layers,
                    slg_start = slg_start_perc/100,
                    slg_end = slg_end_perc/100,
                    apg_switch = apg_switch,
                    cfg_star_switch = cfg_star_switch,
                    cfg_zero_step = cfg_zero_step,
                    audio_cfg_scale= audio_guidance_scale,
                    audio_guide=audio_guide,
                    audio_guide2=audio_guide2,
                    audio_proj= audio_proj_split,
                    audio_scale= audio_scale,
                    audio_context_lens= audio_context_lens,
                    context_scale = context_scale,
                    control_scale_alt = control_net_weight_alt,
                    model_mode = model_mode,
                    causal_block_size = 5,
                    causal_attention = True,
                    fps = fps,
                    overlapped_latents = overlapped_latents,
                    return_latent_slice= return_latent_slice,
                    overlap_noise = sliding_window_overlap_noise,
                    overlap_size = sliding_window_overlap,
                    color_correction_strength = sliding_window_color_correction_strength,
                    conditioning_latents_size = conditioning_latents_size,
                    keep_frames_parsed = keep_frames_parsed,
                    model_filename = model_filename,
                    model_type = base_model_type,
                    loras_slists = loras_slists,
                    NAG_scale = NAG_scale,
                    NAG_tau = NAG_tau,
                    NAG_alpha = NAG_alpha,
                    speakers_bboxes =speakers_bboxes,
                    image_mode =  image_mode,
                    video_prompt_type= video_prompt_type,
                    window_no = window_no, 
                    offloadobj = offloadobj,
                    set_header_text= set_header_text,
                    pre_video_frame = pre_video_frame,
                    original_input_ref_images = original_image_refs[nb_frames_positions:] if original_image_refs is not None else [],
                    image_refs_relative_size = image_refs_relative_size,
                    outpainting_dims = outpainting_dims,
                    face_arc_embeds = face_arc_embeds,
                    exaggeration=exaggeration,
                    pace=pace,
                    temperature=temperature,
                )
            except Exception as e:
                if len(control_audio_tracks) > 0 or len(source_audio_tracks) > 0:
                    cleanup_temp_audio_files(control_audio_tracks + source_audio_tracks)
                remove_temp_filenames(temp_filenames_list)
                clear_gen_cache()
                offloadobj.unload_all()
                trans.cache = None 
                if trans2 is not None: 
                    trans2.cache = None 
                offload.unload_loras_from_model(trans_lora)
                if trans2_lora is not None: 
                    offload.unload_loras_from_model(trans2_lora)
                skip_steps_cache = None
                # if compile:
                #     cache_size = torch._dynamo.config.cache_size_limit                                      
                #     torch.compiler.reset()
                #     torch._dynamo.config.cache_size_limit = cache_size

                gc.collect()
                torch.cuda.empty_cache()
                s = str(e)
                keyword_list = {"CUDA out of memory" : "VRAM", "Tried to allocate":"VRAM", "CUDA error: out of memory": "RAM", "CUDA error: too many resources requested": "RAM"}
                crash_type = ""
                for keyword, tp  in keyword_list.items():
                    if keyword in s:
                        crash_type = tp 
                        break
                state["prompt"] = ""
                if crash_type == "VRAM":
                    new_error = "The generation of the video has encountered an error: it is likely that you have unsufficient VRAM and you should therefore reduce the video resolution or its number of frames."
                elif crash_type == "RAM":
                    new_error = "The generation of the video has encountered an error: it is likely that you have unsufficient RAM and / or Reserved RAM allocation should be reduced using 'perc_reserved_mem_max' or using a different Profile."
                else:
                    new_error = (
                        "The generation of the video has encountered an error, "
                        f"please check your terminal for more information. '{s}'"
                    )
                tb = traceback.format_exc().split('\n')[:-1] 
                print('\n'.join(tb))
                send_cmd("error", new_error)
                notifier.reset_progress(state)
                return

            if skip_steps_cache != None :
                skip_steps_cache.previous_residual = None
                skip_steps_cache.previous_modulated_input = None
                print(f"Skipped Steps:{skip_steps_cache.skipped_steps}/{skip_steps_cache.num_steps}" )
            BGRA_frames = None
            if samples != None:
                if isinstance(samples, dict):
                    overlapped_latents = samples.get("latent_slice", None)
                    BGRA_frames = samples.get("BGRA_frames", None)
                    generated_audio = samples.get("audio", generated_audio)
                    samples = samples.get("x", None)
                if samples is not None:
                    samples = samples.to("cpu")
            clear_gen_cache()
            offloadobj.unload_all()
            gc.collect()
            torch.cuda.empty_cache()

            # time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss")
            # save_prompt = "_in_" + original_prompts[0]
            # file_name = f"{time_flag}_seed{seed}_{sanitize_file_name(save_prompt[:50]).strip()}.mp4"
            # sample = samples.cpu()
            # cache_video( tensor=sample[None].clone(), save_file=os.path.join(save_path, file_name), fps=16, nrow=1, normalize=True, value_range=(-1, 1))
            if samples == None:
                abort = True
                state["prompt"] = ""
                send_cmd("output")  
            else:
                sample = samples.cpu()
                if generated_audio is not None:
                    output_new_audio_data = generated_audio
                abort = not (is_image or audio_only) and sample.shape[1] < current_video_length    
                # if True: # for testing
                #     torch.save(sample, "output.pt")
                # else:
                #     sample =torch.load("output.pt")
                if gen.get("extra_windows",0) > 0:
                    sliding_window = True 
                if sliding_window :
                    # guide_start_frame = guide_end_frame
                    guide_start_frame += current_video_length
                    if discard_last_frames > 0:
                        sample = sample[: , :-discard_last_frames]
                        guide_start_frame -= discard_last_frames
                    if reuse_frames == 0:
                        pre_video_guide =  sample[:,max_source_video_frames :].clone()
                    else:
                        pre_video_guide =  sample[:, -reuse_frames:].clone()


                if prefix_video != None and window_no == 1:
                    # remove source video overlapped frames at the beginning of the generation
                    sample = torch.cat([ prefix_video[:, :-source_video_overlap_frames_count], sample], dim = 1)
                    guide_start_frame -= source_video_overlap_frames_count 
                elif sliding_window and window_no > 1 and reuse_frames > 0:
                    # remove sliding window overlapped frames at the beginning of the generation
                    sample = sample[: , reuse_frames:]
                    guide_start_frame -= reuse_frames 

                num_frames_generated = guide_start_frame - (source_video_frames_count - source_video_overlap_frames_count) 

                if len(temporal_upsampling) > 0 or len(spatial_upsampling) > 0:                
                    send_cmd("progress", [0, get_latest_status(state,"Upsampling")])
                
                output_fps  = fps
                if len(temporal_upsampling) > 0:
                    sample, previous_last_frame, output_fps = perform_temporal_upsampling(sample, previous_last_frame if sliding_window and window_no > 1 else None, temporal_upsampling, fps)

                if len(spatial_upsampling) > 0:
                    sample = perform_spatial_upsampling(sample, spatial_upsampling )
                if film_grain_intensity> 0:
                    from postprocessing.film_grain import add_film_grain
                    sample = add_film_grain(sample, film_grain_intensity, film_grain_saturation) 
                if sliding_window :
                    if frames_already_processed == None:
                        frames_already_processed = sample
                    else:
                        sample = torch.cat([frames_already_processed, sample], dim=1)
                    frames_already_processed = sample

                time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss")
                save_prompt = original_prompts[0]
                if audio_only:
                    extension = "wav"
                elif is_image:
                    extension = "jpg"
                else:
                    container = server_config.get("video_container", "mp4")
                    extension = container 

                file_name = f"{time_flag}_seed{seed}_{sanitize_file_name(truncate_for_filesystem(save_prompt)).strip()}.{extension}"
                video_path = os.path.join(save_path, file_name)
                any_mmaudio = MMAudio_setting != 0 and server_config.get("mmaudio_enabled", 0) != 0 and sample.shape[1] >=fps
                if BGRA_frames is not None:
                    from models.wan.alpha.utils import write_zip_file
                    write_zip_file(os.path.splitext(video_path)[0] + ".zip", BGRA_frames)
                    BGRA_frames = None 
                if audio_only:
                    import soundfile as sf
                    audio_path = os.path.join(image_save_path, file_name)
                    sf.write(audio_path, sample.squeeze(0), wan_model.sr)
                    video_path= audio_path                      
                elif is_image:    
                    image_path = os.path.join(image_save_path, file_name)
                    sample =  sample.transpose(1,0)  #c f h w -> f c h w 
                    new_image_path = []
                    for no, img in enumerate(sample):  
                        img_path = os.path.splitext(image_path)[0] + ("" if no==0 else f"_{no}") + ".jpg" 
                        new_image_path.append(save_image(img, save_file = img_path, quality = server_config.get("image_output_codec", None)))

                    video_path= new_image_path
                elif len(control_audio_tracks) > 0 or len(source_audio_tracks) > 0 or output_new_audio_filepath is not None or any_mmaudio or output_new_audio_data is not None or audio_source is not None:
                    video_path = os.path.join(save_path, file_name)
                    save_path_tmp = video_path.rsplit('.', 1)[0] + f"_tmp.{container}"
                    save_video( tensor=sample[None], save_file=save_path_tmp, fps=output_fps, nrow=1, normalize=True, value_range=(-1, 1), codec_type = server_config.get("video_output_codec", None), container=container)
                    output_new_audio_temp_filepath = None
                    new_audio_added_from_audio_start =  reset_control_aligment or generated_audio is not None # if not beginning of audio will be skipped
                    source_audio_duration = source_video_frames_count / fps
                    if any_mmaudio:
                        send_cmd("progress", [0, get_latest_status(state,"MMAudio Soundtrack Generation")])
                        from postprocessing.mmaudio.mmaudio import video_to_audio
                        output_new_audio_filepath = output_new_audio_temp_filepath = get_available_filename(save_path, f"tmp{time_flag}.wav" )
                        video_to_audio(save_path_tmp, prompt = MMAudio_prompt, negative_prompt = MMAudio_neg_prompt, seed = seed, num_steps = 25, cfg_strength = 4.5, duration= sample.shape[1] /fps, save_path = output_new_audio_filepath, persistent_models = server_config.get("mmaudio_enabled", 0) == 2, audio_file_only = True, verboseLevel = verbose_level)
                        new_audio_added_from_audio_start =  False
                    elif audio_source is not None:
                        output_new_audio_filepath = audio_source
                        new_audio_added_from_audio_start =  True
                    elif output_new_audio_data is not None:
                        import soundfile as sf
                        output_new_audio_filepath = output_new_audio_temp_filepath = get_available_filename(save_path, f"tmp{time_flag}.wav" )
                        sf.write(output_new_audio_filepath, output_new_audio_data, audio_sampling_rate)                       
                    if output_new_audio_filepath is not None:
                        new_audio_tracks = [output_new_audio_filepath]
                    else:
                        new_audio_tracks = control_audio_tracks

                    combine_and_concatenate_video_with_audio_tracks(video_path, save_path_tmp,  source_audio_tracks, new_audio_tracks, source_audio_duration, audio_sampling_rate, new_audio_from_start = new_audio_added_from_audio_start, source_audio_metadata= source_audio_metadata, verbose = verbose_level>=2 )
                    os.remove(save_path_tmp)
                    if output_new_audio_temp_filepath is not None: os.remove(output_new_audio_temp_filepath)

                else:
                    save_video( tensor=sample[None], save_file=video_path, fps=output_fps, nrow=1, normalize=True, value_range=(-1, 1),  codec_type= server_config.get("video_output_codec", None), container= container)

                end_time = time.time()

                inputs = get_function_arguments(generate_video, locals())
                inputs.pop("send_cmd")
                inputs.pop("task")
                inputs.pop("mode")
                inputs["model_type"] = model_type
                inputs["model_filename"] = original_filename
                if is_image:
                    inputs["image_quality"] = server_config.get("image_output_codec", None)
                else:
                    inputs["video_quality"] = server_config.get("video_output_codec", None)

                modules = get_model_recursive_prop(model_type, "modules", return_list= True)
                if len(modules) > 0 : inputs["modules"] = modules
                if len(transformer_loras_filenames) > 0:
                    inputs.update({
                    "transformer_loras_filenames" : transformer_loras_filenames,
                    "transformer_loras_multipliers" : transformer_loras_multipliers
                    })
                embedded_images = (
                    {img_name: inputs[img_name] for img_name in image_names_list}
                    if server_config.get("embed_source_images", False)
                    else None
                )
                configs = prepare_inputs_dict("metadata", inputs, model_type)
                if sliding_window: configs["window_no"] = window_no
                configs["prompt"] = "\n".join(original_prompts)
                if prompt_enhancer_image_caption_model != None and prompt_enhancer !=None and len(prompt_enhancer)>0:
                    configs["enhanced_prompt"] = "\n".join(prompts)
                configs["generation_time"] = round(end_time-start_time)
                # if is_image: configs["is_image"] = True
                metadata_mode = metadata_choice if metadata_choice is not None else server_config.get("metadata_type", "metadata")
                video_path = [video_path] if not isinstance(video_path, list) else video_path
                metadata_logger = get_notifications_logger() if metadata_mode == "metadata" else None
                for no, path in enumerate(video_path):
                    if metadata_mode == "json":
                        json_path = Path(path).with_suffix(".json")
                        with open(json_path, "w", encoding="utf-8") as writer:
                            json.dump(configs, writer, indent=4)
                    elif metadata_mode == "metadata":
                        if audio_only:
                            config = _resolve_metadata_config("audio")
                        elif is_image:
                            config = _resolve_metadata_config("image")
                        else:
                            config = _resolve_metadata_config("video", embedded_images=embedded_images)
                        write_metadata_bundle(
                            path,
                            configs,
                            config=config,
                            logger=metadata_logger,
                        )
                    if audio_only:
                        print(f"New audio file saved to Path: "+ path)
                    elif is_image:
                        print(f"New image saved to Path: "+ path)
                    else:
                        print(f"New video saved to Path: "+ path)
                    with lock:
                        if audio_only:
                            audio_file_list.append(path)
                            audio_file_settings_list.append(configs if no > 0 else configs.copy())
                        else:
                            file_list.append(path)
                            file_settings_list.append(configs if no > 0 else configs.copy())
                        gen["last_was_audio"] = audio_only

                embedded_images = None
                notifier.notify_video_ready(video_path=video_path)

                send_cmd("output")

        seed = set_seed(-1)
    notifier.reset_progress(state)
    trans.cache = None
    offload.unload_loras_from_model(trans_lora)
    if not trans2_lora is None:
        offload.unload_loras_from_model(trans2_lora)

    if not trans2 is None:
       trans2.cache = None
 
    if len(control_audio_tracks) > 0 or len(source_audio_tracks) > 0:
        cleanup_temp_audio_files(control_audio_tracks + source_audio_tracks)

    remove_temp_filenames(temp_filenames_list)

def generate_preview(model_type, latents):
    import einops
    if latents is None: return None
    model_handler = get_model_handler(model_type)
    base_model_type = get_base_model_type(model_type)
    if hasattr(model_handler, "get_rgb_factors"):
        latent_rgb_factors, latent_rgb_factors_bias = model_handler.get_rgb_factors(base_model_type )
    else:
        return None
    if latent_rgb_factors is None: return None
    latents = latents.unsqueeze(0) 
    nb_latents = latents.shape[2]
    latents_to_preview = 4
    latents_to_preview = min(nb_latents, latents_to_preview)
    skip_latent =  nb_latents / latents_to_preview
    latent_no = 0
    selected_latents = []
    while latent_no < nb_latents:
        selected_latents.append( latents[:, : , int(latent_no): int(latent_no)+1])
        latent_no += skip_latent 

    latents = torch.cat(selected_latents, dim = 2)
    weight = torch.tensor(latent_rgb_factors, device=latents.device, dtype=latents.dtype).transpose(0, 1)[:, :, None, None, None]
    bias = torch.tensor(latent_rgb_factors_bias, device=latents.device, dtype=latents.dtype)

    images = torch.nn.functional.conv3d(latents, weight, bias=bias, stride=1, padding=0, dilation=1, groups=1)
    images = images.add_(1.0).mul_(127.5)
    images = images.detach().cpu()
    if images.dtype == torch.bfloat16:
        images = images.to(torch.float16)
    images = images.numpy().clip(0, 255).astype(np.uint8)
    images = einops.rearrange(images, 'b c t h w -> (b h) (t w) c')
    h, w, _ = images.shape
    scale = 200 / h
    images= Image.fromarray(images)
    images = images.resize(( int(w*scale),int(h*scale)), resample=Image.Resampling.BILINEAR) 
    return images


def _get_task_inputs_manager() -> TaskInputManager:
    global _task_inputs_manager
    manager = _task_inputs_manager
    if manager is None or manager.server_config is not server_config:
        manager = TaskInputManager(
            server_config=server_config,
            settings_version=settings_version,
            get_model_record=get_model_record,
            get_model_name=get_model_name,
            get_model_def=get_model_def,
            get_base_model_type=get_base_model_type,
            get_model_family=get_model_family,
            test_vace_module=test_vace_module,
            test_class_t2v=test_class_t2v,
            test_any_sliding_window=test_any_sliding_window,
            any_audio_track=any_audio_track,
            get_lora_dir=get_lora_dir,
            settings_loader=None,
            get_settings_file_name=get_settings_file_name,
            set_model_settings=set_model_settings,
            notify_info=notify_info,
            lock=lock,
            get_model_type=get_model_type,
            are_model_types_compatible=are_model_types_compatible,
            get_default_settings=get_default_settings,
            get_model_settings=get_model_settings,
            fix_settings=fix_settings,
            model_types=tuple(model_types),
        )
        _task_inputs_manager = manager
    return manager

def prepare_inputs_dict(target, inputs, model_type = None, model_filename = None ):
    manager = _get_task_inputs_manager()
    return manager.prepare_inputs_dict(target, inputs, model_type=model_type, model_filename=model_filename)


def get_function_arguments(func, locals):
    args_names = list(inspect.signature(func).parameters)
    kwargs = typing.OrderedDict()
    for k in args_names:
        kwargs[k] = locals[k]
    return kwargs


def init_generate(state, input_file_list, last_choice, audio_files_paths, audio_file_selected):
    gen = get_gen_info(state)
    file_list, file_settings_list = get_file_list(state, input_file_list)
    set_file_choice(gen, file_list, last_choice)
    audio_file_list, audio_file_settings_list = get_file_list(state, unpack_audio_list(audio_files_paths), audio_files=True)
    set_file_choice(gen, audio_file_list, audio_file_selected, audio_files=True)

    return get_unique_id(), ""


def eject_video_from_gallery(*_args, **_kwargs):
    raise RuntimeError('Interactive galleries were removed in the headless build.')

def eject_audio_from_gallery(*_args, **_kwargs):
    raise RuntimeError('Interactive galleries were removed in the headless build.')

def add_videos_to_gallery(*_args, **_kwargs):
    raise RuntimeError('Interactive galleries were removed in the headless build.')

def get_model_settings(state, model_type):
    all_settings = state.get("all_settings", None)    
    return None if all_settings == None else all_settings.get(model_type, None)

def set_model_settings(state, model_type, settings):
    all_settings = state.get("all_settings", None)    
    if all_settings == None:
        all_settings = {}
        state["all_settings"] = all_settings
    all_settings[model_type] = settings
    
def collect_current_model_settings(state):
    model_filename = state["model_filename"]
    model_type = state["model_type"]
    settings = get_model_settings(state, model_type)
    settings["state"] = state
    settings = prepare_inputs_dict("metadata", settings)
    settings["model_filename"] = model_filename 
    settings["model_type"] = model_type 
    return settings 

def export_settings(state):
    model_type = state["model_type"]
    text = json.dumps(collect_current_model_settings(state), indent=4)
    text_base64 = base64.b64encode(text.encode('utf8')).decode('utf-8')
    return text_base64, sanitize_file_name(model_type + "_" + datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss") + ".json")


def extract_and_apply_source_images(file_path, current_settings):
    from shared.utils.video_metadata import extract_source_images
    if not os.path.isfile(file_path): return 0
    extracted_files = extract_source_images(file_path)            
    if not extracted_files: return 0
    applied_count = 0
    for name in image_names_list:
        if name in extracted_files:
            img = extracted_files[name]
            img = img if isinstance(img,list) else [img]
            applied_count += len(img)
            current_settings[name] = img        
    return applied_count


def reset_settings(state):
    model_type = state["model_type"]
    ui_defaults = get_default_settings(model_type)
    set_model_settings(state, model_type, ui_defaults)
    notify_info(f"Default Settings have been Restored")
    return str(time.time())

def save_inputs(
            target,
            image_mask_guide,
            lset_name,
            image_mode,
            prompt,
            negative_prompt,
            resolution,
            video_length,
            batch_size,
            seed,
            force_fps,
            num_inference_steps,
            guidance_scale,
            guidance2_scale,
            guidance3_scale,
            switch_threshold,
            switch_threshold2,
            guidance_phases,
            model_switch_phase,
            audio_guidance_scale,
            flow_shift,
            sample_solver,
            embedded_guidance_scale,
            repeat_generation,
            multi_prompts_gen_type,
            multi_images_gen_type,
            skip_steps_cache_type,
            skip_steps_multiplier,
            skip_steps_start_step_perc,
            loras_choices,
            loras_multipliers,
            image_prompt_type,
            image_start,
            image_end,
            model_mode,
            video_source,
            keep_frames_video_source,
            video_guide_outpainting,
            video_prompt_type,
            image_refs,
            frames_positions,
            video_guide,
            image_guide,
            keep_frames_video_guide,
            denoising_strength,
            video_mask,
            image_mask,
            control_net_weight,
            control_net_weight2,
            control_net_weight_alt,
            mask_expand,
            audio_guide,
            audio_guide2,
            audio_source,
            audio_prompt_type,
            speakers_locations,
            sliding_window_size,
            sliding_window_overlap,
            sliding_window_color_correction_strength,
            sliding_window_overlap_noise,
            sliding_window_discard_last_frames,
            image_refs_relative_size,
            remove_background_images_ref,
            temporal_upsampling,
            spatial_upsampling,
            film_grain_intensity,
            film_grain_saturation,
            MMAudio_setting,
            MMAudio_prompt,
            MMAudio_neg_prompt,
            RIFLEx_setting,
            NAG_scale,
            NAG_tau,
            NAG_alpha,
            slg_switch,
            slg_layers,
            slg_start_perc,
            slg_end_perc,
            apg_switch,
            cfg_star_switch,
            cfg_zero_step,
            prompt_enhancer,
            min_frames_if_references,
            override_profile,
            pace,
            exaggeration,
            temperature,
            mode,
            state,
            plugin_data,
):
    payload = SaveInputsPayload(
        image_mask_guide=image_mask_guide,
        lset_name=lset_name,
        image_mode=image_mode,
        prompt=prompt,
        negative_prompt=negative_prompt,
        resolution=resolution,
        video_length=video_length,
        batch_size=batch_size,
        seed=seed,
        force_fps=force_fps,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        guidance2_scale=guidance2_scale,
        guidance3_scale=guidance3_scale,
        switch_threshold=switch_threshold,
        switch_threshold2=switch_threshold2,
        guidance_phases=guidance_phases,
        model_switch_phase=model_switch_phase,
        audio_guidance_scale=audio_guidance_scale,
        flow_shift=flow_shift,
        sample_solver=sample_solver,
        embedded_guidance_scale=embedded_guidance_scale,
        repeat_generation=repeat_generation,
        multi_prompts_gen_type=multi_prompts_gen_type,
        multi_images_gen_type=multi_images_gen_type,
        skip_steps_cache_type=skip_steps_cache_type,
        skip_steps_multiplier=skip_steps_multiplier,
        skip_steps_start_step_perc=skip_steps_start_step_perc,
        loras_choices=loras_choices,
        loras_multipliers=loras_multipliers,
        image_prompt_type=image_prompt_type,
        image_start=image_start,
        image_end=image_end,
        model_mode=model_mode,
        video_source=video_source,
        keep_frames_video_source=keep_frames_video_source,
        video_guide_outpainting=video_guide_outpainting,
        video_prompt_type=video_prompt_type,
        image_refs=image_refs,
        frames_positions=frames_positions,
        video_guide=video_guide,
        image_guide=image_guide,
        keep_frames_video_guide=keep_frames_video_guide,
        denoising_strength=denoising_strength,
        video_mask=video_mask,
        image_mask=image_mask,
        control_net_weight=control_net_weight,
        control_net_weight2=control_net_weight2,
        control_net_weight_alt=control_net_weight_alt,
        mask_expand=mask_expand,
        audio_guide=audio_guide,
        audio_guide2=audio_guide2,
        audio_source=audio_source,
        audio_prompt_type=audio_prompt_type,
        speakers_locations=speakers_locations,
        sliding_window_size=sliding_window_size,
        sliding_window_overlap=sliding_window_overlap,
        sliding_window_color_correction_strength=sliding_window_color_correction_strength,
        sliding_window_overlap_noise=sliding_window_overlap_noise,
        sliding_window_discard_last_frames=sliding_window_discard_last_frames,
        image_refs_relative_size=image_refs_relative_size,
        remove_background_images_ref=remove_background_images_ref,
        temporal_upsampling=temporal_upsampling,
        spatial_upsampling=spatial_upsampling,
        film_grain_intensity=film_grain_intensity,
        film_grain_saturation=film_grain_saturation,
        MMAudio_setting=MMAudio_setting,
        MMAudio_prompt=MMAudio_prompt,
        MMAudio_neg_prompt=MMAudio_neg_prompt,
        RIFLEx_setting=RIFLEx_setting,
        NAG_scale=NAG_scale,
        NAG_tau=NAG_tau,
        NAG_alpha=NAG_alpha,
        slg_switch=slg_switch,
        slg_layers=slg_layers,
        slg_start_perc=slg_start_perc,
        slg_end_perc=slg_end_perc,
        apg_switch=apg_switch,
        cfg_star_switch=cfg_star_switch,
        cfg_zero_step=cfg_zero_step,
        prompt_enhancer=prompt_enhancer,
        min_frames_if_references=min_frames_if_references,
        override_profile=override_profile,
        pace=pace,
        exaggeration=exaggeration,
        temperature=temperature,
        mode=mode,
    )
    request = SaveInputsRequest(
        target=target,
        payload=payload,
        state=state,
        plugin_data=plugin_data or {},
    )
    manager = _get_task_inputs_manager()
    manager.save_inputs(request)



def any_letters(source_str, letters):
    for letter in letters:
        if letter in source_str:
            return True
    return False

def filter_letters(source_str, letters, default= ""):
    ret = ""
    for letter in letters:
        if letter in source_str:
            ret += letter
    if len(ret) == 0:
        return default
    return ret    

def add_to_sequence(source_str, letters):
    ret = source_str
    for letter in letters:
        if not letter in source_str:
            ret += letter
    return ret    

def del_in_sequence(source_str, letters):
    ret = source_str
    for letter in letters:
        if letter in source_str:
            ret = ret.replace(letter, "")
    return ret    

custom_resolutions = None
def get_resolution_choices(current_resolution_choice, model_resolutions= None):
    global custom_resolutions

    resolution_file = "resolutions.json"
    if model_resolutions is not None:
        resolution_choices = model_resolutions
    elif custom_resolutions == None and os.path.isfile(resolution_file) :
        with open(resolution_file, 'r', encoding='utf-8') as f:
            try:
                resolution_choices = json.load(f)
            except Exception as e:
                print(f'Invalid "{resolution_file}" : {e}')
                resolution_choices = None
        if resolution_choices ==  None:
            pass 
        elif not isinstance(resolution_choices, list):
            print(f'"{resolution_file}" should be a list of 2 elements lists ["Label","WxH"]')
            resolution_choices == None
        else:
            for tup in resolution_choices:
                if not isinstance(tup, list) or len(tup) != 2 or not isinstance(tup[0], str) or not isinstance(tup[1], str):
                    print(f'"{resolution_file}" contains an invalid list of two elements: {tup}')
                    resolution_choices == None
                    break
                res_list = tup[1].split("x")
                if len(res_list) != 2 or not is_integer(res_list[0])  or not is_integer(res_list[1]):
                    print(f'"{resolution_file}" contains a resolution value that is not in the format "WxH": {tup[1]}')
                    resolution_choices == None
                    break
        custom_resolutions = resolution_choices
    else:
        resolution_choices = custom_resolutions
    if resolution_choices == None:
        resolution_choices=[
            # 1080p
            ("1920x1088 (16:9)", "1920x1088"),
            ("1088x1920 (9:16)", "1088x1920"),
            ("1920x832 (21:9)", "1920x832"),
            ("832x1920 (9:21)", "832x1920"),
            # 720p
            ("1024x1024 (1:1)", "1024x1024"),
            ("1280x720 (16:9)", "1280x720"),
            ("720x1280 (9:16)", "720x1280"), 
            ("1280x544 (21:9)", "1280x544"),
            ("544x1280 (9:21)", "544x1280"),
            ("1104x832 (4:3)", "1104x832"),
            ("832x1104 (3:4)", "832x1104"),
            ("960x960 (1:1)", "960x960"),
            # 540p
            ("960x544 (16:9)", "960x544"),
            ("544x960 (9:16)", "544x960"),
            # 480p
            ("832x624 (4:3)", "832x624"), 
            ("624x832 (3:4)", "624x832"),
            ("720x720 (1:1)", "720x720"),
            ("832x480 (16:9)", "832x480"),
            ("480x832 (9:16)", "480x832"),
            ("512x512 (1:1)", "512x512"),
        ]

    if current_resolution_choice is not None:
        found = False
        for label, res in resolution_choices:
            if current_resolution_choice == res:
                found = True
                break
        if not found:
            if model_resolutions is None:
                resolution_choices.append( (current_resolution_choice, current_resolution_choice ))
            else:
                current_resolution_choice = resolution_choices[0][1]

    return resolution_choices, current_resolution_choice

group_thresholds = {
    "360p": 320 * 640,    
    "480p": 832 * 624,     
    "540p": 960 * 544,   
    "720p": 1024 * 1024,  
    "1080p": 1920 * 1088,         
    "1440p": 9999 * 9999
}
    
def categorize_resolution(resolution_str):
    width, height = map(int, resolution_str.split('x'))
    pixel_count = width * height
    
    for group in group_thresholds.keys():
        if pixel_count <= group_thresholds[group]:
            return group
    return "1440p"

def group_resolutions(model_def, resolutions, selected_resolution):

    model_resolutions = model_def.get("resolutions", None)
    if model_resolutions is not None:
        selected_group ="Locked"
        available_groups = [selected_group ]
        selected_group_resolutions = model_resolutions
    else:
        grouped_resolutions = {}
        for resolution in resolutions:
            group = categorize_resolution(resolution[1])
            if group not in grouped_resolutions:
                grouped_resolutions[group] = []
            grouped_resolutions[group].append(resolution)
        
        available_groups = [group for group in group_thresholds if group in grouped_resolutions]
    
        selected_group = categorize_resolution(selected_resolution)
        selected_group_resolutions = grouped_resolutions.get(selected_group, [])
        available_groups.reverse()
    return available_groups, selected_group_resolutions, selected_group

def record_last_resolution(state, resolution):

    model_type = state["model_type"]
    model_def = get_model_def(model_type)
    model_resolutions = model_def.get("resolutions", None)
    if model_resolutions is not None: return
    server_config["last_resolution_choice"] = resolution
    selected_group = categorize_resolution(resolution)
    last_resolution_per_group = state["last_resolution_per_group"]
    last_resolution_per_group[selected_group ] = resolution
    server_config["last_resolution_per_group"] = last_resolution_per_group
    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config, indent=4))

def get_max_frames(nb):
    return (nb - 1) * server_config.get("max_frames_multiplier",1) + 1


video_guide_processes = "PEDSLCMU"

memory_profile_choices= [   ("Profile 1, HighRAM_HighVRAM: at least 64 GB of RAM and 24 GB of VRAM, the fastest for short videos with a RTX 3090 / RTX 4090", 1),
                            ("Profile 2, HighRAM_LowVRAM: at least 64 GB of RAM and 12 GB of VRAM, the most versatile profile with high RAM, better suited for RTX 3070/3080/4070/4080 or for RTX 3090 / RTX 4090 with large pictures batches or long videos", 2),
                            ("Profile 3, LowRAM_HighVRAM: at least 32 GB of RAM and 24 GB of VRAM, adapted for RTX 3090 / RTX 4090 with limited RAM for good speed short video",3),
                            ("Profile 4, LowRAM_LowVRAM (Recommended): at least 32 GB of RAM and 12 GB of VRAM, if you have little VRAM or want to generate longer videos",4),
                            ("Profile 5, VerylowRAM_LowVRAM (Fail safe): at least 24 GB of RAM and 10 GB of VRAM, if you don't have much it won't be fast but maybe it will work",5)]

def compute_video_length_label(fps, current_video_length, video_length_locked = None):
    if fps is None:
        ret = f"Number of frames"
    else:
        ret = f"Number of frames ({fps} frames = 1s), current duration: {(current_video_length / fps):.1f}s"
    if video_length_locked is not None:
        ret += ", locked"
    return ret
    

initialize_runtime()


if __name__ == "__main__":
    raise RuntimeError(
        "The Gradio application has been removed. Use the CLI entry point (\"python -m cli.generate\")."
    )
