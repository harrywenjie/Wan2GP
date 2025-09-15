# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Running the Application
```bash
# Default text-to-video mode
python wgp.py

# Image-to-video mode
python wgp.py --i2v

# Specific model modes
python wgp.py --t2v-14B          # 14B text-to-video model
python wgp.py --i2v-1-3B         # 1.3B image-to-video model
python wgp.py --vace-1-3B        # VACE ControlNet model

# Network accessible server
python wgp.py --listen --server-port 8080

# Performance optimized
python wgp.py --compile --attention sage2 --profile 3
```

### Environment Management
```bash
# Setup virtual environment
conda create -n wan2gp python=3.10.9
conda activate wan2gp

# Install dependencies
pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 --index-url https://download.pytorch.org/whl/test/cu129
pip install -r requirements.txt

# Update application
git pull
pip install -r requirements.txt
```

### Development Commands
```bash
# Check VRAM/memory usage during testing
python wgp.py --verbose 2

# Debug mode with lora checking
python wgp.py --verbose 2 --check-loras

# Minimal memory setup for debugging
python wgp.py --profile 4 --attention sdpa --perc-reserved-mem-max 0.2
```

## Code Architecture

### Main Application Structure
- **`wgp.py`**: Main entry point containing Gradio interface, model loading, and generation pipeline
- **`models/`**: Model implementations organized by type:
  - `wan/`: Wan 2.1/2.2 video generation models
  - `flux/`: Flux image generation and editing models
  - `hunyuan/`: Hunyuan video models
  - `ltx_video/`: LTX Video models
  - `qwen/`: Qwen image editing models

### Key Shared Components
- **`shared/utils/`**: Core utilities for image/video processing, audio handling, and file operations
- **`shared/attention.py`**: Attention mechanism implementations (SDPA, Sage, Flash)
- **`shared/gradio/`**: Custom Gradio components including AdvancedMediaGallery
- **`preprocessing/`**: Input processing including MatAnyone mask generation
- **`postprocessing/`**: Output processing and enhancement

### Memory Management System
The application uses a sophisticated memory profile system:
- **Profile 1-5**: Different VRAM/RAM usage strategies
- **`mmgp` library**: Handles model offloading and memory optimization
- **Dynamic loading**: Models loaded/unloaded based on memory constraints

### Lora Support Architecture
- Organized by model type in separate directories (`loras/`, `loras_flux/`, `loras_hunyuan/`, etc.)
- Dynamic multiplier system for Lora weights
- Model-specific Lora compatibility checking
- Preset system for saving/loading Lora configurations

### Audio/Video Pipeline
- **Input processing**: Video frame extraction, audio separation, format conversion
- **Generation pipeline**: Model-specific generation with sliding windows for long videos
- **Output processing**: Video encoding, audio mixing, metadata preservation
- **MMAudio integration**: AI-generated soundtracks

### Configuration System
- **`defaults/`**: Default model configurations and finetune definitions
- **`configs/`**: User-customizable settings
- **Dynamic finetunes**: JSON-based model composition system
- **Queue system**: Background generation with autosave/restore

## Development Patterns

### Model Integration
When adding new models, follow the pattern in existing model directories:
1. Create model-specific directory under `models/`
2. Implement generation pipeline with memory profiling
3. Add Gradio interface components
4. Include Lora support if applicable
5. Add to main model selection system

### Memory Optimization
Always consider VRAM constraints:
- Use appropriate memory profiles for target hardware
- Implement model offloading where possible
- Support quantization options for memory-constrained setups
- Test with different attention modes (SDPA for compatibility, Sage for performance)

### Queue and Generation Management
- All generations go through centralized queue system
- Support for batch operations and background processing
- Proper cleanup of temporary files and GPU memory
- Generation metadata preserved for reproducibility

### UI Development
- Uses custom Gradio components in `shared/gradio/`
- Responsive design considerations for different screen sizes
- Real-time progress feedback and statistics
- Advanced/simple mode toggles for different user experience levels