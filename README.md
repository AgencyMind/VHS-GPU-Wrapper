# ComfyUI VHS GPU Wrapper

A ComfyUI custom node extension that solves multi-GPU device conflicts for VideoHelperSuite (VHS) nodes.

## Problem Solved

In multi-GPU ComfyUI setups using ComfyUI-MultiGPU, VideoHelperSuite nodes can cause "Expected all tensors to be on the same device" errors when used with ControlNet preprocessors. This happens because:

1. VHS nodes load video frames using dynamic device assignment from MultiGPU
2. ControlNet preprocessors (even when wrapped) expect input tensors on specific devices
3. Device mismatch occurs between video loading and preprocessing stages
4. Result: Tensor device conflicts during video processing workflows

## Solution

This extension provides wrapper nodes that ensure consistent device placement during VHS node execution, preventing device conflicts in multi-GPU video processing workflows.

## Installation

### Standard ComfyUI Custom Node Installation

1. Clone to your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/your-username/ComfyUI-VHS-GPU-Wrapper.git
```

2. Restart ComfyUI

3. Wrapper nodes will appear in the Add Node menu under `video/gpu_wrapper`

### Requirements

- ComfyUI with ComfyUI-MultiGPU extension
- VideoHelperSuite extension
- No additional dependencies required

## Usage

### Available Wrapper Nodes

- **Load Video (GPU Wrapper)** - Wraps VHS_LoadVideo
- **Video Combine (GPU Wrapper)** - Wraps VHS_VideoCombine  
- **Load Video Upload (GPU Wrapper)** - Wraps VHS_LoadVideoUpload (if available)

### Drop-in Replacements

Simply replace your existing VHS nodes with the corresponding GPU wrapper versions. All inputs and outputs remain identical, with an additional optional `device` parameter.

**Before:**
```
VHS_LoadVideo → DepthAnything V2 → ControlNet → Generation
```

**After:**
```
Load Video (GPU Wrapper) → DepthAnything V2 (GPU Wrapper) → ControlNet → Generation
```

### Device Selection

Each wrapper node includes a `device` parameter to specify target GPU:
- **Default**: "cuda:0" (or first available CUDA device)
- **Options**: cpu, cuda:0, cuda:1, etc.

Set the same device across your video processing pipeline for consistency.

## Technical Details

### How It Works

The wrapper temporarily overrides `comfy.model_management.get_torch_device()` during VHS node execution and moves all input/output tensors to the target device:

```python
# Save original function
original_get_device = model_management.get_torch_device

# Override with consistent device during execution
model_management.get_torch_device = lambda: torch.device('cuda:0')

try:
    # Execute original VHS node
    result = original_vhs_node.execute(**kwargs)
    # Move output tensors to target device
    moved_result = move_tensors_to_device(result, target_device)
finally:
    # Always restore original function
    model_management.get_torch_device = original_get_device
```

### Device Strategy

- **Target device**: Configurable (default cuda:0)
- **Scope**: Affects both model loading and tensor placement
- **Input handling**: Moves input tensors to target device before processing
- **Output handling**: Ensures output tensors are on target device
- **Restoration**: Original function restored immediately after completion

## Verification

### Check Installation Success

1. **Console**: Look for registration messages:
   ```
   Registered 3 VHS GPU wrapper nodes: ['VHS_LoadVideoWrapper', 'VHS_VideoCombineWrapper', ...]
   ```

2. **Node Menu**: Check `video/gpu_wrapper` category exists

3. **Device Parameter**: Confirm device selection dropdown appears on wrapper nodes

### Test Workflow

Create a test workflow:
```
Load Video (GPU Wrapper) → DepthAnything V2 (GPU Wrapper) → ControlNet → Generation
```

Set both wrappers to the same device (e.g., "cuda:0").

## Troubleshooting

### Import Warnings

If you see warnings like:
```
VHS_LoadVideo not available: No module named 'VideoHelperSuite'
```

This is normal - the extension only wraps VHS nodes that are actually installed.

### Device Conflicts Still Occurring

1. Ensure you're using wrapper versions for **both** VHS and preprocessor nodes
2. Set the same device on all wrapper nodes in your pipeline
3. Check that both ComfyUI-MultiGPU and VideoHelperSuite are active

### Performance

- **None** - Identical performance to original VHS nodes
- **Memory**: Minimal additional GPU memory usage for tensor movement
- **Compatibility**: Works with future VideoHelperSuite updates

## Related Extensions

Works in conjunction with:
- **ComfyUI-GPU-Preprocessor-Wrapper**: Wraps ControlNet preprocessors
- **ComfyUI-MultiGPU**: Provides global multi-GPU memory distribution
- **VideoHelperSuite**: Provides the base video processing nodes

## Contributing

This extension follows the same maintenance-free wrapper pattern as ComfyUI-GPU-Preprocessor-Wrapper, automatically adapting to changes in underlying VHS implementations.

## License

Same as ComfyUI - GPL-3.0