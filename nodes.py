import comfy.model_management as model_management
import torch
import logging

# Set up logging
logger = logging.getLogger(__name__)

def get_device_list():
    """Get list of available devices"""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
    return devices

class VHSMultiGPUWrapper:
    """
    Base wrapper class for VHS nodes that ensures consistent device placement
    and moves input/output tensors to the correct device to prevent multi-GPU device conflicts.
    """
    
    def __init__(self, vhs_node_class):
        self.vhs_node_class = vhs_node_class
    
    def execute(self, device="cuda:0", **kwargs):
        """
        Execute VHS node with device override and tensor movement.
        """
        target_device = torch.device(device)
        
        # Move input tensors to target device
        moved_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                moved_kwargs[key] = value.to(target_device)
                logger.debug(f"Moved input tensor '{key}' from {value.device} to {target_device}")
            else:
                moved_kwargs[key] = value
        
        # Critical: Save original function
        original_get_device = model_management.get_torch_device
        
        try:
            # Temporarily override with consistent device
            model_management.get_torch_device = lambda: target_device
            logger.debug(f"Temporarily overriding get_torch_device() to return: {target_device}")
            
            # Create and execute original VHS node
            vhs_node = self.vhs_node_class()
            function_name = getattr(self.vhs_node_class, 'FUNCTION', 'load_video')
            result = getattr(vhs_node, function_name)(**moved_kwargs)
            
            # Move output tensors to target device
            if isinstance(result, (tuple, list)):
                moved_result = []
                for item in result:
                    if isinstance(item, torch.Tensor):
                        moved_result.append(item.to(target_device))
                        logger.debug(f"Moved output tensor to {target_device}")
                    else:
                        moved_result.append(item)
                result = tuple(moved_result) if isinstance(result, tuple) else moved_result
            elif isinstance(result, torch.Tensor):
                result = result.to(target_device)
                logger.debug(f"Moved single output tensor to {target_device}")
            
            logger.debug(f"Successfully executed {self.vhs_node_class.__name__} on {target_device}")
            return result
            
        except Exception as e:
            logger.error(f"Error in VHSMultiGPUWrapper execution: {e}")
            raise
            
        finally:
            # ALWAYS restore original function, even on exception
            model_management.get_torch_device = original_get_device
            logger.debug("Restored original get_torch_device()")


# Initialize wrapper classes to None
VHS_LoadVideoWrapper = None
VHS_VideoCombineWrapper = None

# Hard-code VHS INPUT_TYPES based on verified source code to avoid import issues
class VHS_LoadVideoWrapper(VHSMultiGPUWrapper):
    def __init__(self):
        pass  # Don't call super().__init__ until runtime
    
    @classmethod
    def INPUT_TYPES(cls):
        devices = get_device_list()
        default_device = "cuda:0" if "cuda:0" in devices else (devices[1] if len(devices) > 1 else devices[0])
        
        return {
            "required": {
                "force_rate": ("FLOAT", {"default": 0, "min": 0, "max": 60, "step": 1}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 999999}),
                "video": ("STRING", {"default": "", "video_upload": True}),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
                "device": (devices, {"default": default_device}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }
        
    RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info")
    FUNCTION = "execute"
    CATEGORY = "video/gpu_wrapper"
    
    def execute(self, device="cuda:0", **kwargs):
        # Find VHS at runtime
        import sys
        modules_copy = dict(sys.modules)
        VHS_LoadVideo = None
        
        for module_name, module in modules_copy.items():
            if hasattr(module, 'NODE_CLASS_MAPPINGS') and module.NODE_CLASS_MAPPINGS:
                if isinstance(module.NODE_CLASS_MAPPINGS, dict):
                    if "VHS_LoadVideo" in module.NODE_CLASS_MAPPINGS:
                        VHS_LoadVideo = module.NODE_CLASS_MAPPINGS["VHS_LoadVideo"]
                        break
        
        if VHS_LoadVideo is None:
            raise RuntimeError("VHS_LoadVideo not found - ensure VideoHelperSuite is installed")
        
        self.vhs_node_class = VHS_LoadVideo
        target_device = torch.device(device)
        
        moved_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                moved_kwargs[key] = value.to(target_device)
            else:
                moved_kwargs[key] = value
        
        original_get_device = model_management.get_torch_device
        
        try:
            model_management.get_torch_device = lambda: target_device
            vhs_node = VHS_LoadVideo()
            function_name = getattr(VHS_LoadVideo, 'FUNCTION', 'load_video')
            result = getattr(vhs_node, function_name)(**moved_kwargs)
            
            if isinstance(result, (tuple, list)):
                moved_result = []
                for item in result:
                    if isinstance(item, torch.Tensor):
                        moved_result.append(item.to(target_device))
                    else:
                        moved_result.append(item)
                result = tuple(moved_result) if isinstance(result, tuple) else moved_result
            elif isinstance(result, torch.Tensor):
                result = result.to(target_device)
            
            return result
        finally:
            model_management.get_torch_device = original_get_device

logger.info("VHS_LoadVideoWrapper created successfully")

# Hard-code VHS_VideoCombine INPUT_TYPES based on verified source code
class VHS_VideoCombineWrapper(VHSMultiGPUWrapper):
    def __init__(self):
        pass
    
    @classmethod 
    def INPUT_TYPES(cls):
        devices = get_device_list()
        default_device = "cuda:0" if "cuda:0" in devices else (devices[1] if len(devices) > 1 else devices[0])
        
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 8, "min": 1, "step": 1}),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
                "format": (["image/gif", "image/webp", "video/mp4", "video/avi"], {}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
                "device": (devices, {"default": default_device}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            },
        }
        
    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    FUNCTION = "execute"
    CATEGORY = "video/gpu_wrapper"
    
    def execute(self, device="cuda:0", **kwargs):
        import sys
        modules_copy = dict(sys.modules)
        VHS_VideoCombine = None
        
        for module_name, module in modules_copy.items():
            if hasattr(module, 'NODE_CLASS_MAPPINGS') and module.NODE_CLASS_MAPPINGS:
                if isinstance(module.NODE_CLASS_MAPPINGS, dict):
                    if "VHS_VideoCombine" in module.NODE_CLASS_MAPPINGS:
                        VHS_VideoCombine = module.NODE_CLASS_MAPPINGS["VHS_VideoCombine"]
                        break
        
        if VHS_VideoCombine is None:
            raise RuntimeError("VHS_VideoCombine not found - ensure VideoHelperSuite is installed")
        
        self.vhs_node_class = VHS_VideoCombine
        target_device = torch.device(device)
        
        moved_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                moved_kwargs[key] = value.to(target_device)
            else:
                moved_kwargs[key] = value
        
        original_get_device = model_management.get_torch_device
        
        try:
            model_management.get_torch_device = lambda: target_device
            vhs_node = VHS_VideoCombine()
            function_name = getattr(VHS_VideoCombine, 'FUNCTION', 'combine')
            result = getattr(vhs_node, function_name)(**moved_kwargs)
            return result
        finally:
            model_management.get_torch_device = original_get_device

logger.info("VHS_VideoCombineWrapper created successfully")

# Register nodes directly since the classes are defined above
NODE_CLASS_MAPPINGS = {
    "VHS_LoadVideoWrapper": VHS_LoadVideoWrapper,
    "VHS_VideoCombineWrapper": VHS_VideoCombineWrapper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_LoadVideoWrapper": "Load Video (GPU Wrapper)",
    "VHS_VideoCombineWrapper": "Video Combine (GPU Wrapper)"
}

logger.info(f"Registered {len(NODE_CLASS_MAPPINGS)} VHS GPU wrapper nodes: {list(NODE_CLASS_MAPPINGS.keys())}")