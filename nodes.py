import comfy.model_management as model_management
import torch
import logging
import os
import folder_paths

# Note: floatOrInt and other VHS types are now inherited dynamically from original VHS

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
    _original_class = None  # Class-level cache
    
    def __init__(self):
        # Initialize without calling super().__init__ since we find the VHS class at runtime
        self.vhs_node_class = None
    
    @classmethod
    def _get_original_class(cls):
        """Get original VHS class with caching"""
        if cls._original_class is None:
            import sys
            modules_copy = dict(sys.modules)
            
            for module_name, module in modules_copy.items():
                if hasattr(module, 'NODE_CLASS_MAPPINGS') and module.NODE_CLASS_MAPPINGS:
                    if isinstance(module.NODE_CLASS_MAPPINGS, dict):
                        if "VHS_LoadVideo" in module.NODE_CLASS_MAPPINGS:
                            cls._original_class = module.NODE_CLASS_MAPPINGS["VHS_LoadVideo"]
                            break
        return cls._original_class
    
    @classmethod
    def INPUT_TYPES(cls):
        """Dynamically inherit INPUT_TYPES from original VHS node and add device parameter"""
        original_class = cls._get_original_class()
        
        if original_class and hasattr(original_class, 'INPUT_TYPES'):
            # Get original INPUT_TYPES dynamically
            original_inputs = original_class.INPUT_TYPES()
            
            # Add device parameter to required section
            devices = get_device_list()
            default_device = "cuda:0" if "cuda:0" in devices else (devices[1] if len(devices) > 1 else devices[0])
            
            # Ensure required section exists
            if "required" not in original_inputs:
                original_inputs["required"] = {}
                
            # Add device parameter at the end of required section
            original_inputs["required"]["device"] = (devices, {"default": default_device})
            
            return original_inputs
        
        # Fallback to static definition if original not found
        devices = get_device_list()
        default_device = "cuda:0" if "cuda:0" in devices else (devices[1] if len(devices) > 1 else devices[0])
        
        logger.warning("VHS_LoadVideo not found, using fallback INPUT_TYPES")
        return {
            "required": {
                "video": (["input", "input/"], {"video_upload": True}),
                "force_rate": ("INT", {"default": 0, "min": 0, "max": 60, "step": 1}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "step": 1}),
                "device": (devices, {"default": default_device}),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }
        
    # These will be dynamically set from original VHS node, with fallbacks
    RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO")  # Fallback
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info")  # Fallback  
    FUNCTION = "execute"
    CATEGORY = "video/gpu_wrapper"
    
    @classmethod
    def get_original_attributes(cls):
        """Get original VHS node attributes for dynamic inheritance"""
        original_class = cls._get_original_class()
        
        if original_class:
            # Update class attributes dynamically
            if hasattr(original_class, 'RETURN_TYPES'):
                cls.RETURN_TYPES = original_class.RETURN_TYPES
            if hasattr(original_class, 'RETURN_NAMES'):
                cls.RETURN_NAMES = original_class.RETURN_NAMES
            if hasattr(original_class, 'CATEGORY'):
                # Keep our wrapper category but note original
                pass  # Keep "video/gpu_wrapper"
            
            return original_class
        
        logger.warning("VHS_LoadVideo not found for attribute inheritance")
        return None
    
    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        """Proxy IS_CHANGED to original VHS node for proper cache invalidation"""
        original_class = cls._get_original_class()
        
        if original_class and hasattr(original_class, 'IS_CHANGED'):
            # Filter out wrapper-specific parameters before calling original
            wrapper_params = {'device'}
            original_kwargs = {k: v for k, v in kwargs.items() if k not in wrapper_params}
            return original_class.IS_CHANGED(video, **original_kwargs)
        
        # Fallback if original doesn't have IS_CHANGED
        return float("NaN")
    
    @classmethod
    def VALIDATE_INPUTS(cls, video, **kwargs):
        """Proxy VALIDATE_INPUTS to original VHS node for proper input validation"""
        original_class = cls._get_original_class()
        
        if original_class and hasattr(original_class, 'VALIDATE_INPUTS'):
            # Filter out wrapper-specific parameters before calling original
            wrapper_params = {'device'}
            original_kwargs = {k: v for k, v in kwargs.items() if k not in wrapper_params}
            return original_class.VALIDATE_INPUTS(video, **original_kwargs)
        
        # Fallback validation
        return True
    
    def execute(self, device="cuda:0", **kwargs):
        # Use cached original class
        if self.vhs_node_class is None:
            self.vhs_node_class = self._get_original_class()
            
            if self.vhs_node_class is None:
                raise RuntimeError("VHS_LoadVideo not found - ensure VideoHelperSuite is installed")
        
        # Use preprocessor wrapper pattern - move inputs, override device, return outputs unchanged
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
            
            logger.debug(f"Successfully executed {self.vhs_node_class.__name__} on {target_device}")
            return result
            
        except Exception as e:
            logger.error(f"Error in VHS_LoadVideoWrapper execution: {e}")
            raise
            
        finally:
            # ALWAYS restore original function, even on exception
            model_management.get_torch_device = original_get_device
            logger.debug("Restored original get_torch_device()")
    

# Initialize dynamic attributes from original VHS node
VHS_LoadVideoWrapper.get_original_attributes()

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

