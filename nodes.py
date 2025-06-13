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
VHS_LoadVideoUploadWrapper = None

# Create placeholder wrapper that checks for VHS at runtime
class VHS_LoadVideoWrapper(VHSMultiGPUWrapper):
    def __init__(self):
        # Don't call super().__init__ yet - VHS might not be loaded
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        # Try to get actual VHS_LoadVideo INPUT_TYPES
        import sys
        modules_copy = dict(sys.modules)
        VHS_LoadVideo = None
        
        for module_name, module in modules_copy.items():
            if hasattr(module, 'NODE_CLASS_MAPPINGS') and module.NODE_CLASS_MAPPINGS:
                if isinstance(module.NODE_CLASS_MAPPINGS, dict):
                    if "VHS_LoadVideo" in module.NODE_CLASS_MAPPINGS:
                        VHS_LoadVideo = module.NODE_CLASS_MAPPINGS["VHS_LoadVideo"]
                        break
        
        if VHS_LoadVideo and hasattr(VHS_LoadVideo, 'INPUT_TYPES'):
            try:
                base_inputs = VHS_LoadVideo.INPUT_TYPES()
                # Add device parameter
                devices = get_device_list()
                default_device = "cuda:0" if "cuda:0" in devices else (devices[1] if len(devices) > 1 else devices[0])
                
                if "optional" not in base_inputs:
                    base_inputs["optional"] = {}
                
                base_inputs["optional"]["device"] = (devices, {"default": default_device})
                return base_inputs
            except:
                pass
        
        # Fallback if VHS not available yet
        devices = get_device_list()
        default_device = "cuda:0" if "cuda:0" in devices else (devices[1] if len(devices) > 1 else devices[0])
        
        return {
            "required": {
                "video": ("STRING", {"default": "video.mp4"}),
            },
            "optional": {
                "device": (devices, {"default": default_device}),
            }
        }
        
    RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info")
    FUNCTION = "execute"
    CATEGORY = "video/gpu_wrapper"
    
    def execute(self, device="cuda:0", **kwargs):
        # Find VHS_LoadVideo at runtime
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
        
        # Now execute with the wrapper logic
        self.vhs_node_class = VHS_LoadVideo
        target_device = torch.device(device)
        
        # Move input tensors to target device
        moved_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                moved_kwargs[key] = value.to(target_device)
            else:
                moved_kwargs[key] = value
        
        # Save original function and override
        original_get_device = model_management.get_torch_device
        
        try:
            model_management.get_torch_device = lambda: target_device
            vhs_node = VHS_LoadVideo()
            function_name = getattr(VHS_LoadVideo, 'FUNCTION', 'load_video')
            result = getattr(vhs_node, function_name)(**moved_kwargs)
            
            # Move output tensors to target device
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

class VHS_VideoCombineWrapper(VHSMultiGPUWrapper):
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        # Try to get actual VHS_VideoCombine INPUT_TYPES
        import sys
        modules_copy = dict(sys.modules)
        VHS_VideoCombine = None
        
        for module_name, module in modules_copy.items():
            if hasattr(module, 'NODE_CLASS_MAPPINGS') and module.NODE_CLASS_MAPPINGS:
                if isinstance(module.NODE_CLASS_MAPPINGS, dict):
                    if "VHS_VideoCombine" in module.NODE_CLASS_MAPPINGS:
                        VHS_VideoCombine = module.NODE_CLASS_MAPPINGS["VHS_VideoCombine"]
                        break
        
        if VHS_VideoCombine and hasattr(VHS_VideoCombine, 'INPUT_TYPES'):
            try:
                base_inputs = VHS_VideoCombine.INPUT_TYPES()
                # Add device parameter
                devices = get_device_list()
                default_device = "cuda:0" if "cuda:0" in devices else (devices[1] if len(devices) > 1 else devices[0])
                
                if "optional" not in base_inputs:
                    base_inputs["optional"] = {}
                
                base_inputs["optional"]["device"] = (devices, {"default": default_device})
                return base_inputs
            except:
                pass
        
        # Fallback if VHS not available yet
        devices = get_device_list()
        default_device = "cuda:0" if "cuda:0" in devices else (devices[1] if len(devices) > 1 else devices[0])
        
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "device": (devices, {"default": default_device}),
            }
        }
        
    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    FUNCTION = "execute"
    CATEGORY = "video/gpu_wrapper"
    
    def execute(self, device="cuda:0", **kwargs):
        # Find VHS_VideoCombine at runtime
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
        
        # Execute with wrapper logic
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

# Register nodes immediately so ComfyUI can discover them
NODE_CLASS_MAPPINGS = {
    "VHS_LoadVideoWrapper": VHS_LoadVideoWrapper,
    "VHS_VideoCombineWrapper": VHS_VideoCombineWrapper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_LoadVideoWrapper": "Load Video (GPU Wrapper)",
    "VHS_VideoCombineWrapper": "Video Combine (GPU Wrapper)",
}

# Keep the background thread for logging/debugging
def create_vhs_wrappers():
    """Create VHS wrappers after VideoHelperSuite has loaded"""
    global VHS_LoadVideoWrapper, VHS_VideoCombineWrapper, VHS_LoadVideoUploadWrapper
    global NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    
    try:
        # The _OpNamespace error is actually a PyTorch dependency issue, not ComfyUI
        # Let's use the standard ComfyUI import pattern for accessing other extension nodes
        import sys
        VHS_NODE_MAPPINGS = None
        
        logger.info("Searching for VideoHelperSuite nodes...")
        
        # Look for VHS nodes in ComfyUI's global node registry
        # VideoHelperSuite structure: from .videohelpersuite.nodes import NODE_CLASS_MAPPINGS
        vhs_module_names = [
            'videohelpersuite.nodes',
            'ComfyUI-VideoHelperSuite.videohelpersuite.nodes', 
            'custom_nodes.ComfyUI-VideoHelperSuite.videohelpersuite.nodes'
        ]
        
        for module_name in vhs_module_names:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                if hasattr(module, 'NODE_CLASS_MAPPINGS'):
                    VHS_NODE_MAPPINGS = module.NODE_CLASS_MAPPINGS
                    logger.info(f"Found VHS NODE_CLASS_MAPPINGS in: {module_name}")
                    break
        
        # If not found in specific modules, search all loaded modules
        if VHS_NODE_MAPPINGS is None:
            logger.info("VHS not found in expected modules, searching all loaded modules...")
            # Create a copy of sys.modules to avoid "dictionary changed size during iteration"
            modules_copy = dict(sys.modules)
            for module_name, module in modules_copy.items():
                if hasattr(module, 'NODE_CLASS_MAPPINGS') and module.NODE_CLASS_MAPPINGS:
                    # VHS nodes start with "VHS_"
                    if isinstance(module.NODE_CLASS_MAPPINGS, dict):
                        if any(key.startswith('VHS_') for key in module.NODE_CLASS_MAPPINGS.keys()):
                            VHS_NODE_MAPPINGS = module.NODE_CLASS_MAPPINGS
                            logger.info(f"Found VHS nodes in module: {module_name}")
                            break
        
        if VHS_NODE_MAPPINGS is None:
            logger.warning("VideoHelperSuite nodes not found - may not be installed or loaded yet")
            return
            
        # VHS_NODE_MAPPINGS is a standard dictionary - verified from VHS source
        all_vhs_nodes = [k for k in VHS_NODE_MAPPINGS.keys() if k.startswith('VHS_')]
        logger.info(f"Available VHS nodes: {all_vhs_nodes}")
        
        # Create wrappers for found VHS nodes
        if "VHS_LoadVideo" in VHS_NODE_MAPPINGS:
            VHS_LoadVideo = VHS_NODE_MAPPINGS["VHS_LoadVideo"]
            
            class VHS_LoadVideoWrapper(VHSMultiGPUWrapper):
                def __init__(self):
                    super().__init__(VHS_LoadVideo)
                
                @classmethod
                def INPUT_TYPES(cls):
                    base_inputs = VHS_LoadVideo.INPUT_TYPES()
                    
                    devices = get_device_list()
                    default_device = "cuda:0" if "cuda:0" in devices else (devices[1] if len(devices) > 1 else devices[0])
                    
                    if "optional" not in base_inputs:
                        base_inputs["optional"] = {}
                    
                    base_inputs["optional"]["device"] = (devices, {"default": default_device})
                    return base_inputs
                    
                RETURN_TYPES = VHS_LoadVideo.RETURN_TYPES
                RETURN_NAMES = getattr(VHS_LoadVideo, 'RETURN_NAMES', ("IMAGE", "frame_count", "audio", "video_info"))
                FUNCTION = "execute"
                CATEGORY = "video/gpu_wrapper"
                
            NODE_CLASS_MAPPINGS["VHS_LoadVideoWrapper"] = VHS_LoadVideoWrapper
            NODE_DISPLAY_NAME_MAPPINGS["VHS_LoadVideoWrapper"] = "Load Video (GPU Wrapper)"
            logger.info("Created VHS_LoadVideoWrapper")
        
        if "VHS_VideoCombine" in VHS_NODE_MAPPINGS:
            VHS_VideoCombine = VHS_NODE_MAPPINGS["VHS_VideoCombine"]
            
            class VHS_VideoCombineWrapper(VHSMultiGPUWrapper):
                def __init__(self):
                    super().__init__(VHS_VideoCombine)
                
                @classmethod
                def INPUT_TYPES(cls):
                    base_inputs = VHS_VideoCombine.INPUT_TYPES()
                    
                    devices = get_device_list()
                    default_device = "cuda:0" if "cuda:0" in devices else (devices[1] if len(devices) > 1 else devices[0])
                    
                    if "optional" not in base_inputs:
                        base_inputs["optional"] = {}
                    
                    base_inputs["optional"]["device"] = (devices, {"default": default_device})
                    return base_inputs
                    
                RETURN_TYPES = VHS_VideoCombine.RETURN_TYPES
                RETURN_NAMES = getattr(VHS_VideoCombine, 'RETURN_NAMES', ("Filenames",))
                FUNCTION = "execute"
                CATEGORY = "video/gpu_wrapper"
                
            NODE_CLASS_MAPPINGS["VHS_VideoCombineWrapper"] = VHS_VideoCombineWrapper
            NODE_DISPLAY_NAME_MAPPINGS["VHS_VideoCombineWrapper"] = "Video Combine (GPU Wrapper)"
            logger.info("Created VHS_VideoCombineWrapper")
            
        logger.info(f"Successfully registered {len(NODE_CLASS_MAPPINGS)} VHS GPU wrapper nodes")
        
    except Exception as e:
        logger.error(f"Error creating VHS wrappers: {e}")
        import traceback
        traceback.print_exc()

# Don't create wrappers immediately - VHS might not be loaded yet
# Instead, defer until ComfyUI startup is complete
import threading
import time

def delayed_wrapper_creation():
    """Create VHS wrappers after a delay to ensure VHS has loaded"""
    time.sleep(2)  # Wait 2 seconds for other nodes to load
    logger.info("Attempting delayed VHS wrapper creation...")
    create_vhs_wrappers()
    
    if not NODE_CLASS_MAPPINGS:
        logger.info("No VHS GPU wrapper nodes registered - VideoHelperSuite may not be loaded")

# Start delayed creation in background thread
wrapper_thread = threading.Thread(target=delayed_wrapper_creation, daemon=True)
wrapper_thread.start()

logger.info("VHS GPU wrapper loading deferred - will attempt after startup delay")