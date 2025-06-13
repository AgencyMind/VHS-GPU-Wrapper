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


# Import and create VHS wrapper with error handling

# Initialize wrapper classes to None
VHS_LoadVideoWrapper = None
VHS_VideoCombineWrapper = None
VHS_LoadVideoUploadWrapper = None

try:
    # Import VideoHelperSuite nodes
    import VideoHelperSuite
    VHS_NODE_MAPPINGS = VideoHelperSuite.NODE_CLASS_MAPPINGS
    logger.info(f"Found VHS nodes: {list(VHS_NODE_MAPPINGS.keys())}")
    
    # VHS_LoadVideo Wrapper
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
            
        logger.info("VHS_LoadVideoWrapper created successfully")
    
    # VHS_VideoCombine Wrapper
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
            
        logger.info("VHS_VideoCombineWrapper created successfully")
    
    # VHS_LoadVideoUpload Wrapper (if it exists)
    if "VHS_LoadVideoUpload" in VHS_NODE_MAPPINGS:
        VHS_LoadVideoUpload = VHS_NODE_MAPPINGS["VHS_LoadVideoUpload"]
        
        class VHS_LoadVideoUploadWrapper(VHSMultiGPUWrapper):
            def __init__(self):
                super().__init__(VHS_LoadVideoUpload)
            
            @classmethod
            def INPUT_TYPES(cls):
                base_inputs = VHS_LoadVideoUpload.INPUT_TYPES()
                
                devices = get_device_list()
                default_device = "cuda:0" if "cuda:0" in devices else (devices[1] if len(devices) > 1 else devices[0])
                
                if "optional" not in base_inputs:
                    base_inputs["optional"] = {}
                
                base_inputs["optional"]["device"] = (devices, {"default": default_device})
                return base_inputs
                
            RETURN_TYPES = VHS_LoadVideoUpload.RETURN_TYPES
            RETURN_NAMES = getattr(VHS_LoadVideoUpload, 'RETURN_NAMES', ("IMAGE", "frame_count", "audio", "video_info"))
            FUNCTION = "execute"
            CATEGORY = "video/gpu_wrapper"
            
        logger.info("VHS_LoadVideoUploadWrapper created successfully")
        
except (ImportError, KeyError) as e:
    logger.warning(f"VideoHelperSuite not available: {e}")
except Exception as e:
    logger.error(f"Error creating VHS wrappers: {e}")
    import traceback
    traceback.print_exc()


# Registration dictionaries
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Only register wrappers for available VHS nodes
if VHS_LoadVideoWrapper:
    NODE_CLASS_MAPPINGS["VHS_LoadVideoWrapper"] = VHS_LoadVideoWrapper
    NODE_DISPLAY_NAME_MAPPINGS["VHS_LoadVideoWrapper"] = "Load Video (GPU Wrapper)"

if VHS_VideoCombineWrapper:
    NODE_CLASS_MAPPINGS["VHS_VideoCombineWrapper"] = VHS_VideoCombineWrapper  
    NODE_DISPLAY_NAME_MAPPINGS["VHS_VideoCombineWrapper"] = "Video Combine (GPU Wrapper)"

if VHS_LoadVideoUploadWrapper:
    NODE_CLASS_MAPPINGS["VHS_LoadVideoUploadWrapper"] = VHS_LoadVideoUploadWrapper
    NODE_DISPLAY_NAME_MAPPINGS["VHS_LoadVideoUploadWrapper"] = "Load Video Upload (GPU Wrapper)"

logger.info(f"Registered {len(NODE_CLASS_MAPPINGS)} VHS GPU wrapper nodes: {list(NODE_CLASS_MAPPINGS.keys())}")