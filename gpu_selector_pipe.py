import torch
import comfy.model_management as mm

class GPUSelectorPipe:
    @classmethod
    def INPUT_TYPES(s):
        devices = ["cpu", "cuda:0"]
        if torch.cuda.device_count() > 1:
            for i in range(1, torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
        
        return {
            "required": {
                "images": ("IMAGE",),
                "device": (devices,),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "move_to_device"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/device"
    
    def move_to_device(self, images, device):
        # Move tensor to specified device
        moved_images = images.to(device)
        return (moved_images,)

class GPUSelectorAny:
    @classmethod
    def INPUT_TYPES(s):
        devices = ["cpu", "cuda:0"]
        if torch.cuda.device_count() > 1:
            for i in range(1, torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
        
        return {
            "required": {
                "input_data": ("*",),  # Accept ANY type
                "device": (devices,),
            }
        }
    
    RETURN_TYPES = ("*",)  # Return ANY type
    RETURN_NAMES = ("output",)
    FUNCTION = "move_to_device"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/device"
    
    def move_to_device(self, input_data, device):
        """Move any type of data to the specified device."""
        if input_data is None:
            return (input_data,)
            
        # Handle different data types
        if isinstance(input_data, torch.Tensor):
            # Direct tensor movement
            moved_data = input_data.to(device)
        elif isinstance(input_data, (list, tuple)):
            # Handle lists/tuples of tensors
            moved_data = type(input_data)(
                item.to(device) if isinstance(item, torch.Tensor) else item 
                for item in input_data
            )
        elif isinstance(input_data, dict):
            # Handle dictionaries containing tensors
            moved_data = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in input_data.items()
            }
        else:
            # For non-tensor types, return as-is
            moved_data = input_data
            
        return (moved_data,)

class GPUSelectorMask:
    @classmethod
    def INPUT_TYPES(s):
        devices = ["cpu", "cuda:0"]
        if torch.cuda.device_count() > 1:
            for i in range(1, torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
        
        return {
            "required": {
                "mask": ("MASK",),
                "device": (devices,),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "move_to_device"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/device"
    
    def move_to_device(self, mask, device):
        # Move tensor to specified device
        moved_mask = mask.to(device)
        return (moved_mask,)

NODE_CLASS_MAPPINGS = {
    "VHS_GPUSelectorPipe": GPUSelectorPipe,
    "VHS_GPUSelectorAny": GPUSelectorAny,
    "VHS_GPUSelectorMask": GPUSelectorMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_GPUSelectorPipe": "GPU Selector Pipe 🎥🅥🅗🅢",
    "VHS_GPUSelectorAny": "GPU Selector (Any) 🎥🅥🅗🅢",
    "VHS_GPUSelectorMask": "GPU Selector (Mask) 🎥🅥🅗🅢",
}