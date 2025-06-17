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

NODE_CLASS_MAPPINGS = {
    "VHS_GPUSelectorPipe": GPUSelectorPipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_GPUSelectorPipe": "GPU Selector Pipe 🎥🅥🅗🅢",
}