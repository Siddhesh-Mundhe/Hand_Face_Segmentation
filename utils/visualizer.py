# utils/visualizer.py

from PIL import Image
import numpy as np

def overlay_masks(image_path, masks):
    base = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

    # Define semi-transparent colors
    colors = {
        "face": (255, 0, 0, 100),   # Red
        "hand": (0, 255, 0, 100),   # Green
    }

    for mask_dict in masks:
        label = mask_dict["label"]
        mask_np = mask_dict["mask"]  # this is a numpy array now, not a URL

        # Convert binary mask to PIL image
        mask_img = Image.fromarray(mask_np).convert("L")  # grayscale mask
        color_overlay = Image.new("RGBA", base.size, colors.get(label, (0, 0, 255, 100)))  # blue fallback

        # Paste colored mask using the grayscale mask as alpha
        overlay.paste(color_overlay, (0, 0), mask_img)

    result = Image.alpha_composite(base, overlay)
    return result.convert("RGB")
