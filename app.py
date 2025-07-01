import os
import sys
import cv2
import gradio as gr
import numpy as np
from PIL import Image

# Add parent directory to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detectors.detector import YOLODetector
from sam2_api.sam2_client import segment_with_sam2
from utils.visualizer import overlay_masks

def run_pipeline(image_pil):
    temp_path = "temp_input.jpg"
    image_pil.save(temp_path)

    image_cv2 = cv2.imread(temp_path)
    detector = YOLODetector()
    detections = detector.detect_faces_and_hands(image_cv2)

    if not detections:
        return "No face or hand detected."

    masks = segment_with_sam2(temp_path, detections)
    if not masks:
        return "No segmentation masks generated."

    result_img = overlay_masks(temp_path, masks)
    return result_img

iface = gr.Interface(
    fn=run_pipeline,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Image(type="pil", label="Segmented Output"),
    title="Face & Hand Segmentation with SAM",
    description="Upload an image to segment faces and hands using Meta's Segment Anything model."
)

if __name__ == "__main__":
    iface.launch(share=True)