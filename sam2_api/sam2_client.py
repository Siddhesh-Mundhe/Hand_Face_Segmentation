# sam2_api/sam2_client.py

import os
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor

# Initialize SAM model globally to avoid reloading each time
_model = None
_predictor = None

def load_sam_model():
    global _model, _predictor

    if _predictor is not None:
        return _predictor

    # Load the model checkpoint
    model_type = "vit_b"
    checkpoint_path = os.path.join("models", "sam_vit_b_01ec64.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device := ("cuda" if torch.cuda.is_available() else "cpu"))

    _predictor = SamPredictor(sam)
    return _predictor


def segment_with_sam2(image_path, bboxes):
    """
    Runs local SAM model and returns mask arrays for each detected box.
    Args:
        image_path (str): path to input image
        bboxes (List[Tuple[label, (x1, y1, x2, y2)]]): list of face/hand detections
    Returns:
        List[Dict]: each dict has "label" and "mask" (as NumPy array)
    """
    predictor = load_sam_model()

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image_rgb)

    # Extract only coordinates for boxes
    box_coords = np.array([box for _, box in bboxes])
    labels = [label for label, _ in bboxes]

    # Convert to tensor and transform for SAM input
    input_boxes = torch.tensor(box_coords, dtype=torch.float, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

    # Run SAM
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False
    )

    results = []
    for i, label in enumerate(labels):
        mask_np = masks[i][0].cpu().numpy().astype(np.uint8) * 255
        results.append({
            "label": label,
            "mask": mask_np
        })

    return results
