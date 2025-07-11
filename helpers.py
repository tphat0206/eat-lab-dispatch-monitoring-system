import numpy as np
import torch
from config import CLASSIFIER_CLASSES

def classify_cropped_object(cropped_image_bgr: np.ndarray, classifier_transform, detection_class_name: str, models) -> str:
    image_tensor = classifier_transform(cropped_image_bgr)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension: (1, C, H, W)
    # Get the label for the detected object

    if detection_class_name == 'dish':
        classification_results = models['dish_classifier'](image_tensor)
    elif detection_class_name == 'tray':
        classification_results = models['tray_classifier'](image_tensor)
    else:
        classification_results = models['classifier'](image_tensor)

    probabilities = torch.softmax(classification_results, dim=1)
    predicted_class_idx = torch.argmax(probabilities, dim=1).item()

    # Get the top class and its confidence
    return CLASSIFIER_CLASSES[predicted_class_idx]