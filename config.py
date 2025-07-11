import os

# --- Detection Configuration ---
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.45

# --- Classifier Configuration ---
CLASSIFIER_INPUT_SIZE = 224
CLASSIFIER_NORM_MEAN = [0.485, 0.456, 0.406]
CLASSIFIER_NORM_STD = [0.229, 0.224, 0.225]
CLASSIFIER_CLASSES = ['empty', 'kakigori', 'not_empty']
# --- Image Saving Configuration ---
SAVE_ANNOTATED_IMAGES = True
IMAGE_SAVE_DIRECTORY = "predict_image"

# --- Video Saving Configuration ---
SAVE_ANNOTATED_VIDEOS = True
VIDEO_SAVE_DIRECTORY = "predict_video"

def ensure_directories_exist():
    """Ensures that the necessary output directories exist."""
    if SAVE_ANNOTATED_IMAGES and not os.path.exists(IMAGE_SAVE_DIRECTORY):
        os.makedirs(IMAGE_SAVE_DIRECTORY)
        print(f"Created image save directory: {IMAGE_SAVE_DIRECTORY}")

    if SAVE_ANNOTATED_VIDEOS and not os.path.exists(VIDEO_SAVE_DIRECTORY):
        os.makedirs(VIDEO_SAVE_DIRECTORY)
        print(f"Created video save directory: {VIDEO_SAVE_DIRECTORY}")