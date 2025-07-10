# --- FastAPI App Setup ---
import os
import tempfile
import time
from contextlib import asynccontextmanager
from http import HTTPStatus

import cv2
import numpy as np
import supervision as sv
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from torchvision import transforms

from config import IMAGE_SAVE_DIRECTORY
from helper import classify_cropped_object
from video_processor import VideoProcessor

file_dir = os.path.dirname(os.path.realpath(__file__))

models = {}
processor = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    models['detector'] = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt', )
    models['detector'].conf = 0.1
    models['detector'].iou = 0.45
    models['detector'].size = (1600, 3000)

    models['dish_classifier'] = torch.hub.load('ultralytics/yolov5', 'custom',
                                               path='runs/train-cls/exp/weights/best.pt')
    models['tray_classifier'] = torch.hub.load('ultralytics/yolov5', 'custom',
                                               path='runs/train-cls/exp2/weights/best.pt')
    models['classifier'] = torch.hub.load('ultralytics/yolov5', 'custom',
                                          path='runs/train-cls/six_categories/weights/best.pt')
    processor['video_processor'] = VideoProcessor()

    #
    yield
    # Clean up the ML models and release the resources
    models.clear()
    processor.clear()

CLASSIFIER_CLASSES = ['empty', 'kakigori', 'not_empty']
classifier_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_application() -> FastAPI:
    application = FastAPI(
        lifespan=lifespan,
        title="YOLOv5 & Classifier API",
        description="API for object detection with YOLOv5 and classification of detected objects.",
        version="1.0.0"
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return application


application = get_application()

@application.get("/", summary="Health Check")
async def read_root():
    """
    A simple health check endpoint to confirm the API is running.
    """
    return {"message": "YOLOv5 and Classifier API is running!"}


@application.post("/predict_image", summary="Perform Object Detection and Classification")
async def predict_image(file: UploadFile = File(...), status=HTTPStatus.OK):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only image files are allowed.")

    img_bytes = await file.read()  # Read the file asynchronously
    file_name = file.filename
    np_array = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    cv_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    detections = models['detector'](cv_image, size=(1600, 3000))
    detections = sv.Detections.from_yolov5(detections)

    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i]
        class_id = detections.class_id[i]
        # Get the label for the detected object
        detection_class_name = models['detector'].names[int(class_id)]

        # Crop the detected object
        cropped_object = img_bgr[int(y1):int(y2), int(x1):int(x2)]
        cv_cropped_object = cv2.cvtColor(cropped_object, cv2.COLOR_RGB2BGR)

        top_class_name = classify_cropped_object(cv_cropped_object, classifier_transform, detection_class_name, models)
        # Get the label for the detected object

        # Prepare the label for drawing
        label = f'{detection_class_name}-{top_class_name}'

        # Draw bounding box and label on the image
        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img_bgr, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    time_str = time.strftime("%Y%m%d-%H%M%S")
    cv2.imwrite(f'{file_dir}/predict_image/{time_str}_{file_name}', img_bgr)

    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={
            'message': 'Success'
        }
    )


@application.post("/predict_video", summary="Perform Object Detection and Classification on a video")
async def predict_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only video files are allowed.")

    file_name = file.filename
    time_str = time.strftime("%Y%m%d-%H%M%S")
    out_put_path = f'{file_dir}/predict_video/{time_str}_{file_name}'
    out = None
    temp_video_path = None
    SAVE_OUTPUT_VIDEO = True
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_file.write(await file.read())
            temp_video_path = temp_video_file.name

        print(f"Temporary video file saved to: {temp_video_path}")

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Could not open video file.")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video properties: {frame_width}x{frame_height} @ {fps} FPS, {total_frames} frames")

        if SAVE_OUTPUT_VIDEO:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
            out = cv2.VideoWriter(out_put_path, fourcc, fps, (frame_width, frame_height))
            if not out.isOpened():
                print(f"Warning: Could not open video writer for {out_put_path}. Output video will not be saved.")
                SAVE_OUTPUT_VIDEO = False  # Disable saving if writer fails

        frame_number = 0
        start_time = time.time()

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_number += 1
            print(f"Processing frame {frame_number}...")

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detections = models['detector'](frame_rgb, size=(1600, 3000))

            sv_detections = sv.Detections.from_yolov5(detections)
            for i in range(len(sv_detections)):
                x1, y1, x2, y2 = sv_detections.xyxy[i]
                conf = sv_detections.confidence[i]
                class_id = sv_detections.class_id[i]

                # Get the label for the detected object
                detection_class_name = models['detector'].names[int(class_id)]

                # Crop the detected object
                cropped_object = frame_bgr[int(y1):int(y2), int(x1):int(x2)]
                cv_cropped_object = cv2.cvtColor(cropped_object, cv2.COLOR_RGB2BGR)

                # Classify the cropped object
                classifier_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                # Get the top class and its confidence
                top_class_name = classify_cropped_object(cv_cropped_object, classifier_transform,
                                                         detection_class_name, models)

                # Prepare the label for drawing
                label = f'{detection_class_name}-{top_class_name}'

                # Draw bounding box and label on the image
                cv2.rectangle(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame_bgr, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Calculate and display FPS
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time > 0:
                    current_fps = frame_number / elapsed_time
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if SAVE_OUTPUT_VIDEO and out is not None:
                out.write(frame_bgr)

    except HTTPException as e:
        raise e

    except Exception as e:
        print(f"Error during video prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during video processing: {e}")
    finally:
        # Clean up temporary video file
        if temp_video_path is not None and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            print(f"Temporary video file removed: {temp_video_path}")
        # Ensure VideoWriter is released even on error if it was opened
        if out and out.isOpened():
            out.release()
            print("VideoWriter released in finally block.")

    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={
            'message': 'Success'
        }
    )


@application.post("/predict_video_async", summary="Submit a video for asynchronous processing")
async def predict_video_async(file: UploadFile = File(...), ):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only video files are allowed.")

    file_name = file.filename
    time_str = time.strftime("%Y%m%d-%H%M%S")
    if processor['video_processor'] is None:
        raise HTTPException(status_code=500, detail="Video processor not initialized. Please wait for startup.")
    video_processor = processor['video_processor']
    temp_video_path = None
    try:
        # Use a more robust suffix extraction from the original filename
        file_suffix = os.path.splitext(file.filename)[1] if os.path.splitext(file.filename)[1] else ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_video_file:
            temp_video_file.write(await file.read())
            temp_video_path = temp_video_file.name

        print(f"Video for processing saved temporarily to: {temp_video_path}")

        # Submit task to VideoProcessor
        task_id = video_processor.submit_video_processing_task(
            temp_video_path,
            True,
            f'{time_str}_{file_name}',
            models
        )

        return JSONResponse(
            content={
                "task_id": task_id,
                "status": video_processor.get_task_status(task_id)["status"],
                "message": "Video processing started in the background."
            },
            status_code=202  # 202 Accepted status code
        )

    except Exception as e:
        # Ensure temporary file is cleaned up if an error occurs before processing starts
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            print(f"Cleaned up temporary input video due to submission error: {temp_video_path}")
        print(f"Error submitting video processing task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit video processing task: {e}")


@application.get("/video_status/{task_id}", summary="Get status and result of a video processing task")
async def get_video_status(task_id: str):
    if processor['video_processor'] is None:
        raise HTTPException(status_code=500, detail="Video processor not initialized.")

    task_info = processor['video_processor'].get_task_status(task_id)

    if not task_info:
        raise HTTPException(status_code=404, detail=f"Task with ID '{task_id}' not found.")

    return JSONResponse(content=task_info)


if __name__ == '__main__':
    uvicorn.run(application, host="0.0.0.0", port=8000)
