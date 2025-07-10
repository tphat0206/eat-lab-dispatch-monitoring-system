import asyncio
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import cv2
import supervision as sv
from fastapi import HTTPException
from torchvision import transforms

from config import (
    VIDEO_SAVE_DIRECTORY
)
from helper import classify_cropped_object


class VideoProcessor:
    def __init__(self):
        self.video_tasks = {}  # In-memory dictionary to store task status and results
        # Initialize a ThreadPoolExecutor for background CPU-bound tasks.
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

    def shutdown_executor(self):
        """Shuts down the ThreadPoolExecutor gracefully on application shutdown."""
        print("VideoProcessor: Shutting down ThreadPoolExecutor...")
        self.executor.shutdown(wait=True)
        print("VideoProcessor: ThreadPoolExecutor shut down.")

    def submit_video_processing_task(
            self,
            temp_video_input_path: str,
            save_output_video: bool,
            output_video_filename: str,
            models
    ) -> str:
        task_id = str(uuid.uuid4())
        self.video_tasks[task_id] = {
            "status": "PENDING",
            "progress": 0,
            "result": None,
            "error": None
        }

        # Get the current event loop and run the background task in the executor
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            self.executor,
            self._process_video_background,
            task_id,
            temp_video_input_path,
            save_output_video,
            output_video_filename,
            models
        )
        return task_id

    def get_task_status(self, task_id: str) -> dict:
        return self.video_tasks.get(task_id)

    def _process_video_background(
            self,
            task_id: str,
            temp_video_input_path: str,
            save_output_video: bool,
            output_video_filename: str,
            models
    ):
        """
        This is the core video processing logic that runs in a background thread.
        It updates the task's status and results in the `video_tasks` dictionary.
        """
        cap = None
        out = None
        try:
            # Update task status to PROCESSING
            self.video_tasks[task_id]["status"] = "PROCESSING"
            self.video_tasks[task_id]["progress"] = 0
            self.video_tasks[task_id]["result"] = None
            self.video_tasks[task_id]["error"] = None

            cap = cv2.VideoCapture(temp_video_input_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file '{temp_video_input_path}' for processing.")

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            saved_video_path = None
            if save_output_video:
                # Generate a unique filename
                full_save_path = os.path.join(VIDEO_SAVE_DIRECTORY, output_video_filename)

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
                out = cv2.VideoWriter(full_save_path, fourcc, original_fps, (original_width, original_height))

                if not out.isOpened():
                    print(
                        f"Error: Could not open video writer for path: {full_save_path}. Output video will not be saved.")
                    save_output_video = False  # Disable saving if writer can't be opened
                else:
                    saved_video_path = full_save_path

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
                    cv2.putText(frame_bgr, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                2)

                    # Calculate and display FPS
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    if elapsed_time > 0:
                        current_fps = frame_number / elapsed_time
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if save_output_video and out is not None:
                    out.write(frame_bgr)

            self.video_tasks[task_id]["status"] = "COMPLETED"
            self.video_tasks[task_id]["progress"] = 100

        except HTTPException as e:
            self.video_tasks[task_id]["status"] = "FAILED"
            self.video_tasks[task_id]["error"] = str(e)
            raise e

        except Exception as e:
            print(f"Error during video prediction: {e}")
            self.video_tasks[task_id]["status"] = "FAILED"
            self.video_tasks[task_id]["error"] = str(e)
            raise HTTPException(status_code=500, detail=f"Internal server error during video processing: {e}")
        finally:
            # Clean up temporary video file
            if temp_video_input_path is not None and os.path.exists(temp_video_input_path):
                os.remove(temp_video_input_path)
                print(f"Temporary video file removed: {temp_video_input_path}")
            # Ensure VideoWriter is released even on error if it was opened
            if out and out.isOpened():
                out.release()
                print("VideoWriter released in finally block.")
