# EATLAB Dispatch Monitoring System
### Author: Phan Huynh Tan Phat


## Setup
Install required packages.

```bash
  pip install -r requirements.txt
```

Build image

```bash
docker build -t eat-lab-dispatch-monitoring-system . 
```

## Run server
```bash
docker compose up -d
```

## APIs call

1. POST /predict_image
Body: { "file": <image> }
After predicting and classifying, return response success, and the image will be saved into the folder `predict_image`

2. POST /predict_video
Body: { "file": <video> }
After predicting and classifying, return response success, and the image will be saved into the folder `predict_image`
If the video is too heavy, it may respond timeout or take a long time.

3. POST /predict_video_assync
Body: { "file": <video> }
Response: {
     "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
     "status": "PENDING",
     "message": "Video processing started in the background."
   }
Added asyncio and `concurrent.futures.ThreadPoolExecutor` for background task management

4. GET /video_status/{task_id}
Response when completed: {
    "status": "COMPLETED",
    "progress": 100,
    "result": null,
    "error": null
}
