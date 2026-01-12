import base64
from http import HTTPStatus
from tempfile import NamedTemporaryFile
from types import GeneratorType
import cv2
from fastapi import FastAPI, File, Header, Request, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from matplotlib import pyplot as plt
import numpy as np
from pydantic import BaseModel
from typing import List, Optional, Union
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import io
import torch
import torchvision.transforms as transforms

from sarfusion.data.wisard import adapt_ir2rgb
from sarfusion.models import YOLOv10WiSARD, FusionDetr


description = """SAR-Fusion API allows you to deal with wilderness images and videos.

## Users

You will be able to:
* **Generate bounding boxes from images and videos** 
* **Plot bounding boxes on images and videos**

"""

image_mime_types = ["image/jpeg", "image/png", "image/gif", "image/bmp", "image/webp"]
video_mime_types = [
    "video/mp4",
    "video/mpeg",
    "video/ogg",
    "video/webm",
    "video/quicktime",
]

VERSIONS = {
    "all": "pasqualedem/RGBTFusionDetr_all_tepn3c4l",
    "rgb": "pasqualedem/RGBTFusionDetr_rgb_b80foknc",
    "ir": "pasqualedem/RGBTFusionDetr_ir_00aprtfc",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(version="all"):
    model = FusionDetr.from_pretrained(VERSIONS[version])
    print("Model loaded successfully")
    model.eval()
    
    print(f"Using device: {DEVICE}")
    model.to(DEVICE)
    return model


def denormalize(image):
    if image.shape[1] == 4:
        image = image[:, :3]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return image * std + mean


def show_image_bboxes(pixel_values, labels=None, predictions=None):
    """
    Draw bounding boxes on an image and optionally overlay predictions.

    Args:
        pixel_values (torch.Tensor): A tensor of shape (3, H, W) representing the image.
                                     Expected to be in range [0, 1] or [0, 255].
        labels (dict, optional): A dictionary with keys:
            - 'boxes': A tensor of shape (N, 4) representing bounding box coordinates
                       in [0, 1] range and in (x, y, h, w) format.
            - 'class_labels': A tensor of shape (N,) with class labels for each bounding box.
        predictions (dict, optional): A dictionary with keys:
            - 'boxes': A tensor of shape (M, 4) representing predicted bounding box coordinates
                       in [0, 1] range and in (x, y, h, w) format.
            - 'labels': A tensor of shape (M,) with predicted class labels.
            - 'scores': A tensor of shape (M,) with confidence scores for each predicted box.

    Returns:
        plt.Figure: A matplotlib figure object with the image and drawn bounding boxes.
    """
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not isinstance(pixel_values, torch.Tensor):
        raise ValueError("pixel_values must be a torch.Tensor")

    if len(pixel_values.shape) == 4:
        if pixel_values.shape[0] == 1:
            pixel_values = pixel_values[0]
        else:
            raise ValueError("Unsupported image shape")
    
    # Convert the image tensor to a numpy array for plotting
    image = pixel_values.permute(1, 2, 0).detach().cpu().numpy()

    # Handle normalization dynamically
    if image.max() > 1:  # Likely in [0, 255] range
        image = image / 255.0  # Normalize to [0, 1] range for visualization

    # Validate labels if provided
    if labels is not None:
        if not isinstance(labels, dict) or 'boxes' not in labels or 'class_labels' not in labels:
            raise ValueError("labels must be a dict containing 'boxes' and 'class_labels'")
        boxes = labels['boxes']
        class_labels = labels['class_labels']

        if not isinstance(boxes, torch.Tensor) or boxes.shape[1] != 4:
            raise ValueError("'boxes' must be a torch.Tensor with shape (N, 4)")
        if not isinstance(class_labels, torch.Tensor) or len(class_labels) != len(boxes):
            raise ValueError("'class_labels' must be a torch.Tensor with length equal to the number of boxes")

    # Validate predictions if provided
    if predictions is not None:
        if not isinstance(predictions, dict) or \
                'boxes' not in predictions or 'labels' not in predictions or 'scores' not in predictions:
            raise ValueError(
                "predictions must be a dict containing 'boxes', 'labels', and 'scores'"
            )
        pred_boxes = predictions['boxes']
        pred_labels = predictions['labels']
        pred_scores = predictions['scores']

        if not isinstance(pred_boxes, torch.Tensor) or pred_boxes.shape[1] != 4:
            raise ValueError("'predictions[\"boxes\"]' must be a torch.Tensor with shape (M, 4)")
        if not isinstance(pred_labels, torch.Tensor) or len(pred_labels) != len(pred_boxes):
            raise ValueError("'predictions[\"labels\"]' must be a torch.Tensor with length equal to the number of boxes")
        if not isinstance(pred_scores, torch.Tensor) or len(pred_scores) != len(pred_boxes):
            raise ValueError("'predictions[\"scores\"]' must be a torch.Tensor with length equal to the number of boxes")

    # Get image dimensions
    height, width = image.shape[:2]

    # Create a matplotlib figure
    fig, ax = plt.subplots(1, figsize=(14, 14))
    ax.imshow(image)

    # Draw ground truth bounding boxes if labels are provided
    if labels is not None:
        for box, label in zip(boxes, class_labels):
            xc, yc, w, h = box.cpu()
            xmin = (xc - w / 2) * width
            ymin = (yc - h / 2) * height
            width_box = w * width
            height_box = h * height
            rect = Rectangle((xmin, ymin), width_box, height_box, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, str(label.item()), color='white', fontsize=5, 
                    bbox=dict(facecolor='red', alpha=0.5, edgecolor='none'))

    # Draw predicted bounding boxes if provided
    if predictions is not None:
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            xc, yc, w, h = box.cpu()
            xmin = (xc - w / 2) * width
            ymin = (yc - h / 2) * height
            width_box = w * width
            height_box = h * height
            rect = Rectangle((xmin, ymin), width_box, height_box, linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
            ax.text(
                xmin, ymin - 5, f"{label.item()} ({score.item():.2f})",
                color='white', fontsize=5, bbox=dict(facecolor='blue', alpha=0.5, edgecolor='none')
            )

    ax.axis('off')
    canvas = FigureCanvas(fig)
    canvas.draw()  # Render the figure onto the canvas
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    width, height = canvas.get_width_height()
    image = image.reshape(height, width, 3)  # Reshape to an (H, W, 3) array
    return image


def image_preprocess(tensor):
    global PROCESSOR
    if PROCESSOR is not None:
        out = PROCESSOR(tensor, return_tensors="pt")
        pixel_values = out["pixel_values"]
        if len(pixel_values.shape) == 4:
            pixel_values = pixel_values[0]
        return pixel_values
    else:
        raise NotImplementedError("Preprocessing not implemented for this model")


def get_bytes_img(array):
    bytes_pred = io.BytesIO()
    plt.imsave(bytes_pred, array, format="png")
    bytes_pred.seek(0)
    return bytes_pred


def make_video(frames, output_path):
    fps = 30
    frame_width, frame_height = frames[0].shape[1], frames[0].shape[0]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # or use 'X264' or 'MJPG'
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Write frames to the video file
    for frame in frames:
        out.write(frame)

    # Release the VideoWriter object
    out.release()


app = FastAPI(
    title="SAR Fusion-API 👥🚁",
    description=description,
    version="1.0.0",
)

# Instantiate the model
MODELS = {
    version: build_model(version) for version in VERSIONS
}
PROCESSOR = MODELS["all"].processor if hasattr(MODELS["all"], "processor") else None


class BoundingBox(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    class_id: int


@app.get("/", tags=["General"])  # path operation decorator
def _index(request: Request):
    """Root endpoint."""

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to SAR-Fusion API! Please, read the `/docs`!"},
    }
    return response


def fusion_prepare(rgb_file, infrared_file):
    rgb_img = Image.open(rgb_file.file)
    ir_img = Image.open(infrared_file.file)
    rgb_img, ir_img = adapt_ir2rgb(rgb_img, ir_img)

    tensor_rgb = image_preprocess(rgb_img)
    ir_inputs = image_preprocess(ir_img)
    tensor_ir = ir_inputs[0:1]
    input = torch.cat([tensor_rgb, tensor_ir], dim=0)
    return input


def get_input(rgb_file, infrared_file):
    input = []
    if rgb_file is not None and infrared_file is not None:
        input = fusion_prepare(rgb_file, infrared_file)
    elif rgb_file is not None:
        rgb_img = Image.open(rgb_file.file)
        input = image_preprocess(rgb_img)
    elif infrared_file is not None:
        ir_img = Image.open(infrared_file.file)
        ir_tensor = image_preprocess(ir_img)
        if ir_tensor.shape[0] == 3:
            ir_tensor = ir_tensor[:1]
        input = ir_tensor
    return input.unsqueeze(0).to(DEVICE)


def turn_orig_img_to_rgb(predictions):
    if len(predictions.orig_img.shape) == 2 or predictions.orig_img.shape[2] == 1:
        # Repeat the grayscale channel to have 3 channels
        predictions.orig_img = predictions.orig_img.repeat(3, 2)
    elif predictions.orig_img.shape[2] == 4:
        # Remove the infrared channel
        predictions.orig_img = predictions.orig_img[:, :, :3]
    return predictions

def video_frame_iterator(video_path, ir=False):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Normalize the frame between 0 and 1
        tensor = torch.tensor(frame).permute(2, 0, 1)
        
        # Apply the yolo_preprocess function
        tensor = image_preprocess(tensor)
        
        # Check the condition on the shape of the tensor
        if tensor.shape[0] == 3 and ir:
            tensor = tensor[:1]
        
        yield tensor.unsqueeze(0).to(DEVICE)
    
    cap.release()
    
    
def fusion_video_prepare_iterator(rgb_file, infrared_file):
    rgb_cap = cv2.VideoCapture(rgb_file)
    ir_cap = cv2.VideoCapture(infrared_file)
    
    if not rgb_cap.isOpened() or not ir_cap.isOpened():
        raise ValueError("Error opening video file")
    
    while True:
        ret_rgb, frame_rgb = rgb_cap.read()
        ret_ir, frame_ir = ir_cap.read()
        
        if not ret_rgb or not ret_ir:
            break
        
        
        rgb_tensor = torch.tensor(frame_rgb).permute(2, 0, 1)
        ir_tensor = torch.tensor(frame_ir).permute(2, 0, 1)
        
        rgb_tensor, ir_tensor = adapt_ir2rgb(rgb_tensor, ir_tensor)
        
        rgb_tensor = image_preprocess(rgb_tensor)
        ir_tensor = image_preprocess(ir_tensor)[:1]
        
        input = torch.cat([rgb_tensor, ir_tensor], dim=0).unsqueeze(0).to(DEVICE)
        
        yield input
    
    rgb_cap.release()
    ir_cap.release()
    
    
def get_temp_video_file(video_file):
    suffix = video_file.filename.split(".")[-1]
    temp_file = NamedTemporaryFile(delete=False, suffix=f".{suffix}")
    temp_file.write(video_file.file.read())
    return temp_file.name


def get_video_input(rgb_file, infrared_file):
    input = [] 
    if rgb_file is not None and infrared_file is not None:
        rgb_file = get_temp_video_file(rgb_file)
        infrared_file = get_temp_video_file(infrared_file)
        input = fusion_video_prepare_iterator(rgb_file, infrared_file)
    elif rgb_file is not None:
        filename = get_temp_video_file(rgb_file)
        input = video_frame_iterator(filename)
    elif infrared_file is not None:
        filename = get_temp_video_file(infrared_file)
        return video_frame_iterator(filename, ir=True)
    return input


@app.post(
    "/predict",
    summary="Predict bounding boxes from RGB and Infrared images",
)
async def predict(
    rgb_file: Optional[UploadFile] = File(None),
    infrared_file: Optional[UploadFile] = File(None),
    return_plots: Optional[bool] = Header(False),
    version: Optional[str] = Header("all", examples=["all", "rgb", "ir"]),
    threshold: Optional[float] = Header(0.5, ge=0, le=1),
):
    if rgb_file is None and infrared_file is None:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    input = get_input(rgb_file, infrared_file)
    model = MODELS[version]

    with torch.no_grad():
        output = model(input, threshold=threshold)
    boxes = output["predictions"][0]["boxes"].cpu().numpy()
    scores = output["predictions"][0]["scores"].cpu().numpy()
    labels = output["predictions"][0]["labels"].cpu().numpy()

    # Format the predictions for the response
    bounding_boxes = [
        BoundingBox(
            xmin=box[0],
            ymin=box[1],
            xmax=box[2],
            ymax=box[3],
            confidence=score,
            class_id=int(label),
        )
        for box, score, label in zip(boxes, scores, labels)
    ]

    if return_plots:
        plot = show_image_bboxes(
            denormalize(input.cpu()),
            None,
            predictions=output["predictions"][0],
        )
        return StreamingResponse(
            get_bytes_img(plot), media_type="image/png"
        )

    return bounding_boxes


@app.post("/predict/video", summary="Predict bounding boxes from a video")
def predict_video(
    rgb_video_file: Optional[UploadFile] = File(None),
    infrared_video_file: Optional[UploadFile] = File(None),
    return_plots: Optional[bool] = Header(False),
    version: Optional[str] = Header("all", examples=["all", "rgb", "ir"]),
    threshold: Optional[float] = Header(0.5, ge=0, le=1),
):
    if rgb_video_file is None and infrared_video_file is None:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    predictions = []
    input = list(get_video_input(rgb_video_file, infrared_video_file))
    model = MODELS[version]

    result = []
    for frame in input:
        with torch.no_grad():
            predictions = model(frame, threshold=threshold)
        result.append(predictions)

    if return_plots:
        # Convert the predictions to RGB if necessary
        images = [denormalize(frame.cpu()) for frame in input]
        # Get plots
        plots = [
            show_image_bboxes(image, None, predictions=frame_result["predictions"][0])
            for image, frame_result in zip(images, result)
        ]
        video_path = NamedTemporaryFile(delete=False, suffix=".mp4").name
        make_video(plots, video_path)
        return FileResponse(video_path, media_type="video/mp4", filename="predictions.mp4")
 
    return [
        [
            BoundingBox(
                xmin=box[0],
                ymin=box[1],
                xmax=box[2],
                ymax=box[3],
                confidence=score,
                class_id=int(label),
            )
            for box, score, label in zip(
                frame_result["predictions"][0]["boxes"].cpu().numpy(),
                frame_result["predictions"][0]["scores"].cpu().numpy(),
                frame_result["predictions"][0]["labels"].cpu().numpy(),
            )
        ]
        for frame_result in result
    ]


def run_app():
    import uvicorn

    uvicorn.run(
        "sarfusion.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["sarfusion"],
    )
