#!/usr/bin/env python3
"""
Flask REST API for SAM2 Image Segmentation

Provides a REST API endpoint for segmenting objects in images using SAM2.
Designed for iOS client integration using multipart/form-data.

Endpoint: POST /api/segment
- Accepts: multipart/form-data with 'image' field
- Query params: x (float), y (float), device (str, default 'mps'/'cpu')
- Returns: JSON with segmentation masks and polygon data

@todo: Add authentication/API keys
@todo: Add rate limiting
@todo: Add support for multiple points
"""

# import os
import json
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename
import signal
import time

from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = Path("output")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "heic", "heif"}
CHECKPOINT = "./sam/checkpoints/sam2.1_hiera_large.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
REQUEST_TIMEOUT = 60  # 10 minutes in seconds

# Global predictor (initialized on first request to save startup time)
predictor = None


def timeout_handler(signum, frame):
    """Handle request timeout."""
    raise TimeoutError("Request exceeded maximum processing time")


def allowed_file(filename):
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def init_predictor(device=None):
    """
    Initialize the SAM2 predictor (lazy loading).

    Args:
        device: Device to run inference on. If None, auto-selects the best available.

    Returns:
        SAM2ImagePredictor: Initialized predictor
    """
    global predictor

    if predictor is None:
        # Auto-select device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        torch_device = torch.device(device)

        # Build SAM2 model
        sam2_model = build_sam2(MODEL_CONFIG, CHECKPOINT, device=torch_device)
        predictor = SAM2ImagePredictor(sam2_model)

    return predictor


def extract_polygon_from_mask(mask):
    """
    Extract polygon coordinates from a binary mask.

    Args:
        mask: Binary mask (2D numpy array)

    Returns:
        List of [x, y] coordinates forming the polygon
    """
    # Ensure mask is uint8
    mask_uint8 = mask.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return []

    # Get the largest contour (main shape)
    largest_contour = max(contours, key=cv2.contourArea)

    # Simplify polygon to reduce number of points
    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Convert to list of [x, y] coordinates
    polygon = approx_polygon.reshape(-1, 2).tolist()

    return [[float(x), float(y)] for x, y in polygon]


def extract_mask_data(mask, score, image_shape):
    """
    Extract detailed information from a segmentation mask.

    Args:
        mask: Binary mask (2D numpy array)
        score: Confidence score for the mask
        image_shape: Original image shape (height, width)

    Returns:
        Dictionary containing mask information
    """
    # Calculate bounding box
    y_indices, x_indices = np.where(mask)

    if len(x_indices) == 0 or len(y_indices) == 0:
        return None

    x1, x2 = float(x_indices.min()), float(x_indices.max())
    y1, y2 = float(y_indices.min()), float(y_indices.max())

    # Extract polygon
    polygon = extract_polygon_from_mask(mask)

    mask_data = {
        "score": float(score),
        "area": int(np.sum(mask)),
        "bbox": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": x2 - x1,
            "height": y2 - y1,
            "center_x": (x1 + x2) / 2,
            "center_y": (y1 + y2) / 2,
        },
        "polygon": polygon,
        "image_dimensions": {
            "width": image_shape[1],
            "height": image_shape[0],
        },
    }

    return mask_data


def create_visualization(image, mask, output_path):
    """
    Create and save visualization image with segmentation mask overlay.

    Args:
        image: Original image (numpy array)
        mask: Binary mask (2D numpy array)
        output_path: Path to save the visualization

    Returns:
        Path: Path to the saved visualization image
    """
    # Create a copy of the image
    vis_img = image.copy()

    # Ensure mask is boolean type for indexing
    mask_bool = mask.astype(bool)

    # Create colored overlay for mask
    overlay = np.zeros_like(image)
    overlay[mask_bool] = [0, 255, 0]  # Green color for mask

    # Blend with original image
    vis_img = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)

    # Draw contour
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)

    # Save the visualization
    cv2.imwrite(str(output_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

    return output_path


def create_polygon_debug_image(image, mask_data_list, point_coords, output_path):
    """
    Create a debug visualization showing polygon contours for detected segments.

    Args:
        image: Original image (numpy array)
        mask_data_list: List of mask data dictionaries
        point_coords: Original point coordinates [x, y]
        output_path: Path to save the debug image

    Returns:
        Path: Path to the saved debug image
    """
    # Create a copy for drawing (convert RGB to BGR for OpenCV)
    debug_img = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    for idx, mask_data in enumerate(mask_data_list):
        if "polygon" not in mask_data or not mask_data["polygon"]:
            continue

        # Get polygon points
        polygon = mask_data["polygon"]
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))

        # Color based on score (higher score = more green)
        score = mask_data["score"]
        color = (0, int(255 * score), int(255 * (1 - score)))

        # Draw filled polygon with transparency
        overlay = debug_img.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, debug_img, 0.7, 0, debug_img)

        # Draw polygon outline
        cv2.polylines(debug_img, [pts], True, color, 2)

        # Draw polygon vertices
        for point in polygon:
            cv2.circle(debug_img, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

        # Add label with mask number and score
        center_x = int(mask_data["bbox"]["center_x"])
        center_y = int(mask_data["bbox"]["center_y"])
        label = f"Mask {idx + 1} (score: {score:.3f})"

        # Add text background
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            debug_img,
            (center_x - 5, center_y - text_height - 5),
            (center_x + text_width + 5, center_y + 5),
            (0, 0, 0),
            -1,
        )

        # Add text
        cv2.putText(
            debug_img,
            label,
            (center_x, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Draw the input point
    if point_coords is not None:
        cv2.circle(
            debug_img,
            (int(point_coords[0]), int(point_coords[1])),
            8,
            (255, 0, 0),
            -1,
        )
        cv2.circle(
            debug_img,
            (int(point_coords[0]), int(point_coords[1])),
            10,
            (255, 255, 255),
            2,
        )

    # Save the debug image
    cv2.imwrite(str(output_path), debug_img)

    return output_path


def process_image(image_path, x, y, device=None):
    """
    Process an image and segment the object at the given point.

    Args:
        image_path: Path to the image file
        x: X-coordinate of the point
        y: Y-coordinate of the point
        device: Device to run inference on (None for auto-detection)

    Returns:
        tuple: (result dict, best_mask, image) - Segmentation results, best mask, and original image
    """
    # Load image
    image = np.array(Image.open(str(image_path)).convert("RGB"))

    # Validate coordinates
    height, width = image.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(
            f"Coordinates ({x}, {y}) are out of bounds for image size ({width}, {height})"
        )

    # Determine device for autocast
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Initialize predictor
    pred = init_predictor(device)

    # Run inference
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        pred.set_image(image)

        masks, scores, logits = pred.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),  # 1 = foreground point
            multimask_output=True,
        )

    # Sort masks by score in descending order
    sorted_indices = np.argsort(scores)[::-1]
    masks = masks[sorted_indices]
    scores = scores[sorted_indices]

    # Extract data for all masks
    masks_data = []
    for mask, score in zip(masks, scores):
        mask_data = extract_mask_data(mask, score, image.shape[:2])
        if mask_data:
            masks_data.append(mask_data)

    result = {
        "point": {"x": float(x), "y": float(y)},
        "masks": masks_data,
        "summary": {
            "total_masks": len(masks_data),
            "best_score": float(scores[0]) if len(scores) > 0 else 0.0,
        },
    }

    # Return best mask for visualization
    best_mask = masks[0] if len(masks) > 0 else None

    return result, best_mask, image


@app.route("/api/segment", methods=["POST"])
def segment_object():
    """
    Main endpoint for object segmentation.

    Accepts multipart/form-data with an image file.
    Query parameters:
        - x: float (required) - X coordinate of the point
        - y: float (required) - Y coordinate of the point
        - visualize: bool (default true) - whether to create visualization images

    Returns:
        JSON response with segmentation data including polygon coordinates
    """
    try:
        # Validate request
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "No image file selected"}), 400

        if not allowed_file(file.filename):
            return (
                jsonify(
                    {
                        "error": f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
                    }
                ),
                400,
            )

        # Set timeout alarm for Unix systems (Linux, macOS)
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(REQUEST_TIMEOUT)
        except (AttributeError, ValueError):
            # Windows doesn't support signal.SIGALRM, skip timeout
            pass

        # Get query parameters
        x = request.args.get("x")
        y = request.args.get("y")

        if x is None or y is None:
            return (
                jsonify({"error": "Both 'x' and 'y' coordinates are required"}),
                400,
            )

        try:
            x = float(x)
            y = float(y)
        except ValueError:
            return jsonify({"error": "Coordinates must be valid numbers"}), 400

        visualize = request.args.get("visualize", "true").lower() in [
            "true",
            "1",
            "yes",
        ]

        # Create output directory with datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
        output_dir = UPLOAD_FOLDER / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        filename = secure_filename(file.filename)
        image_path = output_dir / filename
        file.save(str(image_path))

        # Process image
        result, best_mask, image = process_image(image_path, x, y, device=None)

        # Create visualizations (if requested)
        vis_filename = None
        debug_filename = None

        if visualize and best_mask is not None:
            # Create main visualization with best mask
            vis_filename = f"{Path(filename).stem}_segmented.jpg"
            vis_path = output_dir / vis_filename
            create_visualization(image, best_mask, vis_path)

            # Create polygon debug visualization
            debug_filename = f"{Path(filename).stem}_debug_polygons.jpg"
            debug_path = output_dir / debug_filename
            create_polygon_debug_image(image, result["masks"], [x, y], debug_path)

        # Save JSON result
        json_path = output_dir / "segmentation_result.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

        # Add metadata to response
        result["metadata"] = {
            "timestamp": timestamp,
            "original_filename": filename,
            "saved_to": str(output_dir),
        }

        # Add visualization filenames if created
        if vis_filename:
            result["metadata"]["visualization_filename"] = vis_filename
        if debug_filename:
            result["metadata"]["debug_polygon_filename"] = debug_filename

        return jsonify(result), 200

    except TimeoutError as e:
        return jsonify({"error": str(e)}), 408
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return (
            jsonify(
                {
                    "error": "Model checkpoint not found. Please ensure SAM2 checkpoint exists.",
                    "details": str(e),
                }
            ),
            500,
        )
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
    finally:
        # Disable timeout alarm
        try:
            signal.alarm(0)
        except (AttributeError, ValueError):
            pass


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return (
        jsonify(
            {
                "status": "healthy",
                "model_loaded": predictor is not None,
                "timestamp": datetime.now().isoformat(),
                "model": "SAM2 (Segment Anything Model 2)",
                "checkpoint": CHECKPOINT,
            }
        ),
        200,
    )


@app.route("/", methods=["GET"])
def index():
    """Root endpoint with API documentation."""
    return (
        jsonify(
            {
                "name": "SAM2 Image Segmentation API",
                "version": "1.0.0",
                "model": "SAM2 (Segment Anything Model 2)",
                "endpoints": {
                    "POST /api/segment": {
                        "description": "Segment object at a given point in an image",
                        "accepts": "multipart/form-data with 'image' field",
                        "query_params": {
                            "x": "float (required) - X coordinate of the point",
                            "y": "float (required) - Y coordinate of the point",
                            "visualize": "bool (true/false, default true) - create visualization images",
                        },
                        "returns": "JSON with segmentation masks and polygon data",
                    },
                    "GET /api/health": {
                        "description": "Health check endpoint",
                        "returns": "Service status",
                    },
                },
                "allowed_image_types": list(ALLOWED_EXTENSIONS),
            }
        ),
        200,
    )


if __name__ == "__main__":
    # Ensure output directory exists
    UPLOAD_FOLDER.mkdir(exist_ok=True)

    # Check if model checkpoint exists
    if not Path(CHECKPOINT).exists():
        print("WARNING: Model checkpoint not found!")
        print(f"Please ensure this file exists: {CHECKPOINT}")
        print("\nDownload SAM2 checkpoints using:")
        print("  cd repo/checkpoints && ./download_ckpts.sh")

    if not Path(MODEL_CONFIG).exists():
        print("WARNING: Model config not found!")
        print(f"Please ensure this file exists: {MODEL_CONFIG}")

    # Run Flask app
    print("\nStarting SAM2 Image Segmentation API...")
    print("Model: SAM2 (Segment Anything Model 2)")
    print("Ready to accept requests from iOS clients\n")
    app.run(host="0.0.0.0", port=5005, debug=True)
