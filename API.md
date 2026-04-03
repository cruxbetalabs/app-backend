# SAM2 Image Segmentation API

This document describes the Flask API implemented in `app.py`.

## Overview

The service exposes a small HTTP API for point-based image segmentation using SAM2.

- Base URL: `http://localhost:5005`
- Default port: `5005`
- Content type for segmentation requests: `multipart/form-data`
- Response type: `application/json`
- Authentication: none

## Runtime Notes

- The model is loaded lazily on the first segmentation request.
- The predictor is kept in a global variable after initialization.
- Device selection is automatic inside the app: `cuda` if available, otherwise `mps`, otherwise `cpu`.
- Request timeout is 60 seconds.
- Uploaded files and generated artifacts are stored under `output/<timestamp>/`.

## Supported Image Types

The API accepts uploaded image filenames with these extensions:

- `png`
- `jpg`
- `jpeg`
- `heic`
- `heif`

## Endpoints

### `GET /`

Returns a summary of the API and available endpoints.

#### Response

Status: `200 OK`

Example response:

```json
{
  "allowed_image_types": ["png", "jpg", "jpeg", "heic", "heif"],
  "endpoints": {
    "GET /api/health": {
      "description": "Health check endpoint",
      "returns": "Service status"
    },
    "POST /api/segment": {
      "accepts": "multipart/form-data with 'image' field",
      "description": "Segment object at a given point in an image",
      "query_params": {
        "visualize": "bool (true/false, default true) - create visualization images",
        "x": "float (required) - X coordinate of the point",
        "y": "float (required) - Y coordinate of the point"
      },
      "returns": "JSON with segmentation masks and polygon data"
    }
  },
  "model": "SAM2 (Segment Anything Model 2)",
  "name": "SAM2 Image Segmentation API",
  "version": "1.0.0"
}
```

### `GET /api/health`

Returns service health information.

#### Response

Status: `200 OK`

Example response:

```json
{
  "checkpoint": "./sam/checkpoints/sam2.1_hiera_large.pt",
  "model": "SAM2 (Segment Anything Model 2)",
  "model_loaded": false,
  "status": "healthy",
  "timestamp": "2026-03-11T12:34:56.123456"
}
```

#### Field Notes

- `model_loaded` is `true` only after the first successful segmentation request initializes the predictor.
- `checkpoint` is the configured checkpoint path, not a validation result.

### `POST /api/segment`

Segments the object located at the provided point in the uploaded image.

#### Request

Content type: `multipart/form-data`

Form fields:

- `image`: required image file upload

Query parameters:

- `x`: required float, horizontal coordinate in image pixel space
- `y`: required float, vertical coordinate in image pixel space
- `visualize`: optional boolean-like string, defaults to `true`

Accepted `visualize` truthy values:

- `true`
- `1`
- `yes`

Any other value is treated as `false`.

#### Coordinate Rules

- Coordinates are validated against the image bounds after the image is loaded.
- Valid range is `0 <= x < image_width` and `0 <= y < image_height`.
- Coordinates are parsed as floats.

#### Processing Behavior

- The service generates up to three masks because `multimask_output=True` is enabled.
- Returned masks are sorted by descending confidence score.
- The best mask is used for the main visualization image when visualization is enabled.
- A JSON result file is always written to disk as `segmentation_result.json`.

#### Success Response

Status: `200 OK`

Example request:

```bash
curl -X POST "http://localhost:5005/api/segment?x=320&y=240&visualize=true" \
  -F "image=@/absolute/path/to/image.jpg"
```

Example response:

```json
{
  "metadata": {
    "debug_polygon_filename": "image_debug_polygons.jpg",
    "original_filename": "image.jpg",
    "saved_to": "output/20260311_120746_532",
    "timestamp": "20260311_120746_532",
    "visualization_filename": "image_segmented.jpg"
  },
  "masks": [
    {
      "area": 18432,
      "bbox": {
        "center_x": 312.5,
        "center_y": 241.0,
        "height": 156.0,
        "width": 145.0,
        "x1": 240.0,
        "x2": 385.0,
        "y1": 163.0,
        "y2": 319.0
      },
      "image_dimensions": {
        "height": 480,
        "width": 640
      },
      "polygon": [
        [240.0, 163.0],
        [385.0, 170.0],
        [376.0, 319.0],
        [246.0, 310.0]
      ],
      "score": 0.987
    }
  ],
  "point": {
    "x": 320.0,
    "y": 240.0
  },
  "summary": {
    "best_score": 0.987,
    "total_masks": 1
  }
}
```

#### Response Schema

Top-level response fields:

- `point`: echoed input point
- `masks`: array of mask results sorted by score descending
- `summary`: aggregate information about the prediction
- `metadata`: output file metadata added by the route handler

`point` object:

- `x`: float
- `y`: float

Each item in `masks`:

- `score`: float confidence score from SAM2
- `area`: integer count of mask pixels
- `bbox`: bounding box and derived values
- `polygon`: simplified contour as an array of `[x, y]` pairs
- `image_dimensions`: original image width and height

`bbox` object:

- `x1`: float minimum x
- `y1`: float minimum y
- `x2`: float maximum x
- `y2`: float maximum y
- `width`: float computed as `x2 - x1`
- `height`: float computed as `y2 - y1`
- `center_x`: float midpoint of the box on x-axis
- `center_y`: float midpoint of the box on y-axis

`summary` object:

- `total_masks`: integer number of masks returned after filtering empty masks
- `best_score`: float score of the highest-ranked mask, or `0.0` if none exist

`metadata` object:

- `timestamp`: output directory timestamp in `YYYYMMDD_HHMMSS_mmm` format
- `original_filename`: sanitized uploaded filename
- `saved_to`: output directory path on disk
- `visualization_filename`: present only when `visualize=true` and a best mask exists
- `debug_polygon_filename`: present only when `visualize=true` and a best mask exists

#### Generated Files

For each successful request, the service creates a timestamped directory under `output/`.

Files that may be created:

- Original uploaded image
- `segmentation_result.json`
- `<original-name>_segmented.jpg` if visualization is enabled
- `<original-name>_debug_polygons.jpg` if visualization is enabled

## Error Responses

The API returns JSON errors in the form:

```json
{
  "error": "message"
}
```

Some internal errors may also include a `details` field.

### `400 Bad Request`

Possible causes:

- Missing `image` form field
- Empty uploaded filename
- Unsupported file extension
- Missing `x` or `y`
- Non-numeric `x` or `y`
- Coordinates outside image bounds

Examples:

```json
{
  "error": "No image file provided"
}
```

```json
{
  "error": "Both 'x' and 'y' coordinates are required"
}
```

```json
{
  "error": "Coordinates must be valid numbers"
}
```

```json
{
  "error": "Coordinates (900.0, 700.0) are out of bounds for image size (640, 480)"
}
```

### `408 Request Timeout`

Returned when processing exceeds the configured 60 second timeout.

Example:

```json
{
  "error": "Request exceeded maximum processing time"
}
```

### `500 Internal Server Error`

Possible causes:

- SAM2 checkpoint file missing
- Unexpected runtime failure during image processing or model inference

Checkpoint example:

```json
{
  "details": "[Errno 2] No such file or directory: './sam/checkpoints/sam2.1_hiera_large.pt'",
  "error": "Model checkpoint not found. Please ensure SAM2 checkpoint exists."
}
```

Generic example:

```json
{
  "error": "Internal server error: <message>"
}
```

## Example Calls

### Health Check

```bash
curl "http://localhost:5005/api/health"
```

### Segment Without Visualization

```bash
curl -X POST "http://localhost:5005/api/segment?x=140.5&y=212.0&visualize=false" \
  -F "image=@/absolute/path/to/image.png"
```

### Segment With HEIC Upload

```bash
curl -X POST "http://localhost:5005/api/segment?x=512&y=384" \
  -F "image=@/absolute/path/to/photo.heic"
```

## Implementation Notes

- Polygon data is extracted from the largest external contour in each mask.
- Polygon simplification uses OpenCV `approxPolyDP` with epsilon set to `0.5%` of contour perimeter.
- Bounding box coordinates are derived from non-zero mask pixels.
- The current implementation does not expose model selection, device overrides, batch requests, or multi-point prompts through the API.
- Authentication, API keys, rate limiting, and multi-point prompt support are listed as future work in the source file but are not implemented.