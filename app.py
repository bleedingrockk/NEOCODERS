import os
import json
import base64
import uuid
import logging
import time
from google.cloud import storage, vision, pubsub_v1
import firebase_admin
from firebase_admin import auth, exceptions as firebase_exceptions
from flask import Request,request, Flask
app = Flask(__name__)
if not firebase_admin._apps:
    firebase_admin.initialize_app()
storage_client = storage.Client()
vision_client = vision.ImageAnnotatorClient()
publisher_client = pubsub_v1.PublisherClient()
PROJECT_ID = os.environ.get('GCP_PROJECT', 'planar-cycle-467108-b4')
BUCKET_NAME = f"{PROJECT_ID}.appspot.com"
EXTRACTION_TOPIC = f"projects/{PROJECT_ID}/topics/receipts-for-extraction"
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "5"))
ALLOWED_CONTENT_TYPES = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp', 'application/pdf']
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def ingestion_agent(request: Request):
    try:
        request_json = request.get_json(silent=True)
        if not request_json or not all(k in request_json for k in ['file_data_base64', 'user_id']):
            return 'Error: Missing user_id or file_data_base64.', 400
        user_id = request_json['user_id']
        image_content = base64.b64decode(request_json['file_data_base64'])
        if not _verify_user_exists(user_id):
            return f"Error: Invalid or disabled user ID {user_id}", 403
        if not _validate_file_size(image_content):
            return f"Error: File exceeds size limit of {MAX_FILE_SIZE_MB}MB", 413
        content_type = _detect_content_type(image_content)
        if content_type not in ALLOWED_CONTENT_TYPES:
            return f"Error: Content type {content_type} not allowed", 415
        if not _perform_safety_check(image_content, content_type):
            return "Error: Unsafe content detected in image", 422
        if not _validate_content_structure(image_content, content_type):
            return "Error: Insufficient textual content detected", 422
        destination_blob_name = f"processing-receipts/{user_id}-{uuid.uuid4()}.jpg"
        blob = storage_client.bucket(BUCKET_NAME).blob(destination_blob_name)
        blob.upload_from_string(image_content, content_type=content_type)
        gcs_uri = f"gs://{BUCKET_NAME}/{destination_blob_name}"
        logger.info(f"File uploaded to: {gcs_uri}")
        message_payload = json.dumps({'file_path': gcs_uri, 'user_id': user_id}).encode('utf-8')
        future = publisher_client.publish(EXTRACTION_TOPIC, message_payload)
        future.result()
        logger.info(f"Published message to {EXTRACTION_TOPIC} for user {user_id}")
        return "Receipt accepted and processing started.", 200
    except Exception as e:
        logger.exception(f"Error during ingestion: {e}")
        return "Internal Server Error", 500
def _verify_user_exists(user_id: str) -> bool:
    try:
        user_record = auth.get_user(user_id)
        return not user_record.disabled
    except firebase_exceptions.UserNotFoundError:
        logger.warning(f"User {user_id} not found")
        return False
    except Exception as e:
        logger.error(f"Error verifying user {user_id}: {e}")
        return False
def _validate_file_size(image_bytes: bytes) -> bool:
    size_mb = len(image_bytes) / (1024 * 1024)
    return size_mb <= MAX_FILE_SIZE_MB
def _detect_content_type(image_bytes: bytes) -> str:
    import imghdr
    fmt = imghdr.what(None, h=image_bytes)
    if fmt == 'jpeg':
        return 'image/jpeg'
    if fmt == 'png':
        return 'image/png'
    if fmt == 'webp':
        return 'image/webp'
    return 'application/octet-stream'
def _perform_safety_check(image_bytes: bytes, content_type: str) -> bool:
    try:
        if not content_type.startswith('image/'):
            logger.info("Skipping safety check for non-image content")
            return True
        image = vision.Image(content=image_bytes)
        response = vision_client.safe_search_detection(image=image)
        if response.error.message:
            logger.error(f"Vision API error: {response.error.message}")
            return False
        unsafe = [vision.Likelihood.LIKELY, vision.Likelihood.VERY_LIKELY]
        safe_search = response.safe_search_annotation
        if (safe_search.adult in unsafe or
            safe_search.violence in unsafe or
            safe_search.racy in unsafe):
            logger.warning("Unsafe content detected")
            return False
        return True
    except Exception as e:
        logger.error(f"Safety check failed: {e}")
        return True  
def _validate_content_structure(image_bytes: bytes, content_type: str) -> bool:
    try:
        if not content_type.startswith('image/'):
            logger.info("Skipping text validation for non-image content")
            return True
        image = vision.Image(content=image_bytes)
        response = vision_client.text_detection(image=image)
        if response.error.message:
            logger.warning(f"Text detection error: {response.error.message}")
            return True
        annotations = response.text_annotations
        if not annotations or len(annotations[0].description.strip()) < 10:
            logger.warning("Insufficient textual content detected")
            return False
        return True
    except Exception as e:
        logger.error(f"Content structure validation failed: {e}")
        return True

@app.route("/", methods=["POST"])
def handle_request():
    return ingestion_agent(request)