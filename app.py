import os
import json
import base64
import uuid
import logging
from google.cloud import storage, vision, pubsub_v1
import firebase_admin
from firebase_admin import auth, exceptions as firebase_exceptions
from flask import Flask, request

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

@app.route("/", methods=["POST"])
def handle_request():
    return ingestion_agent(request)

@app.route("/health", methods=["GET"])
def health_check():
    return {"status": "healthy", "service": "ingestion-service"}, 200

def ingestion_agent(request):
    try:
        # Log request details for debugging
        logger.info(f"Received request with content-type: {request.content_type}")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        
        # Handle different content types
        if request.content_type == 'application/json':
            request_json = request.get_json(silent=True)
        else:
            # Try to get JSON even if content-type is not set correctly
            try:
                request_json = request.get_json(force=True)
            except Exception as e:
                logger.error(f"Failed to parse JSON: {e}")
                request_json = None
        
        if not request_json:
            logger.error("Invalid or missing JSON body")
            logger.error(f"Raw request data: {request.get_data()[:200]}")  # Log first 200 chars
            return {"error": "Invalid or missing JSON body"}, 400

        file_base64 = request_json.get('file_data_base64')
        user_id = request_json.get('user_id')
        
        logger.info(f"Processing request for user_id: {user_id}")
        logger.info(f"File data length: {len(file_base64) if file_base64 else 0}")

        if not file_base64 or not user_id:
            logger.error(f"Missing required fields - user_id: {bool(user_id)}, file_data_base64: {bool(file_base64)}")
            return {"error": "Missing user_id or file_data_base64"}, 400

        try:
            image_content = base64.b64decode(file_base64)
        except Exception as e:
            logger.error(f"Base64 decode error: {e}")
            return {"error": "Invalid base64 data"}, 400

        if not _verify_user_exists(user_id):
            logger.error(f"User verification failed for: {user_id}")
            return {"error": f"Invalid or disabled user ID {user_id}"}, 403

        if not _validate_file_size(image_content):
            logger.error(f"File size validation failed. Size: {len(image_content) / (1024 * 1024):.2f}MB")
            return {"error": f"File exceeds size limit of {MAX_FILE_SIZE_MB}MB"}, 413

        content_type = _detect_content_type(image_content)
        logger.info(f"Detected content type: {content_type}")
        
        if content_type not in ALLOWED_CONTENT_TYPES:
            logger.error(f"Content type not allowed: {content_type}")
            return {"error": f"Content type {content_type} not allowed"}, 415

        if content_type.startswith("image/"):
            if not _perform_safety_check(image_content):
                logger.error("Safety check failed")
                return {"error": "Unsafe content detected in image"}, 422
                
            if not _validate_content_structure(image_content):
                logger.error("Content structure validation failed")
                return {"error": "Insufficient textual content detected"}, 422

        ext = content_type.split("/")[-1]
        destination_blob_name = f"processing-receipts/{user_id}-{uuid.uuid4()}.{ext}"
        
        blob = storage_client.bucket(BUCKET_NAME).blob(destination_blob_name)
        blob.upload_from_string(image_content, content_type=content_type)
        
        gcs_uri = f"gs://{BUCKET_NAME}/{destination_blob_name}"
        logger.info(f"File uploaded to: {gcs_uri}")

        message_payload = json.dumps({
            'file_path': gcs_uri,
            'user_id': user_id
        }).encode('utf-8')
        
        future = publisher_client.publish(EXTRACTION_TOPIC, message_payload)
        future.result()
        
        logger.info(f"Published message for user {user_id} to {EXTRACTION_TOPIC}")
        return {"message": "Receipt accepted and processing started"}, 200

    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        return {"error": "Internal Server Error"}, 500

def _verify_user_exists(user_id: str) -> bool:
    try:
        user_record = auth.get_user(user_id)
        return not user_record.disabled
    except firebase_exceptions.UserNotFoundError:
        logger.warning(f"User not found: {user_id}")
        return False
    except Exception as e:
        logger.error(f"User verification failed: {e}")
        return False

def _validate_file_size(image_bytes: bytes) -> bool:
    size_mb = len(image_bytes) / (1024 * 1024)
    return size_mb <= MAX_FILE_SIZE_MB

def _detect_content_type(image_bytes: bytes) -> str:
    import imghdr
    fmt = imghdr.what(None, h=image_bytes)
    if fmt == 'jpeg':
        return 'image/jpeg'
    elif fmt == 'png':
        return 'image/png'
    elif fmt == 'webp':
        return 'image/webp'
    # Basic PDF signature check
    if image_bytes.startswith(b'%PDF'):
        return 'application/pdf'
    return 'application/octet-stream'

def _perform_safety_check(image_bytes: bytes) -> bool:
    try:
        image = vision.Image(content=image_bytes)
        response = vision_client.safe_search_detection(image=image)
        if response.error.message:
            logger.error(f"Vision API error: {response.error.message}")
            return False
        safe = response.safe_search_annotation
        if any(getattr(safe, attr) in [vision.Likelihood.LIKELY, vision.Likelihood.VERY_LIKELY]
               for attr in ['adult', 'violence', 'racy']):
            logger.warning("Unsafe image content detected")
            return False
        return True
    except Exception as e:
        logger.error(f"Safety check error: {e}")
        return True  # Fail-safe

def _validate_content_structure(image_bytes: bytes) -> bool:
    try:
        image = vision.Image(content=image_bytes)
        response = vision_client.text_detection(image=image)
        if response.error.message:
            logger.warning(f"Text detection error: {response.error.message}")
            return True  # Don't block
        annotations = response.text_annotations
        if not annotations or len(annotations[0].description.strip()) < 10:
            logger.warning("Low text content in image")
            return False
        return True
    except Exception as e:
        logger.error(f"Text detection failed: {e}")
        return True  # Don't block

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
