import os
import json
import base64
import uuid
import logging

from google.cloud import storage, vision, pubsub_v1
import firebase_admin
from firebase_admin import auth, exceptions as firebase_exceptions

from flask import Flask, request

# Initialize Flask
app = Flask(__name__)

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    firebase_admin.initialize_app()

# GCP Clients
storage_client = storage.Client()
vision_client = vision.ImageAnnotatorClient()
publisher_client = pubsub_v1.PublisherClient()

# Constants
PROJECT_ID = os.environ.get('GCP_PROJECT', 'planar-cycle-467108-b4')
BUCKET_NAME = f"{PROJECT_ID}.appspot.com"
EXTRACTION_TOPIC = f"projects/{PROJECT_ID}/topics/receipts-for-extraction"
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "5"))
ALLOWED_CONTENT_TYPES = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp', 'application/pdf']

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route("/", methods=["POST"])
def handle_pubsub_push():
    try:
        envelope = request.get_json(force=True)
        logging.info(f"Received Pub/Sub message: {envelope}")

        if not envelope or 'message' not in envelope:
            return "Bad Request: Missing 'message'", 400

        message = envelope['message']
        if 'data' not in message:
            return "Bad Request: Missing 'data'", 400

        payload = json.loads(base64.b64decode(message['data']).decode('utf-8'))
        logging.info(f"Decoded Pub/Sub payload: {payload}")

        bucket_name = payload['bucket']
        file_name = payload['name']
        gcs_uri = f"gs://{bucket_name}/{file_name}"

        # You can extract user_id from file name if encoded, or set as unknown
        user_id = "unknown"

        blob = storage_client.bucket(bucket_name).blob(file_name)
        image_bytes = blob.download_as_bytes()

        # Optionally check file size and type
        if not _validate_file_size(image_bytes):
            return f"Error: File too large", 413

        content_type = _detect_content_type(image_bytes)
        if content_type not in ALLOWED_CONTENT_TYPES:
            return f"Error: Content type {content_type} not allowed", 415

        if content_type.startswith("image/"):
            if not _perform_safety_check(image_bytes):
                return "Error: Unsafe content", 422
            if not _validate_content_structure(image_bytes):
                return "Error: Low text content", 422

        # Reupload to a different location (processed area)
        ext = content_type.split("/")[-1]
        new_name = f"processing-receipts/{user_id}-{uuid.uuid4()}.{ext}"
        new_blob = storage_client.bucket(BUCKET_NAME).blob(new_name)
        new_blob.upload_from_string(image_bytes, content_type=content_type)

        new_uri = f"gs://{BUCKET_NAME}/{new_name}"
        logger.info(f"Reuploaded to {new_uri}")

        # Publish for extraction
        message_payload = json.dumps({'file_path': new_uri, 'user_id': user_id}).encode('utf-8')
        publisher_client.publish(EXTRACTION_TOPIC, message_payload)

        return "OK", 200

    except Exception as e:
        logger.exception(f"Error handling GCS Pub/Sub push: {e}")
        return "Internal Server Error", 500



def ingestion_agent(request):
    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return "Invalid or missing JSON body", 400

        file_base64 = request_json.get('file_data_base64')
        user_id = request_json.get('user_id')

        if not file_base64 or not user_id:
            return 'Error: Missing user_id or file_data_base64.', 400

        image_content = base64.b64decode(file_base64)

        # Check user existence
        if not _verify_user_exists(user_id):
            return f"Error: Invalid or disabled user ID {user_id}", 403

        # File validations
        if not _validate_file_size(image_content):
            return f"Error: File exceeds size limit of {MAX_FILE_SIZE_MB}MB", 413

        content_type = _detect_content_type(image_content)
        if content_type not in ALLOWED_CONTENT_TYPES:
            return f"Error: Content type {content_type} not allowed", 415

        if content_type.startswith("image/"):
            if not _perform_safety_check(image_content):
                return "Error: Unsafe content detected in image", 422
            if not _validate_content_structure(image_content):
                return "Error: Insufficient textual content detected", 422

        # Upload to GCS
        ext = content_type.split("/")[-1]
        destination_blob_name = f"processing-receipts/{user_id}-{uuid.uuid4()}.{ext}"
        blob = storage_client.bucket(BUCKET_NAME).blob(destination_blob_name)
        blob.upload_from_string(image_content, content_type=content_type)
        gcs_uri = f"gs://{BUCKET_NAME}/{destination_blob_name}"
        logger.info(f"File uploaded to: {gcs_uri}")

        # Publish message to Pub/Sub
        message_payload = json.dumps({
            'file_path': gcs_uri,
            'user_id': user_id
        }).encode('utf-8')

        future = publisher_client.publish(EXTRACTION_TOPIC, message_payload)
        future.result()
        logger.info(f"Published message for user {user_id} to {EXTRACTION_TOPIC}")

        return "Receipt accepted and processing started.", 200

    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        return "Internal Server Error", 500


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
