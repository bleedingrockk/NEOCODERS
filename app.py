import os
import json
import base64
import uuid
import logging
from google.cloud import storage, vision, pubsub_v1
from google.cloud.exceptions import NotFound, Forbidden
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
        logger.info(f"Received request with content-type: {request.content_type}")
        request_json = request.get_json(silent=True)
        if not request_json:
            logger.error("Invalid or missing JSON body")
            return {"error": "Invalid or missing JSON body"}, 400

        logger.info(f"Request JSON: {request_json}")

        # ✅ Get and decode Pub/Sub message data
        message = request_json.get("message", {})
        data_base64 = message.get("data")
        if not data_base64:
            logger.error("Missing 'data' field in Pub/Sub message 2")
            return {"error": "Missing data"}, 400

        decoded_data = base64.b64decode(data_base64).decode("utf-8")
        print("Decoded data:",decoded_data)
        logger.info(f"Decoded Pub/Sub data: {decoded_data}")
        data_json = json.loads(decoded_data)

        bucket_name = data_json.get("bucket")
        file_name = data_json.get("name")

        if not bucket_name or not file_name:
            logger.error("Missing bucket or file name in decoded data")
            return {"error": "Missing bucket or file name 2 "}, 400
        # Direct format (for backward compatibility)
        else:
            bucket_name = request_json.get("bucket")
            file_name = request_json.get("name")
            logger.info(f"Direct format 2 - bucket: {bucket_name}, name: {file_name}")

        if not bucket_name or not file_name:
            logger.error("Missing bucket or file name in request")
            return {"error": "Missing bucket or file name"}, 400

        logger.info(f"Processing file: gs://{bucket_name}/{file_name}")

        # Check if blob exists and is accessible
        try:
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(file_name)
            
            # Check if blob exists
            if not blob.exists():
                logger.error(f"Blob does not exist: gs://{bucket_name}/{file_name}")
                return {"error": "File not found"}, 404
            
            logger.info(f"Blob exists and is accessible")
            
            # Reload blob to get latest metadata
            blob.reload()
            logger.info(f"Blob reloaded successfully. Size: {blob.size} bytes, Content-Type: {blob.content_type}")
            
        except NotFound as e:
            logger.error(f"Bucket or blob not found: {e}")
            return {"error": "Bucket or file not found"}, 404
        except Forbidden as e:
            logger.error(f"Access forbidden to blob: {e}")
            return {"error": "Access forbidden"}, 403
        except Exception as e:
            logger.error(f"Error accessing blob: {e}")
            return {"error": "Error accessing file"}, 500

        # Extract metadata and user_id
        metadata = blob.metadata or {}
        user_id = metadata.get("user_id")
        logger.info(f"Blob metadata: {metadata}")

        if not user_id:
            user_id = file_name.split("-")[0]
            logger.info(f"Extracted user_id from filename: {user_id}")

        # Validate file size before downloading
        if blob.size and blob.size > (MAX_FILE_SIZE_MB * 1024 * 1024):
            logger.error(f"File too large: {blob.size} bytes (max: {MAX_FILE_SIZE_MB * 1024 * 1024} bytes)")
            return {"error": f"File exceeds size limit of {MAX_FILE_SIZE_MB}MB"}, 413

        # Download image content with error handling
        try:
            logger.info("Starting image download...")
            image_content = blob.download_as_bytes()
            
            if not image_content:
                logger.error("Downloaded image content is empty")
                return {"error": "Downloaded file is empty"}, 422
                
            logger.info(f"Image downloaded successfully. Size: {len(image_content)} bytes")
            
            # Convert to base64
            file_data_base64 = base64.b64encode(image_content).decode("utf-8")
            logger.info(f"Image converted to base64. Base64 length: {len(file_data_base64)}")
            
        except Exception as e:
            logger.error(f"Failed to download image content: {e}")
            return {"error": "Failed to download file content"}, 500

        # Validate required fields
        if not file_data_base64 or not user_id:
            logger.error(f"Missing required fields - user_id: {bool(user_id)}, file_data_base64: {bool(file_data_base64)}")
            return {"error": "Missing user_id or file_data_base64"}, 400

        # Verify user exists
        if not _verify_user_exists(user_id):
            logger.error(f"User verification failed for: {user_id}")
            return {"error": f"Invalid or disabled user ID {user_id}"}, 403

        # Validate file size (double-check with actual content)
        if not _validate_file_size(image_content):
            logger.error(f"File size validation failed. Size: {len(image_content) / (1024 * 1024):.2f}MB")
            return {"error": f"File exceeds size limit of {MAX_FILE_SIZE_MB}MB"}, 413

        # Detect content type
        content_type = _detect_content_type(image_content)
        logger.info(f"Detected content type: {content_type}")

        if content_type not in ALLOWED_CONTENT_TYPES:
            logger.error(f"Content type not allowed: {content_type}")
            return {"error": f"Content type {content_type} not allowed"}, 415

        # Perform safety and content checks for images
        if content_type.startswith("image/"):
            logger.info("Performing safety check for image...")
            if not _perform_safety_check(image_content):
                logger.error("Safety check failed")
                return {"error": "Unsafe content detected in image"}, 422
                
            logger.info("Performing content structure validation...")
            if not _validate_content_structure(image_content):
                logger.error("Content structure validation failed")
                return {"error": "Insufficient textual content detected"}, 422

        # Upload to processing bucket
        try:
            ext = content_type.split("/")[-1]
            destination_blob_name = f"processing-receipts/{user_id}-{uuid.uuid4()}.{ext}"
            
            logger.info(f"Uploading to destination: gs://{BUCKET_NAME}/{destination_blob_name}")
            
            new_blob = storage_client.bucket(BUCKET_NAME).blob(destination_blob_name)
            new_blob.upload_from_string(image_content, content_type=content_type)
            
            gcs_uri = f"gs://{BUCKET_NAME}/{destination_blob_name}"
            logger.info(f"File uploaded successfully to: {gcs_uri}")
            
        except Exception as e:
            logger.error(f"Failed to upload file to processing bucket: {e}")
            return {"error": "Failed to upload file for processing"}, 500

        # Publish to Pub/Sub
        try:
            message_payload = json.dumps({
                'file_path': gcs_uri,
                'user_id': user_id
            }).encode('utf-8')

            logger.info(f"Publishing message to topic: {EXTRACTION_TOPIC}")
            future = publisher_client.publish(EXTRACTION_TOPIC, message_payload)
            message_id = future.result()
            
            logger.info(f"Message published successfully with ID: {message_id} for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to publish message to Pub/Sub: {e}")
            return {"error": "Failed to queue file for processing"}, 500

        return {"message": "Receipt accepted and processing started", "message_id": message_id}, 200

    except Exception as e:
        logger.exception(f"Unhandled error in ingestion_agent: {e}")
        return {"error": "Internal Server Error"}, 500

def _verify_user_exists(user_id: str) -> bool:
    try:
        logger.info(f"Verifying user exists: {user_id}")
        user_record = auth.get_user(user_id)
        is_enabled = not user_record.disabled
        logger.info(f"User verification result - exists: True, enabled: {is_enabled}")
        return is_enabled
    except firebase_exceptions.UserNotFoundError:
        logger.warning(f"User not found: {user_id}")
        return False
    except Exception as e:
        logger.error(f"User verification failed: {e}")
        return False

def _validate_file_size(image_bytes: bytes) -> bool:
    size_mb = len(image_bytes) / (1024 * 1024)
    logger.info(f"File size validation: {size_mb:.2f}MB (max: {MAX_FILE_SIZE_MB}MB)")
    is_valid = size_mb <= MAX_FILE_SIZE_MB
    logger.info(f"File size validation result: {is_valid}")
    return is_valid

def _detect_content_type(image_bytes: bytes) -> str:
    try:
        import imghdr
        
        # Log first few bytes for debugging
        logger.info(f"File signature (first 20 bytes): {image_bytes[:20]}")
        
        fmt = imghdr.what(None, h=image_bytes)
        logger.info(f"imghdr detected format: {fmt}")
        
        if fmt == 'jpeg':
            return 'image/jpeg'
        elif fmt == 'png':
            return 'image/png'
        elif fmt == 'webp':
            return 'image/webp'
            
        # Check for PDF
        if image_bytes.startswith(b'%PDF'):
            logger.info("Detected PDF file")
            return 'application/pdf'
            
        logger.warning("Could not detect content type, defaulting to octet-stream")
        return 'application/octet-stream'
        
    except Exception as e:
        logger.error(f"Error detecting content type: {e}")
        return 'application/octet-stream'

def _perform_safety_check(image_bytes: bytes) -> bool:
    try:
        logger.info("Starting Vision API safety check...")
        image = vision.Image(content=image_bytes)
        response = vision_client.safe_search_detection(image=image)
        
        if response.error.message:
            logger.error(f"Vision API error: {response.error.message}")
            return False
            
        safe = response.safe_search_annotation
        logger.info(f"Safety check results - adult: {safe.adult}, violence: {safe.violence}, racy: {safe.racy}")
        
        unsafe_categories = []
        if safe.adult in [vision.Likelihood.LIKELY, vision.Likelihood.VERY_LIKELY]:
            unsafe_categories.append("adult")
        if safe.violence in [vision.Likelihood.LIKELY, vision.Likelihood.VERY_LIKELY]:
            unsafe_categories.append("violence")
        if safe.racy in [vision.Likelihood.LIKELY, vision.Likelihood.VERY_LIKELY]:
            unsafe_categories.append("racy")
            
        if unsafe_categories:
            logger.warning(f"Unsafe image content detected: {unsafe_categories}")
            return False
            
        logger.info("Safety check passed")
        return True
        
    except Exception as e:
        logger.error(f"Safety check error: {e}")
        # Return True on error to avoid blocking legitimate content
        return True

def _validate_content_structure(image_bytes: bytes) -> bool:
    try:
        logger.info("Starting content structure validation...")
        image = vision.Image(content=image_bytes)
        response = vision_client.text_detection(image=image)
        
        if response.error.message:
            logger.warning(f"Text detection error: {response.error.message}")
            return True  # Don't block on API errors
            
        annotations = response.text_annotations
        
        if not annotations:
            logger.warning("No text annotations found in image")
            return False
            
        text_content = annotations[0].description.strip()
        text_length = len(text_content)
        
        logger.info(f"Detected text length: {text_length} characters")
        logger.info(f"Text preview (first 100 chars): {text_content[:100]}")
        
        if text_length < 10:
            logger.warning("Insufficient text content in image")
            return False
            
        logger.info("Content structure validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Text detection failed: {e}")
        # Return True on error to avoid blocking legitimate content
        return True

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)
