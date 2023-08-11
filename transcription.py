import argparse
from google.cloud import storage
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    return f"gs://{bucket_name}/{destination_blob_name}"

def transcribe_batch_dynamic_batching_v2(
    project_id: str,
    local_file_path: str,
    bucket_name: str,
) -> str:
    """Transcribes audio from a file.

    Args:
        project_id: The Google Cloud project ID.
        local_file_path: Path to the local audio file.
        bucket_name: Name of the GCS bucket where the audio will be uploaded.

    Returns:
        Path to the saved transcription txt file.
    """
    # Upload the audio file to GCS
    destination_file_name = local_file_path.split('/')[-1]
    gcs_uri = upload_to_gcs(bucket_name, local_file_path, destination_file_name)
    
    # Instantiate a Speech client
    client = SpeechClient()

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["hi-IN"],
        model="long",
    )

    file_metadata = cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)

    request = cloud_speech.BatchRecognizeRequest(
        recognizer=f"projects/{project_id}/locations/global/recognizers/_",
        config=config,
        files=[file_metadata],
        recognition_output_config=cloud_speech.RecognitionOutputConfig(
            inline_response_config=cloud_speech.InlineOutputConfig(),
        ),
        processing_strategy=cloud_speech.BatchRecognizeRequest.ProcessingStrategy.DYNAMIC_BATCHING,
    )

    # Transcribe the audio into text
    operation = client.batch_recognize(request=request)

    print("Waiting for operation to complete...")
    response = operation.result(timeout=9999)

    transcript_txt_path = local_file_path.replace(".wav", "_transcription.txt")
    with open(transcript_txt_path, "w") as txt_file:
        for result in response.results[gcs_uri].transcript.results:
            txt_file.write(f"{result.alternatives[0].transcript}\n")

    return transcript_txt_path
