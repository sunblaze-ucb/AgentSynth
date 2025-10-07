"""
Script for uploading training files to Mistral API for fine-tuning.
This script uploads JSONL files created by convert_to_oai_training_file.py
"""

import os
import json
import time
from typing import List, Dict, Any
from pathlib import Path
import requests
from tqdm import tqdm

# ============== CONFIG ==============
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")  # Set this environment variable
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
UPLOAD_DIR = "mistral_data_files/"  # Directory containing the JSONL files
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
# ====================================

class MistralUploader:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = MISTRAL_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def upload_file(self, file_path: str) -> Dict[str, Any]:
        """Upload a file to Mistral API"""
        print(f"Uploading {file_path}...")
        
        # First, upload the file
        upload_url = f"{self.base_url}/files"
        
        with open(file_path, 'rb') as f:
            files = {
                'file': (os.path.basename(file_path), f, 'application/json')
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            for attempt in range(MAX_RETRIES):
                try:
                    response = requests.post(upload_url, files=files, headers=headers)
                    response.raise_for_status()
                    file_data = response.json()
                    print(f"✓ File uploaded successfully: {file_data['id']}")
                    return file_data
                    
                except requests.exceptions.RequestException as e:
                    print(f"✗ Upload attempt {attempt + 1} failed: {e}")
                    if attempt < MAX_RETRIES - 1:
                        print(f"Retrying in {RETRY_DELAY} seconds...")
                        time.sleep(RETRY_DELAY)
                    else:
                        raise
    
    def create_fine_tuning_job(self, training_file_id: str, model: str = "mistral-tiny", 
                              hyperparameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a fine-tuning job"""
        print(f"Creating fine-tuning job for file {training_file_id}...")
        
        job_url = f"{self.base_url}/fine_tuning/jobs"
        
        payload = {
            "model": model,
            "training_file": training_file_id
        }
        
        if hyperparameters:
            payload["hyperparameters"] = hyperparameters
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(job_url, headers=self.headers, json=payload)
                response.raise_for_status()
                job_data = response.json()
                print(f"✓ Fine-tuning job created: {job_data['id']}")
                return job_data
                
            except requests.exceptions.RequestException as e:
                print(f"✗ Job creation attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    print(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise
    
    def get_fine_tuning_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a fine-tuning job"""
        job_url = f"{self.base_url}/fine_tuning/jobs/{job_id}"
        
        try:
            response = requests.get(job_url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"✗ Failed to get job status: {e}")
            raise
    
    def list_fine_tuning_jobs(self) -> List[Dict[str, Any]]:
        """List all fine-tuning jobs"""
        jobs_url = f"{self.base_url}/fine_tuning/jobs"
        
        try:
            response = requests.get(jobs_url, headers=self.headers)
            response.raise_for_status()
            return response.json().get("data", [])
        except requests.exceptions.RequestException as e:
            print(f"✗ Failed to list jobs: {e}")
            raise


def find_jsonl_files(directory: str) -> List[str]:
    """Find all JSONL files in the directory"""
    jsonl_files = []
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist!")
        return jsonl_files
    
    for file in os.listdir(directory):
        if file.endswith('.jsonl'):
            jsonl_files.append(os.path.join(directory, file))
    
    jsonl_files.sort()
    return jsonl_files


def validate_file_size(file_path: str, max_size_mb: int = 512) -> bool:
    """Validate that file size is within limits"""
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        print(f"⚠️  Warning: {file_path} is {file_size_mb:.2f} MB (exceeds {max_size_mb} MB limit)")
        return False
    print(f"✓ {file_path}: {file_size_mb:.2f} MB")
    return True


def main():
    if not MISTRAL_API_KEY:
        print("❌ Error: MISTRAL_API_KEY environment variable not set!")
        print("Please set it with: export MISTRAL_API_KEY='your_api_key_here'")
        return
    
    uploader = MistralUploader(MISTRAL_API_KEY)
    
    # Find all JSONL files
    jsonl_files = find_jsonl_files(UPLOAD_DIR)
    if not jsonl_files:
        print(f"❌ No JSONL files found in {UPLOAD_DIR}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files:")
    for file in jsonl_files:
        print(f"  - {file}")
    
    # Validate file sizes
    print("\nValidating file sizes...")
    valid_files = []
    for file in jsonl_files:
        if validate_file_size(file):
            valid_files.append(file)
    
    if not valid_files:
        print("❌ No valid files to upload!")
        return
    
    # Upload files and create fine-tuning jobs
    uploaded_files = []
    created_jobs = []
    
    print(f"\nUploading {len(valid_files)} files...")
    for file_path in tqdm(valid_files, desc="Uploading files"):
        try:
            file_data = uploader.upload_file(file_path)
            uploaded_files.append({
                "file_path": file_path,
                "file_id": file_data["id"],
                "file_data": file_data
            })
        except Exception as e:
            print(f"❌ Failed to upload {file_path}: {e}")
            continue
    
    if not uploaded_files:
        print("❌ No files were successfully uploaded!")
        return
    
    print(f"\n✓ Successfully uploaded {len(uploaded_files)} files")
    
    # # Create fine-tuning jobs
    # print("\nCreating fine-tuning jobs...")
    # for file_info in tqdm(uploaded_files, desc="Creating jobs"):
    #     try:
    #         job_data = uploader.create_fine_tuning_job(
    #             training_file_id=file_info["file_id"],
    #             model="mistral-tiny",  # You can change this to other models
    #             hyperparameters={
    #                 "n_epochs": 3,  # Adjust as needed
    #                 "learning_rate": 1e-5,  # Adjust as needed
    #             }
    #         )
    #         created_jobs.append({
    #             "file_path": file_info["file_path"],
    #             "file_id": file_info["file_id"],
    #             "job_id": job_data["id"],
    #             "job_data": job_data
    #         })
    #     except Exception as e:
    #         print(f"❌ Failed to create job for {file_info['file_path']}: {e}")
    #         continue
    
    # if not created_jobs:
    #     print("❌ No fine-tuning jobs were created!")
    #     return
    
    # print(f"\n✓ Successfully created {len(created_jobs)} fine-tuning jobs")
    
    # # Save job information
    # job_info_file = os.path.join(UPLOAD_DIR, "mistral_jobs.json")
    # with open(job_info_file, "w") as f:
    #     json.dump(created_jobs, f, indent=2)
    
    # print(f"\n📋 Job information saved to {job_info_file}")
    # print("\nFine-tuning jobs created:")
    # for job in created_jobs:
    #     print(f"  - Job ID: {job['job_id']}")
    #     print(f"    File: {job['file_path']}")
    #     print(f"    Status: {job['job_data'].get('status', 'unknown')}")
    #     print()
    
    print("🔍 You can monitor job progress with:")
    print("  python -c \"from upload_to_mistral import MistralUploader; import os; u = MistralUploader(os.getenv('MISTRAL_API_KEY')); print(u.list_fine_tuning_jobs())\"")


if __name__ == "__main__":
    main()
