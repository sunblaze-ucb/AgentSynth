"""
Script for monitoring Mistral fine-tuning job status.
"""

import os
import json
import time
from typing import List, Dict, Any
from upload_to_mistral import MistralUploader

# ============== CONFIG ==============
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
JOB_INFO_FILE = "oai_data_files/mistral_jobs.json"
# ====================================

def load_job_info() -> List[Dict[str, Any]]:
    """Load job information from file"""
    if not os.path.exists(JOB_INFO_FILE):
        print(f"❌ Job info file not found: {JOB_INFO_FILE}")
        return []
    
    with open(JOB_INFO_FILE, "r") as f:
        return json.load(f)

def monitor_jobs(continuous: bool = False, interval: int = 60):
    """Monitor fine-tuning jobs"""
    if not MISTRAL_API_KEY:
        print("❌ Error: MISTRAL_API_KEY environment variable not set!")
        return
    
    uploader = MistralUploader(MISTRAL_API_KEY)
    job_info = load_job_info()
    
    if not job_info:
        print("❌ No job information found!")
        return
    
    print(f"Monitoring {len(job_info)} fine-tuning jobs...")
    
    while True:
        print(f"\n{'='*60}")
        print(f"Status check at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        all_completed = True
        
        for job in job_info:
            job_id = job["job_id"]
            file_path = job["file_path"]
            
            try:
                status_data = uploader.get_fine_tuning_job_status(job_id)
                status = status_data.get("status", "unknown")
                
                print(f"\nJob ID: {job_id}")
                print(f"File: {os.path.basename(file_path)}")
                print(f"Status: {status}")
                
                if status == "succeeded":
                    print("✅ Job completed successfully!")
                    if "fine_tuned_model" in status_data:
                        print(f"Model: {status_data['fine_tuned_model']}")
                elif status == "failed":
                    print("❌ Job failed!")
                    if "error" in status_data:
                        print(f"Error: {status_data['error']}")
                elif status in ["validating_files", "queued", "running"]:
                    print(f"⏳ Job is {status}...")
                    all_completed = False
                else:
                    print(f"❓ Unknown status: {status}")
                    all_completed = False
                
            except Exception as e:
                print(f"❌ Error checking job {job_id}: {e}")
                all_completed = False
        
        if all_completed:
            print("\n🎉 All jobs completed!")
            break
        
        if not continuous:
            break
        
        print(f"\nWaiting {interval} seconds before next check...")
        time.sleep(interval)

def list_all_jobs():
    """List all fine-tuning jobs from Mistral API"""
    if not MISTRAL_API_KEY:
        print("❌ Error: MISTRAL_API_KEY environment variable not set!")
        return
    
    uploader = MistralUploader(MISTRAL_API_KEY)
    
    try:
        jobs = uploader.list_fine_tuning_jobs()
        print(f"Found {len(jobs)} fine-tuning jobs:")
        
        for job in jobs:
            print(f"\nJob ID: {job['id']}")
            print(f"Status: {job.get('status', 'unknown')}")
            print(f"Model: {job.get('model', 'unknown')}")
            print(f"Created: {job.get('created_at', 'unknown')}")
            if job.get('fine_tuned_model'):
                print(f"Fine-tuned Model: {job['fine_tuned_model']}")
            if job.get('error'):
                print(f"Error: {job['error']}")
                
    except Exception as e:
        print(f"❌ Error listing jobs: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor Mistral fine-tuning jobs")
    parser.add_argument("--continuous", "-c", action="store_true", 
                       help="Monitor continuously (default: single check)")
    parser.add_argument("--interval", "-i", type=int, default=60,
                       help="Interval between checks in seconds (default: 60)")
    parser.add_argument("--list-all", "-l", action="store_true",
                       help="List all jobs from API instead of monitoring saved jobs")
    
    args = parser.parse_args()
    
    if args.list_all:
        list_all_jobs()
    else:
        monitor_jobs(continuous=args.continuous, interval=args.interval)

if __name__ == "__main__":
    main()
