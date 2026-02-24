import sys
import os
import requests
import json
import asyncio

# Set HF transfer for faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

async def test_api(filename):
    base_url = "http://localhost:8080/api/v1/fracture-detection"
    
    # Download a test wrist x-ray image
    test_image_path = f"/Users/hieunguyencong/Documents/FPT-University/CAPSTONE/1-Project/ai/bonix-ai-backend/{filename}"
    print(f"Testing local test image at {test_image_path}...")
        
    print(f"Testing Detect Endpoint: {base_url}/detect")
    try:
        with open(test_image_path, "rb") as f:
            files = {"file": (filename, f, "image/png")}
            # Note: This will trigger model download on first run
            print("Sending request to /detect (first run might take longer due to model download)...")
            detect_resp = requests.post(f"{base_url}/detect", files=files)
            
        print(f"Status: {detect_resp.status_code}")
        if detect_resp.status_code == 200:
            resp_data = detect_resp.json()
            # Print without flooding terminal with large floats
            print("Response successfully received.")
            print(f"Has Fracture: {resp_data['data']['has_fracture']}")
            print(f"Detections Count: {len(resp_data['data']['detections'])}")
            print(f"Processing Time: {resp_data['data']['processing_time_ms']:.2f} ms")
        else:
            print(f"Response: {detect_resp.text}")
    except Exception as e:
        print(f"Error testing detect endpoint: {e}")

async def main():
    base_url = "http://localhost:8080/api/v1/fracture-detection"
    print(f"Testing Health Endpoint: {base_url}/health")
    try:
        health_resp = requests.get(f"{base_url}/health")
        print(f"Status: {health_resp.status_code}")
        print(f"Response: {json.dumps(health_resp.json(), indent=2)}")
    except requests.exceptions.ConnectionError:
        print("\nAPI SERVER IS NOT RUNNING on port 8000.")
        return
        
    print("\n-----------------------------------\n")

    for file in ["test.png", "image.png"]:
        await test_api(file)
        print("\n-----------------------------------\n")

if __name__ == "__main__":
    asyncio.run(main())
