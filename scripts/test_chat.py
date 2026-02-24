import requests
import json
import sys
import time

def test_chat():
    url = "http://localhost:8080/api/v1/rag/chat"
    headers = {"Content-Type": "application/json"}
    
    # Simple query to trigger context retrieval
    data = {
        "message": "what can you support?",
        "conversation_id": None
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            print("\nShared Success! Response received.")
            print("-" * 30)
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
            print("-" * 30)
            print("RESPONSE TEXT:")
            print(response.json()['data']['response'])
            print("-" * 30)
            print("\n>>> NOW CHECK YOUR 'python3 main.py' TERMINAL <<<")
            print("You should see the 'Retrieved context...' log message there.")
        else:
            print(f"\nError: Status code {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to localhost:8080.")
        print("Make sure 'python3 main.py' is running in another terminal!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    test_chat()
