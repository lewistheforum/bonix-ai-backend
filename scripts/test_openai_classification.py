"""
Script to test exactly what OpenAI returns for Step 1 (wrist X-ray classification).
"""
import sys
import os
import io
import base64
from PIL import Image

# Ensure app is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.config import settings
from openai import AsyncOpenAI

def standardize_image(image_bytes: bytes) -> Image.Image:
    """Same logic as fracture_detection_service._standardize_image"""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode == "I;16" or img.mode == "I":
        img = img.point(lambda i: i * (1. / 256)).convert("L")
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

async def test_classification(image_path: str):
    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' not found.")
        return

    if not settings.OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY is not set in environment or .env file.")
        return

    print(f"Reading {image_path}...")
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Standardize image to JPEG using existing logic
    img = standardize_image(image_bytes)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=95)
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    classification_prompt = (
        "You are a medical imaging classification assistant. Carefully examine the attached image and determine whether it is an X-ray of the wrist, hand, or forearm. A wrist/hand/forearm X-ray typically: Is a grayscale radiographic image. Clearly shows bone structures such as the radius and ulna (forearm bones), carpal bones (wrist), metacarpals, or phalanges (fingers). May include an “R” or “L” marker indicating right or left side. Shows distinct skeletal anatomy rather than a blank, overexposed, or non-medical image. If the image clearly shows wrist, hand, or forearm bones in an X-ray format, respond with: YES If it does not clearly show these anatomical bone structures, respond with: NO Reply with ONLY the word 'YES' or 'NO'."
    )

    print(f"\nSending Step 1 classification request to OpenAI...")
    print(f"Prompt: {classification_prompt}")
    print(f"Image Base64 length: {len(base64_image)} chars")
    print(f"Using model: {settings.OPENAI_FRACTURE_DETECTION_MODEL}")
    
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    try:
        # Increase max_tokens slightly to see if OpenAI is trying to explain instead of strictly answering Yes/No
        response = await client.chat.completions.create(
            model=settings.OPENAI_FRACTURE_DETECTION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": classification_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            #max_tokens=200
        )
        
        result_text = response.choices[0].message.content.strip()
        print("\n" + "="*50)
        print("OPENAI RESPONSE:")
        print("="*50)
        print(result_text)
        print("="*50)
        
        if response.usage:
            print(f"\nToken usage: {response.usage.model_dump()}")
        
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_openai_classification.py <path_to_image>")
        sys.exit(1)
        
    import asyncio
    asyncio.run(test_classification(sys.argv[1]))
