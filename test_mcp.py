import os
import sys
from pathlib import Path
import replicate
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

# Import the server module directly
from src.image_generator.server import handle_call_tool, images, IMAGES_DIR

# Load environment variables from .env file
load_dotenv()

# Check if REPLICATE_API_TOKEN is set
if not os.getenv("REPLICATE_API_TOKEN"):
    print("WARNING: REPLICATE_API_TOKEN environment variable is not set.")
    print("Please set it to use the image generation functionality.")
    exit(1)

# Create images directory if it doesn't exist
IMAGES_DIR.mkdir(exist_ok=True)

async def test_generate_image():
    print("Testing generate-image tool...")
    arguments = {
        "prompt": "A cute mouse, detailed fur, photorealistic",
        "negative_prompt": "distorted, blurry, low quality, deformed",
        "width": 768,
        "height": 768,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
    }
    
    try:
        result = await handle_call_tool("generate-image", arguments)
        print("Result type:", type(result))
        print("Result length:", len(result))
        for i, item in enumerate(result):
            print(f"Item {i} type:", type(item))
            print(f"Item {i} attributes:", dir(item))
            print(f"Item {i} dict:", item.dict())
    except Exception as e:
        print("Error calling tool:", str(e))

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_generate_image())
