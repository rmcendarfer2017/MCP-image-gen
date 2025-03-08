import os
import replicate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if REPLICATE_API_TOKEN is set
if not os.getenv("REPLICATE_API_TOKEN"):
    print("WARNING: REPLICATE_API_TOKEN environment variable is not set.")
    print("Please set it to use the image generation functionality.")
    exit(1)

print("Generating image...")
output = replicate.run(
    "stability-ai/sdxl:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316",
    input={
        "prompt": "A beautiful horse running in a field, photorealistic",
        "negative_prompt": "distorted, blurry, low quality, deformed",
        "width": 768,
        "height": 768,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
    }
)

print("Output:", output)
if output and isinstance(output, list) and len(output) > 0:
    image_url = output[0]
    print(f"Image URL: {image_url}")
else:
    print("Failed to generate image.")
