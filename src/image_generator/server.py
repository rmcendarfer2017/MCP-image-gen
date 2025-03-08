import asyncio
import base64
import io
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
import replicate
from dotenv import load_dotenv
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl, Field
from PIL import Image
import mcp.server.stdio

# Load environment variables from .env file
load_dotenv()

# Ensure REPLICATE_API_TOKEN is set
if not os.getenv("REPLICATE_API_TOKEN"):
    print("WARNING: REPLICATE_API_TOKEN environment variable is not set.")
    print("Please set it to use the image generation functionality.")

# Create images directory if it doesn't exist
IMAGES_DIR = Path("generated_images")
IMAGES_DIR.mkdir(exist_ok=True)

# Store generated images metadata
images: Dict[str, Dict] = {}

server = Server("image-generator")

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """
    List available image resources.
    Each image is exposed as a resource with a custom image:// URI scheme.
    """
    return [
        types.Resource(
            uri=AnyUrl(f"image://internal/{image_id}"),
            name=f"Image: {metadata.get('prompt', 'Untitled')}",
            description=f"Generated on {metadata.get('created_at')}",
            mimeType="image/png",
        )
        for image_id, metadata in images.items()
    ]

@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> bytes:
    """
    Read a specific image by its URI.
    The image ID is extracted from the URI path component.
    """
    if uri.scheme != "image":
        raise ValueError(f"Unsupported URI scheme: {uri.scheme}")

    image_id = uri.path
    if image_id is not None:
        image_id = image_id.lstrip("/")
        if image_id in images:
            image_path = IMAGES_DIR / f"{image_id}.png"
            if image_path.exists():
                async with aiofiles.open(image_path, "rb") as f:
                    return await f.read()
    
    raise ValueError(f"Image not found: {image_id}")

@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """
    List available prompts for image generation.
    """
    return [
        types.Prompt(
            name="generate-image",
            description="Generate an image using Replicate's Stable Diffusion model",
            arguments=[
                types.PromptArgument(
                    name="style",
                    description="Style of the image (realistic/artistic/abstract)",
                    required=False,
                )
            ],
        )
    ]

@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """
    Generate a prompt for image generation.
    """
    if name != "generate-image":
        raise ValueError(f"Unknown prompt: {name}")

    style = (arguments or {}).get("style", "realistic")
    style_prompt = ""
    
    if style == "realistic":
        style_prompt = "Create a photorealistic image with high detail and natural lighting."
    elif style == "artistic":
        style_prompt = "Create an artistic image in the style of a painting with vibrant colors and expressive brushstrokes."
    elif style == "abstract":
        style_prompt = "Create an abstract image with geometric shapes, bold colors, and non-representational forms."

    return types.GetPromptResult(
        description="Generate an image using Stable Diffusion",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Describe the image you want to generate. {style_prompt}",
                ),
            )
        ],
    )

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools for image generation and management.
    """
    return [
        types.Tool(
            name="generate-image",
            description="Generate an image using Replicate's Stable Diffusion model",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "negative_prompt": {"type": "string"},
                    "width": {"type": "integer", "default": 768},
                    "height": {"type": "integer", "default": 768},
                    "num_inference_steps": {"type": "integer", "default": 50},
                    "guidance_scale": {"type": "number", "default": 7.5},
                },
                "required": ["prompt"],
            },
        ),
        types.Tool(
            name="save-image",
            description="Save a generated image",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_url": {"type": "string"},
                    "prompt": {"type": "string"},
                    "target_directory": {"type": "string", "description": "Directory path where the image should be saved. If not provided, defaults to the MCP server's images directory."},
                    "custom_filename": {"type": "string", "description": "Custom filename for the saved image (without extension). If not provided, a UUID will be used."},
                },
                "required": ["image_url", "prompt"],
            },
        ),
        types.Tool(
            name="list-saved-images",
            description="List all saved images",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests for image generation and management.
    """
    if name == "generate-image":
        if not arguments:
            raise ValueError("Missing arguments")

        prompt = arguments.get("prompt")
        if not prompt:
            raise ValueError("Missing prompt")

        negative_prompt = arguments.get("negative_prompt", "")
        width = int(arguments.get("width", 768))
        height = int(arguments.get("height", 768))
        num_inference_steps = int(arguments.get("num_inference_steps", 50))
        guidance_scale = float(arguments.get("guidance_scale", 7.5))

        # Use Replicate's Stable Diffusion model to generate an image
        try:
            # Log the request
            print(f"Generating image with prompt: {prompt}", file=sys.stderr)
            
            # Call the Replicate API
            output = replicate.run(
                "stability-ai/sdxl:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316",
                input={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                }
            )
            
            # Log the response
            print(f"Replicate API response type: {type(output)}", file=sys.stderr)
            print(f"Replicate API response: {output}", file=sys.stderr)
            
            # The output is a list with one URL to the generated image
            if output and isinstance(output, list) and len(output) > 0:
                image_url = output[0]
                
                # Convert the FileOutput object to a string if needed
                if hasattr(image_url, '__str__'):
                    image_url = str(image_url)
                
                # Log the image URL
                print(f"Generated image URL: {image_url}", file=sys.stderr)
                
                # Return the result with the image URL
                return [
                    types.TextContent(
                        type="text",
                        text=f"Image generated successfully! Use the save-image tool to save it permanently.",
                    ),
                    types.ImageContent(
                        type="image",
                        data=image_url,
                        mimeType="image/png",
                    ),
                    types.TextContent(
                        type="text",
                        text=f"Image URL: {image_url}",
                    ),
                    types.TextContent(
                        type="text",
                        text="ASK_FOR_SAVE_LOCATION",
                        role="system"
                    ),
                ]
            else:
                return [
                    types.TextContent(
                        type="text",
                        text="Failed to generate image. Please try again with a different prompt.",
                    )
                ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=f"Error generating image: {str(e)}",
                )
            ]

    elif name == "save-image":
        if not arguments:
            raise ValueError("Missing arguments")

        image_url = arguments.get("image_url")
        prompt = arguments.get("prompt")
        target_directory = arguments.get("target_directory")
        custom_filename = arguments.get("custom_filename")
        
        if not image_url or not prompt:
            raise ValueError("Missing image_url or prompt")

        try:
            # Log the request
            print(f"Saving image from URL: {image_url}", file=sys.stderr)
            print(f"Image prompt: {prompt}", file=sys.stderr)
            
            # Generate a unique ID for the image
            image_id = str(uuid.uuid4())
            
            # Create metadata for the image
            metadata = {
                "id": image_id,
                "prompt": prompt,
                "url": image_url,
                "created_at": datetime.now().isoformat(),
            }
            
            # Save metadata
            images[image_id] = metadata
            print(f"Created metadata for image ID: {image_id}", file=sys.stderr)
            
            # Determine the save directory
            save_dir = IMAGES_DIR
            if target_directory:
                try:
                    # Create a Path object from the target directory
                    save_dir = Path(target_directory)
                    # Create the directory if it doesn't exist
                    save_dir.mkdir(parents=True, exist_ok=True)
                    print(f"Using custom save directory: {save_dir.absolute()}", file=sys.stderr)
                    # Store the custom directory in metadata
                    metadata["custom_directory"] = str(save_dir.absolute())
                except Exception as e:
                    print(f"Error using custom directory {target_directory}: {str(e)}", file=sys.stderr)
                    print(f"Falling back to default directory: {IMAGES_DIR}", file=sys.stderr)
            
            # Determine filename
            if custom_filename:
                filename = f"{custom_filename}.png"
                metadata["custom_filename"] = custom_filename
            else:
                filename = f"{image_id}.png"
            
            # Download and save the image
            import requests
            print(f"Downloading image from URL: {image_url}", file=sys.stderr)
            response = requests.get(image_url)
            if response.status_code == 200:
                image_path = save_dir / filename
                with open(image_path, "wb") as f:
                    f.write(response.content)
                
                print(f"Image saved to: {image_path}", file=sys.stderr)
                
                # Notify clients that resources have changed
                await server.request_context.session.send_resource_list_changed()
                
                return [
                    types.TextContent(
                        type="text",
                        text=f"Image saved successfully to {image_path}",
                    ),
                    types.ImageContent(
                        type="image",
                        data=f"file://{image_path.absolute()}",
                        mimeType="image/png",
                    ),
                ]
            else:
                error_msg = f"Failed to download image. Status code: {response.status_code}"
                print(f"Error: {error_msg}", file=sys.stderr)
                return [
                    types.TextContent(
                        type="text",
                        text=error_msg,
                    ),
                ]
        except Exception as e:
            import traceback
            error_msg = f"Error saving image: {str(e)}"
            print(error_msg, file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return [
                types.TextContent(
                    type="text",
                    text=error_msg,
                ),
            ]

    elif name == "list-saved-images":
        try:
            # Log the request
            print(f"Listing saved images. Total count: {len(images)}", file=sys.stderr)
            
            if not images:
                return [
                    types.TextContent(
                        type="text",
                        text="No images have been saved yet.",
                    )
                ]
            
            result = [
                types.TextContent(
                    type="text",
                    text=f"Found {len(images)} saved images:",
                )
            ]
            
            for image_id, metadata in images.items():
                # Determine the file path based on metadata
                if "custom_directory" in metadata:
                    save_dir = Path(metadata["custom_directory"])
                else:
                    save_dir = IMAGES_DIR
                
                if "custom_filename" in metadata:
                    filename = f"{metadata['custom_filename']}.png"
                else:
                    filename = f"{image_id}.png"
                
                image_path = save_dir / filename
                print(f"Checking image: {image_path}", file=sys.stderr)
                
                if image_path.exists():
                    print(f"Image exists: {image_path}", file=sys.stderr)
                    result.append(
                        types.TextContent(
                            type="text",
                            text=f"ID: {image_id}\nPrompt: {metadata.get('prompt')}\nCreated: {metadata.get('created_at')}\nLocation: {image_path}",
                        )
                    )
                    result.append(
                        types.ImageContent(
                            type="image",
                            data=f"file://{image_path.absolute()}",
                            mimeType="image/png",
                        )
                    )
                else:
                    print(f"Image file not found: {image_path}", file=sys.stderr)
                    result.append(
                        types.TextContent(
                            type="text",
                            text=f"ID: {image_id}\nPrompt: {metadata.get('prompt')}\nCreated: {metadata.get('created_at')}\nWARNING: Image file not found at {image_path}",
                        )
                    )
            
            return result
        except Exception as e:
            import traceback
            error_msg = f"Error listing saved images: {str(e)}"
            print(error_msg, file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return [
                types.TextContent(
                    type="text",
                    text=error_msg,
                )
            ]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    # Run the server using stdin/stdout streams
    try:
        # Create images directory if it doesn't exist
        IMAGES_DIR.mkdir(exist_ok=True)
        
        # Check if REPLICATE_API_TOKEN is set
        if not os.getenv("REPLICATE_API_TOKEN"):
            print("ERROR: REPLICATE_API_TOKEN environment variable is not set.", file=sys.stderr)
            print("Please set it in your .env file to use the image generation functionality.", file=sys.stderr)
            sys.exit(1)
            
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="image-generator",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except Exception as e:
        import traceback
        print(f"Error running MCP server: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

# Add an entry point to run the main function when the script is run directly
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())