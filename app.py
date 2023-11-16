from diffusers import DiffusionPipeline, LCMScheduler
import os
import time
import torch
import uuid

# For out of memory issues
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Start the timer
start_time = time.time()

# Directory for the images
image_dir = "output"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# SDXL model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# LCM-LoRA model
lcm_lora_id = "latent-consistency/lcm-lora-sdxl"

# Pipe initiate
pipe = DiffusionPipeline.from_pretrained(model_id)

# Load the LoRA
pipe.load_lora_weights(lcm_lora_id)

# Setup scheduler
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# Device to send to ['cuda' for Nvidia, 'mps' for Apple Silicon]
device = 'mps'
pipe.to(device=device, dtype=torch.float16)

# Positive prompt text
prompt = """
    Photo of a transparent sphere,
    containing a 3D realistic element
    of Godzilla destroying Tokyo.
    Realistic textures and colors.
    fires, explosions, helicopters in the background.
    The sphere is set against a white Christmas themed
    background to enhance its visibility.
"""

# Negative prompt text
neg_prompt = """
    ugly, blurry, poor quality
"""

# Manually set the seed or set to -1 to randomize it.
img_seed = -1
if img_seed == -1:
    generator = None
else:
    generator = torch.Generator(device=device).manual_seed(img_seed)

# Set the batch count [number of images to generate at once]
# Note: more than one image ignores manual seed for some reason
batch_num = 1

# Generate your images
images = pipe(
    prompt=prompt,
    negative_prompt=neg_prompt,
    num_inference_steps=4,
    guidance_scale=1,
    generator=generator,
    num_images_per_prompt=batch_num,
    ).images

# Save images to directory with UUID and Seed for file name
for image in images:
    image_uuid = uuid.uuid4()
    image.save(f'{image_dir}/{image_uuid}-{torch.Generator(device=device).seed()}.png')
    print(f'Saved: {image_dir}/{image_uuid}-{torch.Generator(device=device).seed()}.png')

# End the timer
end_time = time.time()

# Calculate the total time taken
execution_time = end_time - start_time
print(f"The script took {execution_time} seconds to run.")
