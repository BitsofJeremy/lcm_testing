from diffusers import DiffusionPipeline, LCMScheduler
import os
import time
import torch
import uuid

# Start the timer
start_time = time.time()

image_dir = "output"

if not os.path.exists(image_dir):
        os.makedirs(image_dir)

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lcm_lora_id = "latent-consistency/lcm-lora-sdxl"

pipe = DiffusionPipeline.from_pretrained(model_id)

pipe.load_lora_weights(lcm_lora_id)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to(device="mps", dtype=torch.float16)

prompt = """
    Photo of a transparent sphere,
    containing a 3D realistic element
    of Godzilla destroying Tokyo.
    Realistic textures and colors.
    fires, explosions, helicopters in the background.
    The sphere is set against a white Christmas themed
    background to enhance its visibility.
"""

neg_prompt = """
    ugly, blurry, poor quality
"""

image_uuid = uuid.uuid4()

images = pipe(
    prompt=prompt,
    negative_prompt=neg_prompt,
    num_inference_steps=4,
    guidance_scale=1,
).images[0]

images.save(f'{image_dir}/{image_uuid}.png')

# End the timer
end_time = time.time()

print(f'Saved: {image_dir}/{image_uuid}.png')

# Calculate the total time taken
execution_time = end_time - start_time
print(f"The script took {execution_time} seconds to run.")
