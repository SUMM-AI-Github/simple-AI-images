import os
import torch
import logging
import json
from diffusers import AutoPipelineForText2Image, DiffusionPipeline, FluxPipeline
from PIL import Image
from safetensors.torch import load_file


# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FluxImageGenerator:
    def __init__(self, config: dict, device: str = 'cuda'):
        """
        Initializes the Flux Image Generator by loading configuration and setting up the pipeline.
        
        Args:
            config: The config object(dict) containing the required fields.
            device (str): The device to use for inference ('cuda' or 'cpu').
        """
        self.config = config
        self.device = device
        self.pipeline = self._setup_pipeline()


    def _setup_pipeline(self):
        """
        Sets up the FLUX pipeline by loading pre-trained models and LoRA weights.
        
        Returns:
            FluxPipeline: Configured FLUX pipeline.
        """
        logger.info("Started loading FLUX pipeline")

        # Load the FLUX pipeline
        pipeline = FluxPipeline.from_pretrained(
            self.config["pretrained_model_name"],
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="balanced" if self.device == "cuda" else None,
        )

        # Load LoRA weights
        lora_folder = self.config["lora_folder"]
        lora_weights = self.config["lora_weights"]
        pipeline.load_lora_weights(lora_folder, weight_name=lora_weights, adapter_name="lora")
        pipeline.set_adapters("lora")
        pipeline.fuse_lora(adapter_names=["lora"])

        logger.info("Finished loading FLUX pipeline")
        return pipeline

    def generate_images(self, prompt: str, height: int, width: int, num_images_per_prompt: int, num_inference_steps: int = 5, seed: int = None):
        """
        Generates images using the FLUX pipeline.
        
        Args:
            prompt (str): The text prompt for generating the image.
            height (int): Image height.
            width (int): Image width.
            num_images_per_prompt (int): Number of images to generate.
            num_inference_steps (int): Number of inference steps (default is 5).
            seed (int): Seed value for explicitly doing regeneration, else no seed is used.
        
        Returns:
            list[PIL.Image.Image]: List of generated images as PIL Image objects.
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        result_text = f"in the style of p3r5on, {prompt[0].lower() + prompt[1:]}"  # Adjust prompt format
        logger.info(f"Generating images with prompt: {result_text}")

        # Generate images
        output = self.pipeline(
            prompt=result_text,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
        )
        logger.info("Image generation complete.")
        return output.images


if __name__ == "__main__":
    # Example usage
    try:
        # Path to config file
        CONFIG_PATH = "config.json"  # Ensure you have this JSON file with appropriate fields.
        with open(CONFIG_PATH, 'r') as file:
            config = json.load(file)

        # Initialize the Image Generator
        flux_generator = FluxImageGenerator(config, device="cuda")

        # User-provided inputs
        prompt = "A beautiful landscape with mountains and a river"
        height = 512
        width = 512
        num_images = 3

        # Generate images
        images = flux_generator.generate_images(prompt, height, width, num_images)

        # Display or save the images
        for idx, img in enumerate(images):
            img.show()  # Display the image
            img.save(f"output_image_{idx + 1}.png")  # Save the image to a file
            logger.info(f"Image {idx + 1} saved as output_image_{idx + 1}.png")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")