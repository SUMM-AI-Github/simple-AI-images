import os
import json
import openai
import torch
import dotenv
import logging as logger
from transformers import pipeline
from diffusers import AutoPipelineForText2Image, DiffusionPipeline, FluxPipeline
from safetensors.torch import load_file
from huggingface_hub import login

logger.basicConfig(level=logger.INFO)

class EL2PromptModelENG:
    """
    Generates an image prompt for a given Easy Language (Leichte Sprache) text, 
    used for image-generation.
    """
    def __init__(self, config:dict, env_path:str="/workspace/.env"):
        """
        Initialize the EL2PromptModelENG with environment configurations.
        Args:
            config: The config object(dict) containing the required fields.
        """
        # Load environment variables
        dotenv.load_dotenv(env_path)

        # Load configuration for Llama-3.2 3B model
        self.hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")
        
        if self.hugging_face_token:
            logger.info("Logging into Hugging Face...")
            # Login to Hugging Face using the provided token
            login(self.hugging_face_token)
            logger.info("Successfully logged into Hugging Face.")
        else:
            logger.error("Hugging Face token is missing from the config!")
            raise ValueError("Hugging Face token is required for Hugging Face API access.")
        
        self.model_id = config["prompt_model_id"]
        self.prompt_model_pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        logger.info("EL2PromptModelENG initialized successfully.")


    def _get_image_description_with_llm(self, text):
        """
        Generate an English visual description for a given German EL text input using fewshot Llama-3.2 3B.

        Args:
            text (str): Input German EL text to be converted into a visual description.

        Returns:
            str: A visual description in English suitable for text-to-image models.
        """
        logger.debug(f"Started: generating english image prompt for EL text: {text}")
        
        # Define the assistant's behavior via system and user prompts
        messages=[
            { 
                "role": "system", 
                "content": 
                (
                    "You are an assistant tasked with generating visual prompts in English from given German text inputs. "
                    "Your response should describe the key visual aspects of the scene in a precise, clear manner, suitable for use by a text-to-image model like SDXL or Flux. "
                    "Focus on the most visually distinct and relevant features in the given text."
                )
            },
            { 
                "role": "user", 
                "content": text
            }
        ]
        
        outputs = self.prompt_model_pipeline(
            messages,
            max_new_tokens=256,
        )
        description = outputs[0]["generated_text"][-1]
        description = description['content']
        logger.info(f"Generated prompt: {description}")
        logger.info(f"Finished: generating english image prompt")
        return description

    def generate_eng_prompt_from_text(self, text):
        """
        Public method to get a English visual description prompt based on 
        input German EL-Easy Language (Leichte Sprache) paragraph.

        Args:
            text (str): German input EL text to be converted into an English visual description.

        Returns:
            str: A visual description in English.
        """
        return self._get_image_description_with_llm(text)


# Example usage of the class
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Path to config file
    CONFIG_PATH = "config.json"  # Ensure you have this JSON file with appropriate fields.
    with open(CONFIG_PATH, 'r') as file:
        config = json.load(file)


    # Initialize the assistant
    el_to_prompt_assistant = EL2PromptModelENG(config=config, device=device)

    # Test text input
    german_text = "Eine Person, die ein Taxi ruft."
    logger.info("Testing Llama-3.2 3B prompt generation...")
    prompt = el_to_prompt_assistant.generate_eng_prompt_from_text(german_text)
    
    # Print the generated English prompt
    print(f"Generated Prompt: {prompt}")

    logger.info("Successfully connected to Llama-3.2 3B model and generated a prompt.")