import os
import json
import openai
import torch
import dotenv
import logging as logger
from transformers import pipeline
from huggingface_hub import login

logger.basicConfig(level=logger.INFO)

class EL2ImageDescription:
    """
    Generates an image description for a given Easy Language (Leichte Sprache) text, 
    used for image-search.
    """
    def __init__(self, config:dict, env_path:str="/workspace/.env"):
        """
        Initialize the EL2ImageDescription with environment configurations.
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
        
        self.model_id = config["img_desc_model_id"]
        self.img_desc_model_pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        logger.info("EL2ImageDescription initialized successfully.")


    def _get_image_description_with_llm(self, text):
        """
        Generate an English visual description for a given German EL text input using Llama-3.2 3B.

        Args:
            text (str): Input German EL text to be converted into a visual description.

        Returns:
            str: A visual description in English suitable for text-to-image search models.
        """
        logger.debug(f"Started: generating english image description for EL text: {text}")
        
        # Define the assistant's behavior via system and user prompts
        messages=[
            { 
                "role": "system", 
                "content": 
                (
                    "You are a helpful assistant designed to produce a suitable short image description in English for given paragraph in German."

                )
            },
            { 
                "role": "user", 
                "content": text
            }
        ]
        
        outputs = self.img_desc_model_pipeline(
            messages,
            max_new_tokens=256,
        )
        description = outputs[0]["generated_text"][-1]
        description = description['content']
        logger.info(f"Generated description: {description}")
        logger.info(f"Finished: generating english image description")
        return description

    def generate_img_desc_from_text(self, text):
        """
        Public method to get a English visual description based on 
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
    el_to_img_desc_assistant = EL2ImageDescription(config=config, device=device)

    # Test text input
    german_text = "Eine Person, die ein Taxi ruft."
    logger.info("Testing Llama-3.2 3B image description generation...")
    img_description = el_to_img_desc_assistant.generate_img_desc_from_text(german_text)
    
    # Print the generated English description
    print(f"Generated description: {img_description}")

    logger.info("Successfully connected to Llama-3.2 3B model and generated image description.")