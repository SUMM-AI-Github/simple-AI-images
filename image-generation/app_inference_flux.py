import gradio as gr
import logging
import os
import json
import re
import time
import random
import base64
import torch
from io import BytesIO
from PIL import Image
from datetime import datetime
from utils.inference_flux import FluxImageGenerator
from utils.el_to_prompt_model_english_llama import EL2PromptModelENG

# Create the logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename='logs/image_generation.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path (str): Path to the config JSON file.
    
    Returns:
        dict: Configuration data.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r') as file:
        config = json.load(file)
    logger.info("Config file loaded successfully")
    return config

def _check_device() -> str:
    """
    Checks for available computation device (CUDA or CPU).
    
    Returns:
        str: 'cuda' if GPU is available, otherwise 'cpu'.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    return device


config = _load_config("./config.json")
device = _check_device()

# Create the output directory if it doesn't exist
output_base = config["output_images_folder"]
if not os.path.exists(output_base):
    os.makedirs(output_base)

# global variables
RESOLUTION_CHOICES = config["resolution_choices"]


logging.info("Started: Loading required models...")
# Initialize the FluxImageGenerator
image_generator = FluxImageGenerator(config=config, device=device)
# Initialize the EL to prompt generator 
el_to_prompt_assistant = EL2PromptModelENG(config=config)
logging.info("Completed: Loading required models.")

logging.info("Application started successfully!")

def get_easy_language_text_id(output_dir="output"):
    """Get the next available ID for the easy language text inside the output directory."""
    
    pattern = re.compile(r"^EL_(\d+)_")
    max_index = 0
    for folder_name in os.listdir(output_dir):
        match = pattern.match(folder_name)
        if match:
            index = int(match.group(1))
            max_index = max(max_index, index)
    next_index = max_index + 1
    main_text_id = f"EL_{next_index}"
    return main_text_id


def generate_images(user_prompt, resolution="1024*1024"):
    """Generate images using the Flux model deployed in Azure."""
    prompt = f"in the style of p3r5on, {user_prompt[0].lower() + user_prompt[1:]}"
    logging.info(f"Updated Flux prompt: {prompt}")
    # Map resolution to width and height
    resolutions = {
        "1024*1024": (1024, 1024),
        "512*512": (512, 512),
        "1240*1744 (A6)": (1240, 1744)
    }
    height, width = resolutions[resolution]
    
    # Generate images with a prompt
    flux_images = image_generator.generate_images(
        prompt=prompt,
        num_images_per_prompt=3,
        num_inference_steps=5,
        height=height,
        width=width,
        seed=None
    )

    return flux_images


def get_ai_generated_prompt(input_text):
    """Generate a English prompt from EL using the EL2PromptModel model."""
    logging.info(f"Generating prompt for input text: {input_text}")
    generated_prompt = el_to_prompt_assistant.generate_eng_prompt_from_text(input_text)
    logging.info(f"Generated prompt: {generated_prompt}")
    return generated_prompt


def save_results(main_text_id, original_prompt, user_edited_prompt, paragraph_text, images, 
                 resolution, user_name='Anonymous'):
    """Saves the results and associated data to a JSON file and stores images."""
    
    logging.info(f"Started: saving results for Username: {user_name}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"{output_base}/{main_text_id}_{user_name}_{timestamp}"
    os.makedirs(result_dir)
    result_filename = f"{result_dir}/request_details.json"
    
    data = {
        "main_text_id": main_text_id,
        "timestamp": timestamp,
        "user_name": user_name,
        "input_paragraph": paragraph_text,
        "generated_prompt": original_prompt,
        "user_edited_prompt": user_edited_prompt,
        "resolution": resolution,
        "images": []
    }
    
    for idx, img in enumerate(images):
        if isinstance(img, Image.Image):  # Ensure it's a PIL image
            img_path = f"{result_dir}/image_{idx + 1}.png"
            img.save(img_path)
            data['images'].append({
                "image_id": idx + 1,
                "file_path": img_path,
            })
        else:
            logging.warning(f"Skipping image {idx + 1}: Invalid image object")
    
    with open(result_filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    logging.info(f"Completed saving results to {result_filename}")


# Gradio Main Interface
with gr.Blocks() as interface:
    # Input section
    with gr.Row():
        with gr.Column(scale=1):
            user_name_textbox = gr.Textbox(label="Enter your name", placeholder="Name")
        with gr.Column(scale=6):
            input_textbox = gr.Textbox(
                lines=6,
                label="Enter Easy Language (Leichte Sprache) text",
                elem_id="input_paragraph",
                placeholder="Enter a text paragraph..."
            )
        get_prompt_button = gr.Button("Get Relevant Prompt")

    # Generated prompt section
    with gr.Row() as prompt_generation_section:
        generated_prompt_textbox = gr.Textbox(
            lines=3, label="AI Generated Prompt (German)", interactive=False, elem_id="generated_prompt"
        )
        user_edited_prompt_textbox = gr.Textbox(
            lines=3, label="User Editable Prompt (German)", interactive=True, elem_id="user_edited_prompt"
        )
        resolution_dropdown = gr.Dropdown(
            label="Select Resolution",
            choices=RESOLUTION_CHOICES,
            value="1024*1024"
        )
        generate_images_button = gr.Button("Generate Images")
        
    # Display images after generation
    with gr.Row(visible=True) as image_display_section:
        generated_images = [
            gr.Image(interactive=False, type="pil", visible=True, label=f"Generated Image {i + 1}") 
            for i in range(3)
        ]
    
    # Submission section
    with gr.Row(visible=True) as submit_section:
        submit_button = gr.Button("Save Images and Prompts")
        submission_status = gr.Markdown(value="", visible=True)
    models = gr.State()


    def on_get_ai_prompt_clicked(text):
        """Handles the 'Get Relevant Prompt' button click."""
        generated_prompt = get_ai_generated_prompt(text)
        logging.info(f"Generated prompt: {generated_prompt}")
        return generated_prompt, generated_prompt, gr.update(visible=True)

    def on_generate_images_clicked(edited_prompt, resolution):
        """Handles the 'Generate Images' button click."""
        images = generate_images(edited_prompt, resolution)
        image_updates = [gr.update(value=img, visible=True) for img in images]
        return image_updates + [gr.update(visible=True)]

    def submit_form(user_name, paragraph_text, generated_prompt, edited_prompt, resolution, *args):
        """Handles the 'Submit' button click."""
        
        images = args
        main_text_id = get_easy_language_text_id(output_base)
        
        if not user_name:
            return gr.update(value="Please enter your name!", visible=True)
        if not paragraph_text:
            return gr.update(value="Please enter the easy language text!", visible=True)
        if not generated_prompt and edited_prompt:
            return gr.update(value="Please generate the prompt (User_edited_prompt field cannot be empty)!", visible=True)
        
        save_results(
            main_text_id=main_text_id,
            original_prompt=generated_prompt,
            user_edited_prompt=edited_prompt,
            paragraph_text=paragraph_text,
            images=images,
            resolution=resolution,
            user_name=user_name,
        )
        return gr.update(value="Submission Successful!", visible=True)

    # Button actions
    get_prompt_button.click(
        on_get_ai_prompt_clicked,
        inputs=input_textbox,
        outputs=[generated_prompt_textbox, user_edited_prompt_textbox, prompt_generation_section]
    )
    generate_images_button.click(
        on_generate_images_clicked,
        inputs=[user_edited_prompt_textbox, resolution_dropdown],
        outputs=generated_images + [image_display_section]
    )
    submit_button.click(
        submit_form,
        inputs=[user_name_textbox, input_textbox, generated_prompt_textbox, user_edited_prompt_textbox, resolution_dropdown] + generated_images,
        outputs=submission_status
    )

# Launch the app
interface.launch(inline=True, debug=True, share=True)