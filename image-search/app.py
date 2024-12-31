import os
import io
import json
import uuid
import torch
import openai
import logging
import open_clip
import gradio as gr
import qdrant_client
from PIL import Image, ImageFile
from qdrant_client.models import PointStruct
from utils.el_to_img_desc_llama import EL2ImageDescription
from utils.qdrant_manager import QdrantEngineManager
from utils.image_search_model import ClipModel

# Create the logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename='logs/image_search.log',

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
ImageFile.LOAD_TRUNCATED_IMAGES = True
collection_name = config["collection_name"]
image_dir = config["image_directory"]


logging.info("Started: Loading required models...")

# Initialize the EL to image description generator 
el_to_img_desc_assistant = EL2ImageDescription(config=config)

# Initialize Qdrant Engine
qdrant_manager = QdrantEngineManager(host=config["qdrant_host"], port=config["qdrant_port"])

# Initialize the CLIP model
clip_model = ClipModel(model_name=config["clip_model_name"], pretrained="openai", device=device)

logging.info("Completed: Loading required models.")



def prepare_image_embeddings():
    """
    Prepare embeddings for images in the workspace and insert them into the Qdrant DB.
    """
    logging.info("Started: Preparing image embeddings and inserting them into Qdrant DB.")

    # Ensure the collection exists
    qdrant_manager.create_collection_if_not_exists(collection_name, vector_size=768, distance="COSINE")

    # embed all images in the /workspace/images directory
    for image_filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, image_filename)
        image_id = os.path.splitext(image_filename)[0]
        image_id = str(uuid.uuid4())

        # check and skip if already exists in DB
        if qdrant_manager.retrieve(collection_name, [image_id]):
            print(f"Image {image_id} already exists in DB. Skipping.")
            continue

        # Generate embeddings from clip. (images not yet in DB)
        image_embeddings = clip_model.get_image_embeddings(file_path)
        qdrant_manager.upsert_data(collection_name, [
            {"id": image_id, "vector": {"image_embeddings": image_embeddings}, "payload": {"image_filename": image_filename}}
        ])
        
        # Upsert iamge embeddings into Qdrant
        qdrant_manager.upsert_data(collection_name, [
            PointStruct(
                id=image_id,
                vector={"image_embeddings": image_embeddings},
                payload={"image_filename": image_filename},
            )
        ])

    logging.info("Completed: Preparing image embeddings and inserting them into Qdrant DB.")


def get_images(text, num_elements=3):
    if text.strip() == "":
        visible_images = [
            gr.Image(
                visible=True,
                type="pil",
                show_label=False,
                label=f"Image #{ i + 1 }",
            )
            for i in range(0, num_elements)
        ]
        invisible_images = [
            gr.Image(
                visible=False,
                type="pil",
                show_label=False,
                label=f"Image #{ i + 1 }",
            )
            for i in range(num_elements, 10)
        ]
        return visible_images + invisible_images

    try:
        # Get image description from Easy Language (Leichte Sprache) text
        image_description = el_to_img_desc_assistant.generate_img_desc_from_text(text)
    except:
        visible_images = [
            gr.Image(
                visible=True,
                type="pil",
                show_download_button=True,
                show_label=False,
                label=f"Image #{ i + 1 }",
            )
            for i in range(0, num_elements)
        ]
        invisible_images = [
            gr.Image(
                visible=False,
                type="pil",
                show_download_button=True,
                show_label=False,
                label=f"Image #{ i + 1 }",
            )
            for i in range(num_elements, 10)
        ]
        return visible_images + invisible_images

    print(f"Text:\n{text}\n\nImage-Description:\n{image_description}\n\n")

    # Obtain query embeddings
    logging.info("Generating query embeddings for the image description.")
    query_embeddings = clip_model.get_text_embeddings(image_description)

    # Search in Qdrant
    logging.info("Searching for images in Qdrant.")
    query_results = qdrant_manager.search(
        collection_name=collection_name,
        query_vector=("image_embeddings", query_embeddings),
        limit=10,
    )
    
    # Gather images
    logging.info("Gathering top-k images for display.")
    image_filename_to_score_dict = {}
    for query_result in query_results:
        image_filename = getattr(query_result, "payload")["image_filename"]
        score = getattr(query_result, "score")
        image_filename_to_score_dict[image_filename] = float(score)
    predicted_image_filename_list = list(
        dict(
            sorted(
                image_filename_to_score_dict.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ).keys()
    )[:10]

    images = []
    for predicted_image_filename in predicted_image_filename_list:
        images.append(
            clip_model.load_image(os.path.join("/workspace/images", predicted_image_filename))
        )
    
    visible_images = [
        gr.Image(
            visible=True,
            type="pil",
            show_download_button=True,
            show_label=False,
            label=f"Image #{ i + 1 }",
            value=images[i],
        )
        for i in range(0, num_elements)
    ]
    invisible_images = [
        gr.Image(
            visible=False,
            type="pil",
            show_download_button=True,
            show_label=False,
            label=f"Image #{ i + 1 }",
            value=images[i],
        )
        for i in range(num_elements, 10)
    ]
    return visible_images + invisible_images


# gradio image block
def create_image(label, show_download_button=True):
    return gr.Image(
        type="pil",
        show_download_button=show_download_button,
        show_label=False,
        label=label,
    )


# Gradio Main Interface
with gr.Blocks() as demo:

    # input
    with gr.Row():
        # two columns
        with gr.Column(scale=3):
            textbox = gr.Textbox(lines=8, label="Fügen Sie hier Ihren Leichte Sprache Text ein:")

    with gr.Row():
        with gr.Column(scale=1):
            button = gr.Button(value="Bilder finden")
        
    images = []
    with gr.Row():
        for i in range(1, 4):
            with gr.Column(scale=1):
                images.append(create_image(f"Image #{i}"))

    button.click(fn=get_images, inputs=[textbox], outputs=images)


# Launch the app
if __name__ == "__main__":
    prepare_image_embeddings()
    demo.launch(share=True)


