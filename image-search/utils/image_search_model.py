import torch
import open_clip
from PIL import Image
from typing import List, Dict, Any


class ClipModel:
    """
    A class to handle the Clip ViT-L-14 model, including initialization, 
    image preprocessing, and embeddings generation for images and text.
    """
    def __init__(self, model_name: str, pretrained: str, device: str):
        """
        Initialize the ClipModel with a given model and device.

        Args:
            model_name (str): The name of the CLIP model (e.g., "ViT-L-14").
            pretrained (str): The pretrained weights to load (e.g., "openai").
            device (str): The device to run the model on ("cuda" or "cpu").
        """
        self.device = device
        self.model, _, self.processor = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def load_image(self, file: str) -> Image.Image:
        """
        Load an image from a file path and sanitize its metadata.

        Args:
            file (str): Path to the image file.

        Returns:
            Image.Image: The loaded PIL image.
        """
        image = Image.open(file)
        allowed_keys = ["icc_profile", "XML:com.adobe.xmp"]
        for key in list(image.info.keys()):
            if key not in allowed_keys:
                image.info.pop(key)
        return image

    def get_image_embeddings(self, image_path: str) -> List[float]:
        """
        Generate normalized embeddings for an image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            List[float]: The normalized image embeddings.
        """
        image = self.load_image(image_path)
        processed_image = self.processor(image).unsqueeze(0).to(self.device)
        embeddings = self.model.encode_image(processed_image)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.tolist()[0]

    def get_text_embeddings(self, text: str) -> List[float]:
        """
        Generate normalized embeddings for a text description.

        Args:
            text (str): The text description to embed.

        Returns:
            List[float]: The normalized text embeddings.
        """
        tokenized_text = self.tokenizer(text).to(self.device)
        embeddings = self.model.encode_text(tokenized_text)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.tolist()[0]


# Example Usage
if __name__ == "__main__":
    config = {"model_name": "ViT-L-14", "pretrained": "openai", "device": "cuda"}
    clip_model = ClipModel(**config)

    image_path = "path_to_image.jpg"
    text_description = "A description of the image"

    image_embeddings = clip_model.get_image_embeddings(image_path)
    print("Image Embeddings:", image_embeddings)

    text_embeddings = clip_model.get_text_embeddings(text_description)
    print("Text Embeddings:", text_embeddings)
