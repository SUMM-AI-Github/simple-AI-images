# Image Generation Pipeline

**Pipeline to train and generate Easy Language Images (Leichte Sprache Bilder)**


## **Table of Contents**
1. [Setup](#setup)
2. [Finetuning Image Generation model](#finetuning-image-generation-model)
3. [Inference](#inference)
4. [Gradio App](#gradio-app)
5. [Credits](#credits)


## Setup

### Requirements
- python >= 3.9
- Nvidia GPU with at least 40GB VRAM
- git

### Clone the repository
```bash
git clone https://github.com/SUMM-AI-Github/simple-AI-images.git
cd easy-language-images/image-generation
```

### Run environment

You can either use Docker container or create a virtual environment to run the application. 

1. **Using Docker (recommended)**

    Copy the .env.example file to .env and fill in your huggingface token (see: [user access tokens](https://huggingface.co/docs/hub/security-tokens)).

    Then run the following command to start the docker container:
    ```bash
    docker compose up --build -d
    ```

    When the container is running, you can connect to it with your favorite method (e.g. Visual Studio Code Dev-container extension).

2. **Using virtual environment**

    You can directly run the setup_linux.sh file (for linux), which contains all necessary setup steps from creating the virtual environment, to installing all required libraries and also torch by directly identifying the cuda version of your machine.

    **Linux**

    ```bash
    sudo apt update && sudo apt install -y python3-venv

    chmod +x setup_linux.sh
    bash setup_linux.sh
    ```
    ```bash
    source venv/bin/activate
    ```


## Finetuning Image Generation model

In this project, we finetune the FLUX-Schnell model provided by Black Forest Labs.  
We use ai-toolkit finetuning pipeline from: https://github.com/ostris/ai-toolkit.

`ai-toolkit` subdirectory is cloned from https://github.com/ostris/ai-toolkit, but for simplicity we have removed it as git-submodule. If you want the updated code version, you can clone from the link above, however the latest version might not be stable.

### Training Data

Add your training data (.jpg, .jpeg, or .png) images and .txt files inside the `./ai-toolkit/datasets/train_set` folder.  
For each image there should be a corresponding txt file with its caption. For example: alarm.png also have a alarm.txt which has the prompt for it.  
The format of prompt should be: `[trigger], an image of <image description>` (keep `[trigger]`as is and replace `<image description>` with your description of the image, e.g. `[trigger], an image of an alarm clock`).  
The fine-tuning pipeline by default replaces the [trigger] with the word `p3r5on`, which is then also used during inference. 

### Finetuning config

the main config file containing all the finetuning parameters can be found inside `./ai-toolkit/config/`. Currently there is already `train_lora_flux_schnell.yaml` which can be adapted according to your training requirements. 

Important things to consider in the config file:
- name of the training (config: name)
- lora rank: (network: linear)
- dataset path: (datasets: folder_path)
- training steps: (train: steps)

For more details about the finetuning config and setup, look into ai-toolkit/README.md file.

### Start finetuning

- Run the file to start fine-tuning process with the prepared config file:  
`python ai-toolkit/run.py ai-toolkit/config/train_lorsa_flux_schnell.yaml`
- The lora weights for different checkpoints will be stored inside `./ai-toolkit/output/`


## Inference

We provide an inference pipeline and also a Gradio app to run the FLUX model with your finetuned weights. 
It is divided into two parts:
1. Text prompt model
2. Flux inference 

### 1. Text prompt model

As the input is an Easy Language (Leichte Sprache) paragraph in German, we first need to convert it into an English prompt as most image generation models are primarily supporting English. 

Many large language models (LLMs) are likely to work for this case, as the task is designed to be model-agnostic. However, for this example, we will demonstrate using one of the **Llama-3** models provided by Meta Platforms, Inc. 
You can set the model in the `config.json` file under `prompt_model_id`.
Currently it is set to: `meta-llama/Llama-3.2-3B-Instruct` as default.

As `LLama-3.2-3B-Instruct` is a gated model, you will first need access to it by filling your details [here](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct). 
Then change the file name `.env.example` to `.env` and insert your huggingface token inside it.

The model implementation can be found inside `./utils/el_to_prompt_model_english_llama.py`. We already provide an initial suitable system prompt according to the requirements (further iteration on the prompt or trying few-shot prompt might enhance the performance). 

The model is loaded directly onto the GPU (CUDA) if available; otherwise, it defaults to the CPU.

### 2. Flux inference 

You can perform inference using the flux model with your fine-tuned LoRA weights by running the script `./utils/inference_flux.py`.  

Make sure to specify the correct path to the LoRA weights in the `config.json` file under the `lora_weights` field.  

Note that the model requires approximately 48 GB of RAM to load and run inference.  


## Gradio App

We provide a Gradio app that allows you to easily interact with the entire pipeline through an intuitive interface, as shown in the screenshot below:  

![gradio_app_screenshot](./docs/gradio_open_source_example.png)

To launch the Gradio app, run the following command:  
```bash
python app_inference_flux.py
```

In the app, users can input text in Easy Language (Leichte Sprache), generate prompts based on the input, edit the prompts if needed, and then generate images accordingly.  

The entire pipeline is defined in the `app_inference_flux.py` file, which can be customized to suit your specific needs.  


## Credits  
The project builds on several outstanding tools and models:  

- **[AI Toolkit](https://github.com/ostris/ai-toolkit):** A collection of AI tools and utilities, mostly to leverage text-to-image models.
- **[FLUX.1 [schnell]](https://huggingface.co/black-forest-labs/FLUX.1-schnell):**  a 12 billion parameter rectified flow transformer capable of generating images from text descriptions 
- **[Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct):** A 3B instruction-tuned large language model.  
