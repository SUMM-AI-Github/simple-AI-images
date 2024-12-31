# Use Python 3.11-slim as the base image
FROM python:3.11-slim

# Install Git for cloning repositories
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY ./ai-toolkit/requirements.txt /tmp/ai_toolkit_requirements.txt
RUN pip install -r /tmp/ai_toolkit_requirements.txt

COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Clone the diffusers repository and install it
RUN git clone https://github.com/huggingface/diffusers.git /workspace/diffusers
RUN pip install /workspace/diffusers

RUN apt-get update && apt-get install -y libgl1
RUN apt-get update && apt-get install -y libglib2.0-0

# Set the working directory to /workspace
WORKDIR /workspace

# Copy the application script
COPY ./app_inference_flux.py ./app_inference_flux.py
COPY ./.env ./.env

# Set environment variable for Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run the application
CMD ["python", "-u", "app_inference_flux.py"]  
