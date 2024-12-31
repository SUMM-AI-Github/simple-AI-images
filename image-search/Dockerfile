# Use Python 3.11-slim as the base image
FROM python:3.11-slim

# Copy requirements and install Python dependencies
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Set the working directory to /workspace
WORKDIR /workspace

# Copy the application script
COPY ./app.py ./app.py

# Set environment variable for Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run the application
CMD ["python", "app.py"]