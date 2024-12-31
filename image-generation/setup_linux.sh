#!/bin/bash

echo "Starting Linux project setup..."

# Step 1: Check for Python and venv
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.9 or newer and try again."
    exit 1
fi

if ! python3 -m venv --help &> /dev/null; then
    echo "Python venv module is not available. Attempting to install it..."
    if command -v apt &> /dev/null; then
        sudo apt update && sudo apt install -y python3-venv
    elif command -v yum &> /dev/null; then
        sudo yum install -y python3-venv
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y python3-venv
    else
        echo "Unsupported package manager. Please install the python3-venv module manually."
        exit 1
    fi

    if ! python3 -m venv --help &> /dev/null; then
        echo "Failed to install the venv module. Please check your system configuration."
        exit 1
    fi
fi

# Step 2: Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Step 3: Clone diffusers repository if not already cloned
if [ ! -d "./diffusers" ]; then
    echo "Cloning diffusers repository..."
    git clone https://github.com/huggingface/diffusers.git ./diffusers
else
    echo "Diffusers repository already exists. Skipping clone."
fi

# Step 3.1: Install diffusers dependencies
sudo apt update && sudo apt install -y libgl1
sudo apt update && sudo apt install -y libglib2.0-0


# Step 4: Check for CUDA version
echo "Checking for CUDA version..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9.]+')
    echo "CUDA version detected via nvcc: $CUDA_VERSION"
elif command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+')
    echo "CUDA version detected via nvidia-smi: $CUDA_VERSION"
else
    echo "CUDA version not detected. Ensure CUDA is installed and nvcc or nvidia-smi is available."
    echo "Skipping PyTorch installation. Please install PyTorch manually from https://pytorch.org/get-started/locally/"
    exit 1
fi

# Step 5: Install PyTorch based on CUDA version
echo "Installing PyTorch for CUDA version $CUDA_VERSION..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//./}

# Step 6: Verify PyTorch installation
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Step 7: Install required packages
echo "Installing required Python packages..."
pip install -r ./ai-toolkit/requirements.txt
pip install -r requirements.txt

# Step 8: Install diffusers from local path
echo "Installing diffusers package..."
sudo chown -R $USER:$USER ./diffusers
pip install -e ./diffusers

echo "Linux setup completed successfully!"
