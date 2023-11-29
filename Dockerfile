# Start from the PyTorch image
FROM nvcr.io/nvidia/pytorch:21.08-py3
COPY . /culturemovielearn
# Set the working directory to /yolo
WORKDIR /culturemovielearn

# Update the package list and install required packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-glx

# Install the required Python packages
RUN pip install \
    matplotlib>=3.2.2 \
    "numpy>=1.18.5,<1.24.0" \
    opencv-python==4.5.5.64 \
    Pillow>=7.1.2 \
    PyYAML>=5.3.1 \
    requests>=2.23.0 \
    scipy>=1.4.1 \
    "torch>=1.7.0,!=1.12.0" \
    "torchvision>=0.8.1,!=0.13.0" \
    tqdm>=4.41.0 \
    tensorboard>=2.4.1 \
    pandas \
    moviepy \
    argparse \
    pytorchvideo
