# Setup Guide

### Requirements

* NVIDIA GPU with CUDA support
* ZED2i Camera
* MongoDB installed and running locally
* Python 3.12

### Links

* ZED SDK: https://www.stereolabs.com/developers/release/
* MongoDB: https://www.mongodb.com/try/download/community
* Tailscale: https://tailscale.com/download
* GFPGAN Repo: https://github.com/TencentARC/GFPGAN

### Installation

* Install dependencies: pip install opencv_python pymongo insightface streamlit gfpgan
* Install ZED wheel: pip install ./*.whl
* Download weights: wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth

### Execution

* Start main: python main.py
* Start enhanced version: python f2.py
* Start dashboard: streamlit run flask_dashboard.py

### Building from source

* Download ZED SDK from the link above
* Install ZED SDK following official instructions
* Install python dependencies using pip as shown above
* Install pyzed wheel using the wildcard command
* Download GFPGAN weights using wget if they are not present in the weights folder
* This includes GFPGANv1.3.pth and the parsing and detection models

### Tailscale usage

* Use Tailscale IP for remote access to the Flask stream
* Update STREAM URL in flask dashboard script with your Tailscale IP
* Tailscale allows secure connection between the camera server and the dashboard

### Min system spec

* CPU: Intel i7 or equivalent
* RAM: 16GB or more
* GPU: NVIDIA RTX 20 series or newer for optimal performance
* Disk: 10GB of free space

### Recommended system spec

* CPU: Intel i9 or equivalent
* RAM: 32GB or more
* GPU: NVIDIA RTX 3080 or newer for best performance
* VRAM: 8GB or more
