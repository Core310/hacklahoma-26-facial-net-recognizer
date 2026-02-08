# Setup Guide

## Build & Run from Source

### Prerequisites
* **NVIDIA** GPU Required for ZED SDK, InsightFace, and GFPGAN inference.
* **CUDA** Must have CUDA 11.x or 12.x installed matching your ZED SDK version.
* **ZED** Camera ZED 2 or ZED 2i connected via USB 3.0.
* **ZED** SDK Install the latest ZED SDK for your OS.
* **MongoDB** Must be installed and running locally on port 27017.
* **Python** 3.10 or 3.12 recommended.

### 1. Installation

**Clone the repository:**
```bash
git clone [https://github.com/yourusername/fnr-facial-net-recognizer.git](https://github.com/yourusername/fnr-facial-net-recognizer.git)
cd fnr-facial-net-recognizer
Set up Python Environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install Dependencies:

Bash
# Install core libraries
pip install opencv-python numpy pymongo flask streamlit

# Install AI engines (GPU versions)
pip install insightface onnxruntime-gpu
pip install gfpgan basicsr

# Install ZED Python API (if not installed by SDK)
# Navigate to: "C:\Program Files (x86)\ZED SDK\python" and run:
python get_python_api.py
2. Download Model Weights
The system requires the GFPGAN pre-trained model for super resolution.

Bash
# Download GFPGANv1.3.pth to the root directory
wget [https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth)
(On Windows, you can manually download this file from the link above and place it in the project folder)

3. Start Database
Ensure your MongoDB instance is running. If you have Docker installed, this is the easiest way:

Bash
docker run -d -p 27017:27017 --name face-db mongo:latest
4. Run the System
You will need two separate terminal windows.

Terminal 1: The Backend (Camera & AI Engine) This script captures video, runs recognition, and hosts the video stream server.

Bash
python main.py
# Wait until you see: "System Running!"
# Note the IP address displayed: e.g., [http://192.168.1.50:5001/video_feed](http://192.168.1.50:5001/video_feed)
Terminal 2: The Dashboard (Frontend) Important: Open dashboard.py and ensure STREAM_URL matches the IP from Terminal 1.

Bash
streamlit run dashboard.py
Open your browser to the URL shown (usually http://localhost:8501).

Tailscale Usage
Use Tailscale IP for remote access to the Flask stream.

Update STREAM_URL in the dashboard script with your Tailscale IP (e.g., http://100.x.x.x:5001/video_feed).

Tailscale allows secure connection between the camera server and the dashboard without exposing ports to the public internet.

System Specifications
Minimum

CPU Intel i7 or equivalent

RAM 16GB or more

GPU NVIDIA RTX 20 series or newer

Disk 10GB free space

Recommended

CPU Intel i9 or equivalent

RAM 32GB or more

GPU NVIDIA RTX 3080 or newer for best performance

VRAM 8GB or more