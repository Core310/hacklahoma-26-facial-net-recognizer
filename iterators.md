# Development Iterations

### Main

* This is the initial version of the face recognition system
* It uses InsightFace for detection and embeddings
* ZED SDK provides depth information to calculate distance
* Simple logic is used to decide whether to register a new person based on distance

### F2

* This version introduces several enhancements for stability
* Face tracking is added to keep track of individuals across frames
* GFPGAN is integrated to restore and enhance face images
* A registration buffer ensures that only clear faces are saved to the database
* Yaw estimation helps in selecting the best face angle for registration

### FC3

* The final iteration focuses on integration and usability
* A Flask server is included to provide a live video feed
* Streamlit dashboard connects to this feed for real time monitoring
* Images are automatically saved to the local disk during registration
* Logic for matching and creating identities is further refined
