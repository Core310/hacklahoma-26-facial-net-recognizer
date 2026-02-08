# Key Imports and Technologies

### GFPGAN

* GFPGAN is used for face restoration and enhancement
* It stands for Generative Facial Prior GAN
* The system uses it to improve the quality of face images before saving them
* Pretrained weights such as GFPGANv1.3.pth are required for operation

### MongoDB

* MongoDB serves as the primary database for the application
* It stores identity information including face embeddings
* Using a document database allows for flexible storage of person metadata
* The system connects to MongoDB running on the local machine by default

### InsightFace

* InsightFace provides the models for face detection and recognition
* It generates embeddings that are used to compare faces in the database
* The buffalo_l model is used for balanced speed and accuracy
* It is configured to run on CUDA enabled GPUs for real time performance

### ZED SDK

* The ZED SDK is required to interface with the ZED2i camera
* It provides depth sensing capabilities that help filter out distant faces
* The SDK also offers object detection features used in later iterations
