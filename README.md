# real-time_face_recognition_FaceNet

Real-time face recognition based on FaceNet

### Environment

- python==3.6.5
- tensorflow==1.12.0
- opencv-python

### Getting Started

1. Download and place pretrained FaceNet model files in "models" Folder.

   | Model name                                                   | LFW accuracy | Training dataset | Architecture                                                 |
   | ------------------------------------------------------------ | ------------ | ---------------- | ------------------------------------------------------------ |
   | [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) | 0.9965       | VGGFace2         | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |
   | [20180408-102900](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) | 0.9905       | CASIA-WebFace    | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |

2. Place user images in "Face_Image" Folder (10 images for test have been placed here).

3. Run "user_manager.py" to manage (add or delete) users.

4. Run "main.py" to start the camera for recognition.

