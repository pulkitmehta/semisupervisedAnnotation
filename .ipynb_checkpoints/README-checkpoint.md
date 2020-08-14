# semisupervisedAnnotation
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
 
# Problem Statement
Problem: Semi supervised annotation
Deep Learning is awesome, it has given wings to computer vision and object
detection/classification. But generating enough training data still remains a challenge! it
takes countless hours for somebody to manually draw boxes around birds and people so
that they can be then used to train the model.
Can you think of a better approach, what if the model is iteratively trained and it helps you
during the annotation itself.

# Solution

## Expected Inputs:
- A Video to be Annotated
- What classes of objects to be extracted and annotated? (Array Like: Say [Car, Bus, person,...])
- Frames of Interest (FOI) means What percentage of frames to be taken into account from the video. Say there are 20 frames in the video supplied, and we say 80% to be taken. So the system will take 80% frames which will be equally and linearly spaced.

Why am I deciding to keep FOI? Well for many videos with visually less content and movements can cause a lot of similar/redudant frames resulting in increase in data redudancy and Low quality of data corpus.

- A Confidence parameter which will be threshold from 0 to 1 which will set the confidence of this annotation model to annotate a frame. This parameter will directly impact the quality of training data it will produce. Say for poor lit conditions we would want to take object which was judged as car with a confidence of 0.5 while if we keep it too low than it can affect the data quality badly as objects which are not car will also be car accordingly.

## Expected Outputs:
- The final output will be a base image and a text file of bounding box coordinates
- This will be organised in a good directory Structure as explained below:

```
output       
│
└───Class_1
│   │
│   └───Base
│   │   │   0.jpg
│   │   │   10.jpg
│   │   │   ...
│   └───BBoxes
│       │   0.txt
│       │   10.txt
│       │   ...
│   
└───Class_2
│   │
│   └───Base
│   │   │   2.jpg
│   │   │   3.jpg
│   │   │   ...
│   └───BBoxes
│       │   2.txt
│       │   3.txt
│       │   ...
│   
```
## More Info
I will Use Darknet Model with YOLO object detection which will output Bounding Boxes in array like form.
I Implemented YOLO in one of my previous Repo : https://github.com/pulkitmehta/Realtime-Object-Detection
Most of the code for Model Building has been taken from this repo.

- TensorFlow version 1.13.1
- Note the model I used is able to detect objects in coco_classes.txt. So we can use Transfer learning for our custom objects with less dataset.

## Instructions:
1. Keep Input video in 'video for annot' folder one level outside the root directory (GitHub does not allow large files upload.)
2. Save Model Weights file in one level outside root directory. Can be download from official DarkNet site: https://pjreddie.com/media/files/yolov3.weights
3. Run main.py
4. Follow Instructions as prompted in command line.
5. The Script will populate the output directory.
6. To Check Results You can run checkResults.py which will display results and we can check by drawing Bounding Boxes on Base Images.
7. More Info can be found in Demonstration Video which I have Uploaded.