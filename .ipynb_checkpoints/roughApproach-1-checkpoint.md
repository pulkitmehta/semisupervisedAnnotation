# Problem Statement
Problem: Semi supervised annotation
Deep Learning is awesome, it has given wings to computer vision and object
detection/classification. But generating enough training data still remains a challenge! it
takes countless hours for somebody to manually draw boxes around birds and people so
that they can be then used to train the model.
Can you think of a better approach, what if the model is iteratively trained and it helps you
during the annotation itself.

# Solution
Note: This is my rough approach I have decided. The project is yet to be made.

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
│   │   │   class_1_1.png
│   │   │   class_1_2.png
│   │   │   ...
│   └───BBoxes
│       │   class_1_1.txt
│       │   class_1_2.txt
│       │   ...
│   
└───Class_2
│   │
│   └───Base
│   │   │   class_2_1.png
│   │   │   class_2_2.png
│   │   │   ...
│   └───BBoxes
│       │   class_2_1.txt
│       │   class_2_2.txt
│       │   ...
│   
```

I will Use Darknet Model with YOLO object detection which will output Bounding Boxes in array like form.
Research paper will be linked later!

### Pseudo Code:
initialize dir structure

for frame in frames:

    for bb in bbs:
        
        if class_num in class_nums: //if detected box is in choice user wants.
            
            save respected outputs
                
        else:
            Dont care
            
            