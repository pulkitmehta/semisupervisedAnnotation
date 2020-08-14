import numpy as np
import cv2
import os
import tensorflow as tf
import darknet
from darknet import Model, load_model_weights

'''
Run if  Tf version >=2
del tf    ## delete existing TensorFlow instance (version>=2)
import tensorflow.compat.v1 as tf  ## Enable Backward compatibility for v1 code
'''


_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1
_ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]
_MODEL_SIZE = (416, 416)



def load_class_names(file_name):
    """Returns a list of class names read from `file_name`."""
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

print("Initializing Model...", end = '')
tf.reset_default_graph()
batch_size = 1
class_names = load_class_names('./coco_classes.txt')
n_classes = len(class_names)
"""Maximum objects to detect in image and other YOLO params"""
max_output_size = 100 
iou_threshold = 0.5
confidence_threshold = 0.3

model = Model(n_classes=n_classes, input_size=_MODEL_SIZE,
                max_output_size=max_output_size,
                iou_threshold=iou_threshold,
                confidence_threshold=confidence_threshold)

inputs = tf.placeholder(tf.float32, [batch_size, 416, 416, 3])

detections = model(inputs, training=False)

model_vars = tf.global_variables(scope='yolo_v3_model')
assign_ops = load_model_weights(model_vars, '../yolov3.weights')
sess= tf.Session()
wts=sess.run(assign_ops)
print("Done!")


inpt_folder = '../videos for annot/'
vid_names = os.listdir(inpt_folder)

print("Choose Video from list:")
i = 1
for vid in vid_names:
    print(i, vid)
    i = i + 1
del i
print("Enter Choice Num:", end=' ')
vid_num = int(input())-1
inpt_path = inpt_folder+vid_names[vid_num]

print("Enter Frames of Interest ratio (0<FOI<=1):", end = ' ')
FOI = float(input())

print("Reading Video...", end = '')
cap= cv2.VideoCapture(inpt_path)
frames= []
while cap.isOpened():
    ret, frame = cap.read()
    if ret==False:
        break
    else:
        frames.append(frame)
print("Done!")

frm_idxes = np.round(np.linspace(0,len(frames)-1,int(len(frames)*FOI))).astype(int)

print("Enter Confidence Parameter (0 to 1):", end = ' ')
confidence = float(input())


i=0
for class_name in class_names:
    print(i, class_name)
    i = i + 1
del i

    
print("Enter Class Indexes to populate (Separated with ','; No Spaces!):", end = ' ')
cls_nums = input()

cls_nums = cls_nums.split(',')

    

for cls_num in cls_nums:
    
    try:
        cl = "./output/Class_"+cls_num
        cl_base = cl+"/base/"
        cl_bb = cl+"/BBoxes/"
        os.mkdir(cl)
        os.mkdir(cl_base)
        os.mkdir(cl_bb)
    except:
        pass

del cls_num

i = 0
for frm_idx in frm_idxes:
    img = frames[frm_idx]
    testimg= cv2.resize(img, _MODEL_SIZE)

    testimg=testimg.reshape(1,416,416,3)

    detection_result = sess.run(detections, feed_dict={inputs: testimg})
    detection_result = detection_result[0]
    testimg = testimg.reshape(416,416,3)
    
    
    for cls_num in cls_nums:
        cls_num = int(cls_num)

        
        
        
        for box in detection_result[cls_num]:
            if box[-1] >= confidence:
                f_path = "./output/Class_"+str(cls_num)+"/BBoxes/"+str(i)+".txt"
                
                cord = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                file = open(f_path,"w") 
                file.write(str(cord)+"\n")
                file.close()
                print(f_path, "written successfully")
                
                
                image_name = "./output/Class_"+str(cls_num)+"/base/"+str(i)+".jpg"
                cv2.imwrite(image_name, testimg) 
                print(image_name, "written succesfully of shape", testimg.shape)
                i = i + 1           