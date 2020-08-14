import os
import cv2

while True:
    print("Enter Class Num: ")
    cl = input()
    path = './output/Class_'+cl+'/base/'
    path_box = './output/Class_'+cl+'/BBoxes/'
    print(path)
    print(os.listdir(path))
    print("Enter Image Index Number:")
    file_idx = input()
    path = path + file_idx + ".jpg"
    path_box = path_box + file_idx + ".txt"


    print("Reading Bounding Box File...")
    boxfile = open(path_box,"r+")  
  

    coordstr = boxfile.read()
    boxfile.close()

    cords = coordstr[1:-2].split(',')
    print("Box Coordinates:  ",cords)
    base = cv2.imread(path)

    x1=int(cords[0])
    y1=int(cords[1])
    x2=int(cords[2])
    y2=int(cords[3])



    thickness= max(base.shape[0],base.shape[1])//300
    base=cv2.rectangle(base,(x1,y1),(x2,y2),(0,0,255),thickness=thickness)


    cv2.imshow('image',base)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
        
    
cv2.destroyAllWindows()