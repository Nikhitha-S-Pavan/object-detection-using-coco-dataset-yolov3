import cv2
import numpy as np


# Reading config, weights and labels
net = cv2.dnn.readNet('yolo-coco/yolov3.weights', 'yolo-coco/yolov3.cfg')
classes = []

with open('yolo-coco/coco.names', 'r') as i:
   classes =  i.read().splitlines()

print(classes)

# read input

#img = cv2.imread('image.jpg')
cap = cv2.VideoCapture("pedestrians.mp4")

while True:
   _, img = cap.read()
   height, width, _ = img.shape

   #cv2.imshow("image", img)
   #cv2.waitKey(0)
   #cv2.destroyAllWindows()


   # connect to network
   blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), swapRB=True, crop=False)
   #for b in blob:
   #   for n, img_blob in enumerate(b):
   #      cv2.imshow(str(n), img_blob)

   net.setInput(blob)
   output_layer_names = net.getUnconnectedOutLayersNames()
   layer_outputs = net.forward(output_layer_names)

   # define list to store data
   boxes=[]
   confidences=[]
   class_ids = []


   for layer in layer_outputs:
      for detection in layer:
         scores = detection[5:]
         class_id = np.argmax(scores)
         confidence = scores[class_id]

         if confidence > 0.5:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*(height))
            x = int(center_x-w/2)
            y = int(center_y-h/2)
            boxes.append([x,y,w,h])
            class_ids.append(class_id)
            confidences.append(float(confidence))
   print(len(boxes))
   indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
   print(indexes.flatten())

   font = cv2.FONT_HERSHEY_PLAIN
   colors = np.random.uniform(0,255,size = (len(boxes),3))

   for i in indexes.flatten():
      x,y,w,h=boxes[i]
      label = classes[class_ids[i]]
      confidence = str(round(confidences[i],2))
      color = colors[i]
      cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
      cv2.putText(img,label + " " + confidence, (x,y+20), font, 2 ,colors[i], 2)

   cv2.imshow("image", img)
   key = cv2.waitKey(1)
   if key ==27:
      break
cap.release()
cv2.destroyAllWindows()
