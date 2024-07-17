import cv2
import numpy as np

# Load YOLO pre-trained model
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = yolo_net.getLayerNames()
output_layers = yolo_net.getUnconnectedOutLayersNames()

# Load the image
image = cv2.imread("parking lot.jpg")

# Resize the image to YOLO network input size
height, width, channels = image.shape
blob = cv2.dnn.blobFromImage(image, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)

# Set the input to the YOLO network
yolo_net.setInput(blob)

# Forward pass to get the detections
outs = yolo_net.forward(output_layers)

# Process the detections
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Confidence threshold
            # YOLO returns the center (x, y) coordinates and the width and height of the bounding box
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Calculate the top-left corner of the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-max suppression to remove overlapping boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

# Count the number of empty parking spaces
num_empty_slots = 0
for i in range(len(boxes)):
    if i in indexes:
        class_id = class_ids[i]
        if classes[class_id] == 'car':  # Change 'car' to the class name for empty parking space if needed
            num_empty_slots += 1

# Calculate the total number of parking slots
total_slots = len(boxes)

# Calculate the number of occupied parking slots
num_occupied_slots = total_slots - num_empty_slots

# Draw the bounding boxes and labels
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y + 30), font, 1, color, 2)

# Display the number of parking slots left on the image
cv2.putText(image, f"Occupied Slots: {num_empty_slots}", (10, 30), font, 1.5, (0, 255, 0), 2)
cv2.putText(image, f"Empty Slots: {num_occupied_slots}", (10, 60), font, 1.5, (0, 0, 255), 2)

# Save the output image
cv2.imwrite("output2.jpg", image)
# Display the result
cv2.imshow("output2.jpg",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
