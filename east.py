import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

# Load pre-trained EAST text detector model
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define the output layer names for the EAST model
layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

def decode_predictions(scores, geometry, conf_threshold=0.8):
    rects = []
    confidences = []
    angles = []  # Store rotation angles
    
    height, width = scores.shape[2:4]

    for y in range(height):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]  # top
        xData1 = geometry[0, 1, y]  # right
        xData2 = geometry[0, 2, y]  # bottom
        xData3 = geometry[0, 3, y]  # left
        anglesData = geometry[0, 4, y]

        for x in range(width):
            score = scoresData[x]
            if score < conf_threshold:
                continue

            offsetX, offsetY = x * 4.0, y * 4.0

            # Extract angle and compute trigonometric functions
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # Get dimensions of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # Calculate rotated bounding box corners
            x1 = offsetX + (cos * xData1[x]) + (sin * xData2[x])
            y1 = offsetY - (sin * xData1[x]) + (cos * xData2[x])

            x2 = x1 - w * cos
            y2 = y1 - w * sin

            x3 = x2 - h * sin
            y3 = y2 + h * cos

            x4 = x1 - h * sin
            y4 = y1 + h * cos

            # Store corners and confidence
            corners = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)
            rects.append(corners)
            confidences.append(float(score))
            angles.append(angle)

    return rects, confidences, angles

def draw_rotated_box(img, box, color=(0, 255, 0), thickness=2):
    # Convert box points to integers
    box = np.int0(box)
    # Draw the rotated rectangle
    cv2.drawContours(img, [box], 0, color, thickness)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    H, W = frame.shape[:2]

    # Resize frame while maintaining aspect ratio
    newW, newH = (320, 320)
    rW, rH = W / float(newW), H / float(newH)
    resized = cv2.resize(frame, (newW, newH))

    # Prepare image for the network
    blob = cv2.dnn.blobFromImage(resized, 1.0, (newW, newH),
                                (123.68, 116.78, 103.94), swapRB=True, crop=False)

    net.setInput(blob)
    scores, geometry = net.forward(layer_names)

    # Decode predictions
    rects, confidences, angles = decode_predictions(scores, geometry)

    # Apply NMS to remove overlapping boxes
    if len(confidences) > 0:
        confidences = np.array(confidences)
        # Get indices of boxes to keep after NMS
        indices = cv2.dnn.NMSBoxesRotated(
            [cv2.minAreaRect(box) for box in rects],
            confidences,
            score_threshold=0.5,
            nms_threshold=0.4
        )

        # Draw retained boxes
        for i in indices:
            i = i[0] if isinstance(i, np.ndarray) else i
            # Scale corners back to original image size
            corners = rects[i] * np.array([rW, rH])
            # Draw the rotated box
            draw_rotated_box(orig, corners)
            
            # Calculate center point for text
            center_x = int(np.mean(corners[:, 0]))
            center_y = int(np.mean(corners[:, 1]))
            
            # Display confidence score
            text = f"Text {confidences[i]:.2f}"
            cv2.putText(orig, text, (center_x, center_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("EAST Text Detection", orig)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()