from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import time
import cv2

def decode_predictions(scores, geometry, min_confidence=1):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)

def main():
    # Configuration
    width = 320
    height = 320
    min_confidence = 0.6
    model_path = "frozen_east_text_detection.pb"

    # Initialize dimensions
    W, H = None, None
    rW, rH = None, None

    # Define EAST detector output layers
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    # Load the EAST text detector
    print("[INFO] loading EAST text detector...")
    try:
        net = cv2.dnn.readNet(model_path)
    except cv2.error:
        print(f"Error: Could not load the EAST model from {model_path}")
        print("Make sure the model file exists in the current directory")
        return

    # Initialize video stream
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)  # Give camera more time to warm up

    # Initialize frame counter and start time
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Grab frame and check if it's valid
        frame = vs.read()
        if frame is None:
            print("Error: Could not read frame from camera")
            break
            
        # Increment frame counter
        frame_count += 1
        
        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
        # Resize frame
        frame = imutils.resize(frame, width=1000)
        orig = frame.copy()

        if W is None or H is None:
            (H, W) = frame.shape[:2]
            rW = W / float(width)
            rH = H / float(height)

        frame = cv2.resize(frame, (width, height))

        blob = cv2.dnn.blobFromImage(frame, 1.0, (width, height),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)

        (rects, confidences) = decode_predictions(scores, geometry, min_confidence)
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        # Draw FPS on frame
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(orig, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)

        for (startX, startY, endX, endY) in boxes:
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

        cv2.imshow("Text Detection", orig)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    end_time = time.time()
    total_time = end_time - start_time
    final_fps = frame_count / total_time
    print(f"[INFO] elapsed time: {total_time:.2f} seconds")
    print(f"[INFO] approx. FPS: {final_fps:.2f}")

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()