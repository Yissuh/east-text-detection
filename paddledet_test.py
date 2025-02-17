import cv2
import numpy as np
import time
from paddleocr import PaddleOCR

# Initialize PaddleOCR for text detection
ocr = PaddleOCR(
    lang='en',
    det_model_dir="ch_PP-OCRv4_det_infer",
    det=True,
    rec=False,
    cls=False,
    use_angle_cls=True,
    det_db_thresh=0.3,
    det_db_box_thresh=0.3,
    use_gpu=False
)

def process_frame(frame, detector):
    """Process a single frame for text detection."""
    # Detect text boxes
    result = detector.ocr(frame, rec=False)
    
    # Draw boxes if detected
    if result:
        for boxes in result:
            if boxes:
                for box in boxes:
                    if box is not None:
                        points = np.array(box).astype(np.int32)
                        
                        # Draw bounding box
                        cv2.polylines(frame, [points], True, (0, 255, 0), 2)
                        
                        # Add "Text detected" label above the box
                        text_position = (int(points[0][0]), int(points[0][1] - 10))
                        cv2.putText(frame, 
                                    "Text detected",
                                    text_position,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    2)
    
    return frame

def main():
    cap = cv2.VideoCapture(0)  # Change to 1 if using an external webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_frame_time = time.time()  # Track time for FPS calculation

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Start FPS calculation
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time  # Update previous time

            # Process the frame
            processed_frame = process_frame(frame, ocr)

            # Display FPS on screen
            cv2.putText(processed_frame, 
                        f"FPS: {fps:.2f}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 255), 
                        2)

            # Show the result
            cv2.imshow('Text Detection', processed_frame)

            # Break loop on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
