import cv2

# Initialize the tracker
tracker = cv2.legacy.TrackerMOSSE().create()

# Read the video
cap = cv2.VideoCapture('video.mp4')

# Read the first frame
ret, frame = cap.read()

# Select the bounding box of the object we want to track
bbox = cv2.selectROI("Tracking", frame, False)
tracker.init(frame, bbox)

while True:
    # Read a new frame
    ret, frame = cap.read()

    # Update tracker
    success, bbox = tracker.update(frame)

    if success:
        # Draw bounding box
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        cv2.putText(frame, "Lost", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display result
    cv2.imshow("Tracking", frame)

    # Break the loop if 'esc' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
