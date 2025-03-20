import cv2
img = cv2.imread('raja.jpg',cv2.IMREAD_COLOR)

# Load the Haar Cascade Classifier for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Detect faces in the image
faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

# Loop through each detected face
for (x, y, w, h) in faces:
    # Draw Rectangle around the face
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Green Rectangle with thickness 1

# Check if any face is detected
if len(faces) > 0:
    print(f"{len(faces)} Face Detected!")
else:
    print("No Face Detected!")

# Display the Final Image with Face Detection
cv2.imshow("Face Detection", img)
cv2.waitKey(0)  # Wait for key press
cv2.destroyAllWindows()  # Close all windows
