import cv2
import numpy as np
import os

# Parameters
save_dir = "./data/my_faces"
os.makedirs(save_dir, exist_ok=True)
num_images = 10
img_shape = (64, 64)  # resize to common resolution

cap = cv2.VideoCapture(0)

print("Press SPACE to capture an image of your face.")
print("Press ESC to finish.")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Capture", gray)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        face_img = cv2.resize(gray, img_shape)
        cv2.imwrite(os.path.join(save_dir, f"face_{count}.png"), face_img)
        count += 1
        print(f"Captured image {count}")
        if count >= num_images:
            break

cap.release()
cv2.destroyAllWindows()
