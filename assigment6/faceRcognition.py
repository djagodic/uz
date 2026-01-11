import cv2
import numpy as np
import os

def dualPCA(X):
    X = np.array(X).T
    m, N = X.shape
    mu = np.mean(X, axis=1, keepdims=True)
    Xd = X - mu
    C = (1/(N-1))*Xd.T@Xd
    U, S, _ = np.linalg.svd(C)

    U = Xd @ U @ np.diag(np.sqrt(1 / ((S + 1e-15) * (N - 1))))
    return mu, C, U, S


folder_path = "./data/my_faces"
img_shape = [64, 64]
# Load images
files = sorted(os.listdir(folder_path))
X = []
for f in files:
    path = os.path.join(folder_path, f)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if not X:
            img_shape = img.shape

    X.append(img.reshape(-1))  # flatten to vector

#X = np.array(X).T  # D x N, each column is an image -> v dual PCA naredim to

# Compute PCA
mu, _, U, _ = dualPCA(X)  # D x N

def recognize_face(face_img, mu, U, threshold=3000):
    """
    face_img: grayscale image (img_shape)
    mu: mean image (D,1)
    U: PCA eigenvectors (D,N)
    threshold: L2 norm threshold to decide recognition
    """
    x = face_img.reshape(-1,1)
    y = U.T @ (x - mu)
    x_rec = U @ y + mu
    error = np.linalg.norm(x - x_rec)
    recognized = error < threshold
    return recognized, x_rec, error


# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_crop = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, img_shape)
        recognized, x_rec, error = recognize_face(face_resized, mu, U, threshold=6000)

        color = (0,255,0) if recognized else (0,0,255)
        label = "ME" if recognized else "Unknown"

        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Recognition PCA", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()