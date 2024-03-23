import customtkinter as ctk
from PIL import Image, ImageTk
from multiprocessing import Process
import cv2
from tkinter import filedialog
from ultralytics import YOLO
import cvzone
import math

# Your code
def start():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    model = YOLO("D:\\Nanomod\\runs\\detect\\train9\\weights\\best.pt")

    classNames = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
    myColor = (0, 0, 255)
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read frame from webcam. Exiting ...")
            break

        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                print(currentClass)
                if conf > 0.65:  # Changed this line
                    if currentClass == 'BIODEGRADABLE':
                        myColor = (0, 255, 0)
                    elif currentClass == 'CARDBOARD':
                        myColor = (0, 0, 255)
                    elif currentClass == 'GLASS':
                        myColor = (255, 0, 0)
                    elif currentClass == 'METAL':
                        myColor = (255, 255, 0)
                    elif currentClass == 'PAPER':
                        myColor = (0, 255, 255)
                    elif currentClass == 'PLASTIC':
                        myColor = (255, 0, 255)

                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                     (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                       colorT=(0, 0, 0), colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 1)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def predict():
    # Open file dialog to select the image file
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if not filepath:
        return

    # Load the image
    img = cv2.imread(filepath)

    # Load YOLOv8
    model = YOLO("D:\\Nanomod\\runs\\detect\\train9\weights\\best.pt")

    # Predict the classes
    results = model(img,show = True)

    # The results variable now contains the predicted classes
    print(results)

if __name__ == "__main__":
    # Create the main window
    root = ctk.CTk()
    root.title("Waste Classifier")
    root.geometry("800x600")  # Set the size of the window

    # Add a fancy title
    title = ctk.CTkLabel(root, text="Waste Classifier", font=("Helvetica", 32))
    title.pack()

    # Add an image
    image = Image.open("dustbin.jpg")
    photo = ImageTk.PhotoImage(image)
    label = ctk.CTkLabel(root, image=photo)
    label.image = photo
    label.pack()

    # Add a button
    start_button = ctk.CTkButton(root, text="Start", command=lambda: Process(target=start).start())
    start_button.pack()

    # Add a button for predicting using an image
    predict_button = ctk.CTkButton(root, text="Predict Image", command=predict)
    predict_button.pack()

    root.mainloop()
