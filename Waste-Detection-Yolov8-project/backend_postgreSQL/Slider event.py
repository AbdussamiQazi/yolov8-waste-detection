import customtkinter as ctk
import matplotlib.pyplot as plt
from multiprocessing import Process
import cv2
from tkinter import filedialog
from ultralytics import YOLO
import cvzone
import math
from PIL import Image, ImageTk
from collections import Counter
from tkinter import *
from PIL import Image


import time
import psycopg2



def start():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    model = YOLO("D:\\Nanomod\\runs\\detect\\train9\\weights\\best.pt")

    classNames = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
    myColor = (0, 0, 255)

    last_added = {class_name: 0 for class_name in classNames}

    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="universe",
        host="localhost",
        port="5432"
    )


    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS waste (
            id SERIAL PRIMARY KEY,
            class_name VARCHAR(255),
            box_xmin INTEGER,
            box_ymin INTEGER,
            box_xmax INTEGER,
            box_ymax INTEGER,
            confidence FLOAT
        )
    """)

    # Commit the transaction
    conn.commit()

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read frame from webcam. Exiting ...")
            break

        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1


                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
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


                    if time.time() - last_added[currentClass] >= 7.0:
                        # Update the last added time for this class
                        last_added[currentClass] = time.time()


                        cur.execute("INSERT INTO waste (class_name, box_xmin, box_ymin, box_xmax, box_ymax,confidence) VALUES (%s, %s, %s, %s, %s,%s)",
                                    (currentClass, x1, y1, x2, y2,confidence))


                        conn.commit()



        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            visualize_data(graph_type)
            break


    conn.close()

def visualize_data(graph_type):
    # Connect to your postgres DB
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="universe",
        host="localhost",
        port="5432"
    )

    cur = conn.cursor()

    # Execute the query
    cur.execute("SELECT class_name FROM waste;")

    # Fetch the results of the query
    rows = cur.fetchall()

    # Close the cursor and the connection
    cur.close()
    conn.close()

    class_names = [row[0] for row in rows]

    class_counts = Counter(class_names)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    # Create a graph based on the specified graph type
    if graph_type == "bar":
        plt.bar(class_counts.keys(), class_counts.values(), color=colors)
    elif graph_type == "pie":
        plt.pie(class_counts.values(), labels=class_counts.keys(), colors=colors, autopct='%1.1f%%')
    elif graph_type == "histogram":
        plt.hist(class_names, bins=len(class_counts), color=colors[0])
    elif graph_type == "line":
        plt.plot(list(class_counts.keys()), list(class_counts.values()), color=colors[0])

    plt.xlabel('Class Name')
    plt.ylabel('Count')
    plt.title('Number of Detected Classes')
    plt.show()


def predict():
    classNames = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]

    # Open file dialog to select the image file
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if not filepath:
        return


    img = cv2.imread(filepath)


    img = cv2.resize(img, (1280, 720))


    model = YOLO("D:\\Nanomod\\runs\\detect\\train9\\weights\\best.pt")
    names = model.names


    results = model(img, show=True)

    # Connect to your postgres DB
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="universe",
        host="localhost",
        port="5432"
    )

    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS waste (
            id SERIAL PRIMARY KEY,
            class_name VARCHAR(255),
            box_xmin INTEGER,
            box_ymin INTEGER,
            box_xmax INTEGER,
            box_ymax INTEGER,
            confidence FLOAT
        )
    """)

    conn.commit()

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for xyxy, c, confidence in zip(boxes.xyxy, boxes.cls, boxes.conf):
            x1, y1 = (int(xyxy[0]),int(xyxy[1]))
            x2, y2 = (int(xyxy[2]),int(xyxy[3]))
            class_name = names[int(c)]
            conf = float(confidence)
            print(class_name)
            print(conf)
            print((x1,y1),(x2,y2))
            cv2.rectangle(img,(x1, y1), (x2, y2),(0,255,0))

            cur.execute("""
                INSERT INTO waste (class_name, box_xmin, box_ymin, box_xmax, box_ymax,confidence )
                VALUES (%s, %s, %s, %s, %s,%s)
            """, (class_name, x1, y1, x2, y2,conf))

        conn.commit()
    conn.close()




    return img


from tkinter import PhotoImage, Label

def create_child_window():
    child_window = ctk.CTkToplevel(root)
    child_window.geometry("800x600")
    frame = ctk.CTkFrame(child_window,width= 700, height= 500)
    frame.propagate(False)
    frame.pack()
    start_button = ctk.CTkButton(frame, text="Start", command=lambda: Process(target=start).start(),font= ("Arial",20), hover_color= "darkblue")
    start_button.pack()
    predict_button = ctk.CTkButton(frame, text="Predict Image", command=predict,font=("Arial",20),hover_color= "darkblue")
    predict_button.pack()
    def close():
        child_window.destroy()
        child_window.update()
    close_but = ctk.CTkButton(frame, text = "Close", command = close,hover_color= "darkblue", corner_radius=50)
    close_but.pack()
    graph_type = StringVar()
    radiobutton1= ctk.CTkRadioButton(frame, text="Bar Graph", variable=graph_type, value="bar", font = ("Arial",20))
    radiobutton2= ctk.CTkRadioButton(frame, text="Pie Chart", variable=graph_type, value="pie", font = ("Arial",20))
    radiobutton3= ctk.CTkRadioButton(frame, text="Histogram", variable=graph_type, value="histogram", font = ("Arial",20))
    radiobutton4= ctk.CTkRadioButton(frame, text="Line Graph", variable=graph_type, value="line", font = ("Arial",20))
    radiobutton1.pack(padx=20, pady=10)
    radiobutton2.pack(padx=20, pady=10)
    radiobutton3.pack(padx=20, pady=10)
    radiobutton4.pack(padx=20, pady=10)
    # Create a button for visualizing the data
    visualize_button = ctk.CTkButton(frame, text="Visualize Your Data", command=lambda: visualize_data(graph_type.get()),font=("Arial",20),hover_color= "darkblue")
    visualize_button.pack(padx =20, pady = 10)
    def sliding(value):
        label.configure(text = value)
        confidence_value = value
        child_window.update()
    slider = ctk.CTkSlider(frame, from_=0, to= 1, command = sliding)
    slider.pack()
    slider.set(0)
    label = ctk.CTkLabel(frame, text = slider.get(), font =("Arial",20))
    label.pack()
if __name__ == "__main__":
    root = ctk.CTk()
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("green")
    root.title("Waste Classifier")
    root.geometry("1980x1080")
    frame = ctk.CTkFrame(root, width=1400, height=900)
    frame.pack_propagate(False)  # Don't allow the widgets inside to determine the frame's width and height
    frame.pack()
    mode = "dark"

    def change():

        global mode
        if mode == "dark":
            ctk.set_appearance_mode("light")
            mode = "light"

        else:
            ctk.set_appearance_mode("dark")
            mode = "dark"


    Change = ctk.CTkButton(frame, text=".", command=change,height = 20,width =20,hover_color= "darkblue")
    Change.pack(padx = 100, pady = 20)

    bg_image = Image.open("Designer.png")
    bg_photo = ImageTk.PhotoImage(bg_image)

    # Create a canvas for the background image
    bg_canvas = Canvas(frame, width=800, height=350)
    bg_canvas.pack(fill="both", expand=True)

    bg_canvas.create_image(0, 0, image=bg_photo, anchor="nw")
    title = ctk.CTkLabel(frame, text="Waste Classifier", font=("Arial Black", 47))
    title.pack()

    tips = [
        "Tip 1: Embark on your journey towards a cleaner world by clicking on the ‘Start’ button.",
        "Tip 2: This activates our live camera detection, powered by our meticulously trained YOLO model.",
        "Tip 3:Click on ‘Data Visualise’ to delve into the contents of our PostgreSQL database",
        "Tip 4: Have an image you’d like us to analyse? Simply press ‘Predict’, and pass your image for detection.",

    ]
    for tip in tips:
        tip_label = ctk.CTkLabel(frame, text=tip, font=("Arial Black", 16))
        tip_label.pack()

    get_started_button = ctk.CTkButton(frame, text="Get Started", command=create_child_window, font = ("Arial",20),hover_color= "darkblue",border_width = 10,border_color= "white")
    get_started_button.pack()

    root.mainloop()


