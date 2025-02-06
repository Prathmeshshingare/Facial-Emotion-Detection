import tkinter as tk
from tkinter import messagebox
import getpass
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model


username_entry = None
password_entry = None
root = None


def authenticate():
    global username_entry, password_entry, root
   
    username = username_entry.get()
    password = password_entry.get()
   
    if username == "mini" and password == "616263":
        messagebox.showinfo("Authentication", "Authentication successful. Starting facial emotion analysis...")
        root.destroy()  # Close the login window
        start_emotion_analysis()  # Start the facial emotion analysis
    else:
        messagebox.showerror("Authentication Error", "Incorrect username or password. Authentication failed.")


def start_emotion_analysis():
    # Load the trained model
    model = load_model("output.h5")
   
    # Load the Haar cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
   
    # Start video capture from default camera
    cap = cv2.VideoCapture(0)
   
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
       
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
       
        # Loop through each detected face
        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) which is the face area
            roi_gray = gray_frame[y:y+h, x:x+w]
           
            # Resize the ROI to match the input shape of the model
            roi_gray_resized = cv2.resize(roi_gray, (48, 48))
           
            # Expand dimensions to make it a batch of one image
            roi_gray_resized = np.expand_dims(roi_gray_resized, axis=-1)
           
            # Normalize pixel values to range [0, 1]
            roi_gray_resized = roi_gray_resized / 255.0
           
            # Expand dimensions to make it a batch of one image
            input_image = np.expand_dims(roi_gray_resized, axis=0)
           
            # Make predictions on the input image
            predictions = model.predict(input_image)
           
            # Find the index corresponding to the predicted emotion
            max_index = np.argmax(predictions[0])
           
            # Define the list of emotions
            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
           
            # Get the predicted emotion
            predicted_emotion = emotions[max_index]
           
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)
           
            # Display the predicted emotion above the rectangle
            cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
       
        # Display the frame with annotations
        cv2.imshow('Facial Emotion Analysis', frame)
       
        # Check if 'q' key is pressed to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


def go_back():
    global root
    root.destroy()  # Close the login window
    main_menu()  # Return to the main menu


def main_menu():
    global username_entry, password_entry, root
   
    # Create the main window for login
    root = tk.Tk()
    root.title("Login")
   
    # Set window size to full screen
    root.attributes('-fullscreen', True)
   
    # Create and place login label
    login_label = tk.Label(root, text="Login Here", font=("Helvetica", 24))
    login_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
   
    # Create and place username label and entry
    username_label = tk.Label(root, text="Username:")
    username_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.E)
   
    username_entry = tk.Entry(root)
    username_entry.grid(row=1, column=1, padx=5, pady=5)
   
    # Create and place password label and entry
    password_label = tk.Label(root, text="Password:")
    password_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.E)
   
    password_entry = tk.Entry(root, show="*")
    password_entry.grid(row=2, column=1, padx=5, pady=5)
   
    # Create and place login button
    login_button = tk.Button(root, text="Login", command=authenticate)
    login_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
   
    # Create and place back button
    back_button = tk.Button(root, text="Back", command=go_back)
    back_button.grid(row=4, column=0, columnspan=2, padx=5, pady=5)
   
    # Run the main event loop
    root.mainloop()


# Start with the main menu
main_menu()



