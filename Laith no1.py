import os
import subprocess
import threading
import sys
import webbrowser
from multiprocessing import Process
from tkinter import messagebox, filedialog
import customtkinter as ctk
import cv2
import dlib
import mediapipe as mp
import numpy as np
import pyautogui
import tensorflow as tf
from PIL import Image, ImageDraw
from PyQt5.QtCore import Qt, QTimer, QPoint, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout

# Load dlib face recognition models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Shape predictor
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")  # Recognition model

# Initialize MediaPipe Face Mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmarks (MediaPipe's 468+10-point model)
RIGHT_EYE_PUPIL = [468, 469, 470, 471, 472]
LEFT_EYE_PUPIL = [473, 474, 475, 476, 477]

# Additional landmarks
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_EYE = [33, 160, 158, 133, 153, 144]

# Cursor speed multiplier
speed_multiplier = 1.0
screen_width, screen_height = pyautogui.size()

ear_threshold = 0.25  # Threshold for detecting eye closure


def find_click2speak_path():
    """
    Searches for the Click2Speak executable in common installation directories.
    Returns the path if found, otherwise returns None.
    """
    # Common installation paths for Click2Speak
    possible_paths = [
        r"C:\Program Files\Click2Speak\Click2Speak.Client.exe",  # Windows 64-bit
        r"C:\Program Files (x86)\Click2Speak\Click2Speak.Client.exe",  # Windows 32-bit
        "/Applications/Click2Speak.app/Contents/MacOS/Click2Speak",  # macOS
        "/usr/local/bin/click2speak",  # Linux
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None  # Click2Speak not found


# Bubble overlay class
class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(0, 0, screen_width, screen_height)
        self.bubble_pos = QPoint(screen_width // 2, screen_height // 2)
        self.bubble_radius = 40  # Bubble radius
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_bubble_position)
        self.timer.start(10)  # Update every 10ms for smooth movement

    def update_bubble_position(self):
        """
        Update the bubble position to follow the cursor.
        """
        x, y = pyautogui.position()
        self.bubble_pos.setX(x)
        self.bubble_pos.setY(y)
        self.update()

    def paintEvent(self, event):
        """
        Draw the bubble as the cursor.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        outline_color = QColor(0, 0, 255, 155)  # Transparent blue

        # Draw the bubble (outline only, no fill)
        painter.setPen(QPen(outline_color, 5))
        painter.drawEllipse(self.bubble_pos, self.bubble_radius, self.bubble_radius)

    def is_cursor_in_bubble(self, target_pos):
        """
        Check if a target position is inside the bubble.
        """
        return (self.bubble_pos.x() - self.bubble_radius <= target_pos.x() <= self.bubble_pos.x() + self.bubble_radius
                and
                self.bubble_pos.y() - self.bubble_radius <= target_pos.y() <= self.bubble_pos.y() + self.bubble_radius)


def show_startup_instructions():
    """
    Displays the instructions for using the eye tracker app.
    """
    instructions = """
       **Eye Tracker Instructions:**

       1. **Face Recognition**:
          - Look at the camera to enable face recognition.
          - Ensure your face is well-lit and clearly visible.
          - You can upload a photo of yourself for face recognition by clicking the **"Add Photo for Face Recognition"** button in the settings.

       2. **Eye Tracking**:
          - Use your eyes to control the cursor.
          - Wink with your left eye for clicks and right eye for right-clicks.

       3. **Cursor Speed**:
          - Adjust cursor speed using the slider in settings.

       4. **Handwriting Recognition**:
          - Open the handwriting recognition window from the taskbar to draw and recognize digits or symbols.

       5. **Click2Speak Keyboard**:
          - The **Keyboard Button** in the taskbar allows you to launch the Click2Speak on-screen keyboard.
          - **Important**: You must install Click2Speak to use this feature.
            - Click the **"Install Click2Speak Keyboard"** button in the settings to download and install it.
            - Once installed, you can launch it using the **"Launch Click2Speak"** button.

       6. **Troubleshooting**:
          - If the app doesn't start, ensure your antivirus isn't blocking it.
          - Make sure all required files (e.g., models) are in the same folder as the executable.
       """
    messagebox.showinfo("Instructions", instructions)


def run_instructions_in_thread():
    """
    Runs the instructions pop-up in a separate thread.
    """
    instructions_thread = threading.Thread(target=show_startup_instructions, daemon=True)
    instructions_thread.start()

def run_eyewrittin_gui():
    """Run the EyewrittinRecognizer GUI in a separate process."""
    eyewrittin_window = EyewrittinRecognizer()
    eyewrittin_window.root.mainloop()

class Taskbar(QWidget):
    speed_signal = pyqtSignal(float)

    def __init__(self):
        super().__init__()

        # Full taskbar size when visible
        self.taskbar_height = 50  # Set height of the taskbar when visible
        self.hidden_height = 10  # Set height of the taskbar when hidden (black bar)

        # Taskbar setup at the top-center of the screen
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        # Enable transparency when there's something under the taskbar
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Calculate width of all buttons and space between them
        button_width = 100  # Approximate width of each button
        num_buttons = 8  # Add one more for the settings button
        taskbar_width = button_width * num_buttons + 40  # Total taskbar width with spacing

        # Set taskbar position to be under the top center of the screen
        self.setGeometry((screen_width - taskbar_width) // 2, 0, taskbar_width, self.hidden_height)

        self.setMouseTracking(True)  # Track mouse movement for hover detection

        layout = QHBoxLayout()

        # "Mouse Speed:" label
        label = QPushButton("Mouse Speed:")
        label.setEnabled(False)
        label.setStyleSheet(""" 
            QPushButton {
                background-color: transparent;
                color: white;
                font-size: 16px;
                border: none;
            }
        """)
        layout.addWidget(label)

        # Speed buttons
        speeds = [("Slow", 1.0), ("Normal", 1.5), ("Fast", 2.0)]
        for label, multiplier in speeds:
            button = QPushButton(label)
            button.setStyleSheet(""" 
                QPushButton {
                    background-color: rgba(100, 100, 255, 200);
                    border: none;
                    color: white;
                    font-size: 16px;
                    padding: 8px 16px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: rgba(120, 120, 255, 250);
                }
            """)
            button.clicked.connect(lambda _, m=multiplier: self.set_speed(m))
            layout.addWidget(button)

        # Additional buttons (Keyboard, Calculator)
        buttons = [
            ("Keyboard", r"C:\Program Files (x86)\Click2Speak\Click2Speak.Client.exe"),
            ("Calculator", "calc")
        ]

        for name, command in buttons:
            button = QPushButton(name)
            button.setStyleSheet(""" 
                QPushButton {
                    background-color: rgba(200, 50, 50, 200);
                    border: none;
                    color: white;
                    font-size: 16px;
                    padding: 8px 16px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: rgba(220, 70, 70, 250);
                }
            """)
            button.clicked.connect(lambda _, c=command: self.run_command(c))
            layout.addWidget(button)

        # Add Eyewrittin Recognition button
        eyewrittin_button = QPushButton("Eyewrittin")
        eyewrittin_button.setStyleSheet(""" 
            QPushButton {
                background-color: rgba(50, 150, 50, 200);
                border: none;
                color: white;
                font-size: 16px;
                padding: 8px 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: rgba(70, 170, 70, 250);
            }
        """)
        eyewrittin_button.clicked.connect(self.launch_eyewrittin_recognition)
        layout.addWidget(eyewrittin_button)

        # Add Settings Button
        settings_button = QPushButton("Settings")
        settings_button.setStyleSheet(""" 
            QPushButton {
                background-color: rgba(50, 150, 50, 200);
                border: none;
                color: white;
                font-size: 16px;
                padding: 8px 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: rgba(70, 170, 70, 250);
            }
        """)
        settings_button.clicked.connect(self.open_settings)
        layout.addWidget(settings_button)

        self.setLayout(layout)

        # Hide/show timer
        self.hide_timer = QTimer(self)
        self.hide_timer.timeout.connect(self.check_mouse_position)
        self.hide_timer.start(100)

        # Show the Settings UI on startup
        self.open_settings()

    def open_settings(self):
        """
        Opens the settings window in a separate thread.
        """
        settings_thread = threading.Thread(target=self.run_settings_window, daemon=True)
        settings_thread.start()

    def run_settings_window(self):
        """
        Runs the CustomTkinter settings window.
        """
        # Initialize the CustomTkinter settings window
        settings_window = SettingsUI()
        settings_window.mainloop()

    def set_speed(self, multiplier):
        global speed_multiplier
        speed_multiplier = multiplier
        self.speed_signal.emit(multiplier)

    @staticmethod
    def run_command(command):
        try:
            pyautogui.hotkey('win', 'r')  # Open Run command
            pyautogui.typewrite(command)
            pyautogui.press('enter')
        except Exception as e:
            print(f"Error launching {command}: {e}")

    def launch_eyewrittin_recognition(self):
        """Launch the EyewrittinRecognizer window in a separate process."""
        eyewrittin_process = Process(target=run_eyewrittin_gui)
        eyewrittin_process.start()

    def check_mouse_position(self):
        x, y = pyautogui.position()
        taskbar_x = (screen_width - self.width()) // 2
        taskbar_y = 0

        # Check if the mouse is hovering over the taskbar
        if taskbar_x <= x <= taskbar_x + self.width() and taskbar_y <= y <= taskbar_y + self.taskbar_height:
            self.show_taskbar()
        else:
            self.hide_taskbar()

    def show_taskbar(self):
        # Adjust taskbar geometry and appearance when showing
        self.setGeometry((screen_width - self.width()) // 2, 0, self.width(), self.taskbar_height)
        self.setStyleSheet("background-color: rgba(50, 50, 50, 150); border-radius: 20px;")
        self.show()

    def hide_taskbar(self):
        # Adjust taskbar geometry and appearance when hiding (black bar)
        self.setGeometry((screen_width - self.width()) // 2, 0, self.width(), self.hidden_height)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.hide()


class SettingsUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Eye Tracker Settings")
        self.geometry("600x400")
        self.resizable(False, False)

        # Center the window on the screen
        self.center_window()

        # Set appearance mode and color theme
        ctk.set_appearance_mode("System")  # "System", "Dark", or "Light"
        ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue", etc.

        # Create the UI
        self.create_widgets()

    def center_window(self):
        """
        Centers the window on the screen.
        """
        window_width = 600  # Width of the settings window
        window_height = 400  # Height of the settings window

        # Calculate the center position
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        # Set the window's position
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def create_widgets(self):
        """
        Creates the widgets for the settings UI.
        """
        # Cursor Speed Slider
        self.speed_label = ctk.CTkLabel(self, text="Cursor Speed:", font=("Arial", 16))
        self.speed_label.pack(pady=10)

        self.speed_slider = ctk.CTkSlider(self, from_=1, to=3, number_of_steps=2, command=self.update_speed)
        self.speed_slider.set(speed_multiplier * 2)  # Map speed_multiplier to slider
        self.speed_slider.pack(pady=10)

        # Enable/Disable Features
        self.enable_face_recognition = ctk.CTkCheckBox(self, text="Enable Face Recognition", font=("Arial", 14))
        self.enable_face_recognition.pack(pady=10)

        self.enable_eye_tracking = ctk.CTkCheckBox(self, text="Enable Eye Tracking", font=("Arial", 14))
        self.enable_eye_tracking.pack(pady=10)

        # Add Photo Button
        self.add_photo_button = ctk.CTkButton(
            self, text="Add Photo for Face Recognition", font=("Arial", 14), command=self.add_photo
        )
        self.add_photo_button.pack(pady=10)

        # Install Click2Speak Button
        self.install_click2speak_button = ctk.CTkButton(
            self, text="Install Click2Speak Keyboard", font=("Arial", 14), command=self.install_click2speak
        )
        self.install_click2speak_button.pack(pady=10)

        # Launch Click2Speak Button
        self.launch_click2speak_button = ctk.CTkButton(
            self, text="Launch Click2Speak", font=("Arial", 14), command=self.launch_click2speak
        )
        self.launch_click2speak_button.pack(pady=10)

        # Instructions Button
        self.instructions_button = ctk.CTkButton(
            self, text="View Instructions", font=("Arial", 14), command=self.show_instructions
        )
        self.instructions_button.pack(pady=10)

        # Save Button
        self.save_button = ctk.CTkButton(
            self, text="Save Settings", font=("Arial", 14), command=self.save_settings
        )
        self.save_button.pack(pady=10)

    def update_speed(self, value):
        """
        Updates the cursor speed based on the slider value.
        """
        global speed_multiplier
        speed_multiplier = value / 2.0
        print(f"Cursor speed set to: {speed_multiplier}")

    def install_click2speak(self):
        """
        Opens the Click2Speak download page in the default web browser.
        """
        webbrowser.open("https://www.click2speak.net/download-sw/")

    def launch_click2speak(self):
        """
        Launches Click2Speak if it is installed.
        """
        click2speak_path = find_click2speak_path()  # Call the function to find Click2Speak
        if click2speak_path:
            try:
                subprocess.Popen([click2speak_path])  # Launch Click2Speak
                print("Click2Speak launched successfully!")
            except Exception as e:
                print(f"Failed to launch Click2Speak: {e}")
        else:
            print("Click2Speak is not installed. Please install it first.")

    def show_instructions(self):
        """
        Displays the instructions for using the eye tracker app.
        """
        instructions = """
           **Eye Tracker Instructions:**

           1. **Face Recognition**:
              - Look at the camera to enable face recognition.
              - Ensure your face is well-lit and clearly visible.
              - You can upload a photo of yourself for face recognition by clicking the **"Add Photo for Face Recognition"** button in the settings.

           2. **Eye Tracking**:
              - Use your eyes to control the cursor.
              - Wink with your left eye for clicks and right eye for right-clicks.

           3. **Cursor Speed**:
              - Adjust cursor speed using the slider in settings.

           4. **Handwriting Recognition**:
              - Open the handwriting recognition window from the taskbar to draw and recognize digits or symbols.

           5. **Click2Speak Keyboard**:
              - The **Keyboard Button** in the taskbar allows you to launch the Click2Speak on-screen keyboard.
              - **Important**: You must install Click2Speak to use this feature.
                - Click the **"Install Click2Speak Keyboard"** button in the settings to download and install it.
                - Once installed, you can launch it using the **"Launch Click2Speak"** button.

           6. **Troubleshooting**:
              - If the app doesn't start, ensure your antivirus isn't blocking it.
              - Make sure all required files (e.g., models) are in the same folder as the executable.
           """
        messagebox.showinfo("Instructions", instructions)

    def save_settings(self):
        """
        Saves the settings and closes the settings window.
        """
        print("Settings saved!")
        self.destroy()

    def add_photo(self):
        """
        Allows the user to upload a photo for face recognition.
        """
        # Open a file dialog to select an image
        file_path = filedialog.askopenfilename(
            title="Select a Photo for Face Recognition",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )

        if file_path:
            try:
                # Load the selected image
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError("Unable to load the image. Please ensure the file is a valid image.")

                # Process the image for face recognition
                descriptor = get_face_descriptor(image)
                if descriptor is not None:
                    # Save the descriptor for future use (e.g., in a global variable or file)
                    global reference_descriptor
                    reference_descriptor = descriptor
                    messagebox.showinfo("Success", "Photo successfully added for face recognition!")
                else:
                    messagebox.showerror("Error", "No face detected in the selected photo. Please try another image.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process the image: {e}")


# Constants
WIDTH = 1000  # Reduced from 3500
HEIGHT = 500  # Reduced from 750
MODEL_PATH = 'mnist_trained_model_3_layer.h5'


class EyewrittinRecognizer:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.image1 = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))  # This is the instance variable
        self.draw = ImageDraw.Draw(self.image1)
        self.preds = []
        self.i = 0  # Initialize i for image counter
        self.recognized_digits = ""  # Store recognized digits
        self.setup_gui()

    def setup_gui(self):
        self.root = ctk.CTk()
        self.root.resizable(0, 0)
        self.root.title('EYEWRITTIN RECOGNIZER')

        # Canvas for drawing numbers
        self.canv = ctk.CTkCanvas(self.root, width=WIDTH, height=HEIGHT, bg='white')
        self.canv.grid(row=0, column=0, columnspan=3, padx=10, pady=17)
        self.canv.bind("<B1-Motion>", self.paint)

        # Textbox for recognized digits
        self.sol = self.create_textbox(30, width=WIDTH // 3)

        # Image box
        self.labimg = Image.open("Blank.png")
        self.labimg = ctk.CTkImage(dark_image=self.labimg, size=(WIDTH // 5, HEIGHT // 5))
        self.image_label = ctk.CTkLabel(self.root, image=self.labimg, text="")
        self.image_label.grid(row=2, column=2, padx=10, pady=5, rowspan=2)

        # Buttons
        self.create_buttons()

        self.root.mainloop()

    def create_textbox(self, font_size, width=None):
        text_font = ctk.CTkFont(family="Bahnschrift", size=font_size, weight='bold')
        textbox = ctk.CTkTextbox(self.root, exportselection=0,
                                 padx=10, pady=10, height=HEIGHT // 10, width=width, font=text_font,
                                 text_color='#3085ff')
        textbox.grid(row=2, column=0, padx=0, pady=3, columnspan=2)
        return textbox

    def create_buttons(self):
        button_font = ctk.CTkFont(family="Bahnschrift", size=20)  # Increased font size
        button_height = HEIGHT // 15  # Increased button height
        button_width = WIDTH // 6  # Increased button width

        self.Pred = ctk.CTkButton(
            self.root, text="Recognize", command=self.mod, fg_color='#0056C4', hover_color='#007dfe',
            font=button_font, height=button_height, width=button_width
        )
        self.ClrCanvas = ctk.CTkButton(
            self.root, text="Clear Canvas", command=self.clear_canvas, fg_color='#B50000', hover_color='#dd0000',
            font=button_font, height=button_height, width=button_width
        )
        self.ClrDigits = ctk.CTkButton(
            self.root, text="Clear Recognized Digits", command=self.clear_digits, fg_color='#B50000',
            hover_color='#dd0000',
            font=button_font, height=button_height, width=button_width
        )

        self.Pred.grid(row=1, column=0, padx=10, pady=10, sticky='ew')
        self.ClrCanvas.grid(row=1, column=1, padx=10, pady=10, sticky='ew')
        self.ClrDigits.grid(row=1, column=2, padx=10, pady=10, sticky='ew')

    def paint(self, event):
        d = 15
        x1, y1 = (event.x - d), (event.y - d)
        x2, y2 = (event.x + d), (event.y + d)
        self.canv.create_oval(x1, y1, x2, y2, fill="black", width=25)
        self.draw.line([x1, y1, x2, y2], fill="black", width=35)

    def clear_canvas(self):
        self.canv.delete('all')
        self.draw.rectangle((0, 0, WIDTH, HEIGHT), fill=(255, 255, 255, 0))

    def clear_digits(self):
        self.recognized_digits = ""  # Clear the stored recognized digits
        self.sol.delete('1.0', ctk.END)

    def testing(self, img):
        img = cv2.bitwise_not(img)
        img = cv2.resize(img, (28, 28))
        img = img.reshape(1, 28, 28, 1)
        img = img.astype('float32')
        img = img / 255.0
        return self.model.predict(img)

    def num_to_sym(self, x):
        symbol_map = {
            10: '+', 11: '-', 12: '*', 13: '/', 14: '(', 15: ')', 16: '.'
        }
        return symbol_map.get(x, str(x))

    def solve_exp(self, preds):
        ans = ""
        for ind, acc in preds:
            ans += self.num_to_sym(ind)
            print(self.num_to_sym(ind) + " " + str(acc))

        try:
            fin = eval(ans)
            fin = float(f"{fin:.4f}")
            return f"{ans} = {fin}"
        except Exception:
            return f"{ans} (Invalid expression)"

    def mod(self):
        self.image1.save('image.png')  # Access image1 using self.
        img = cv2.imread('image.png')

        pad = 5
        h, w = img.shape[:2]
        im2 = ~(np.ones((h + pad * 2, w + pad * 2, 3), dtype=np.uint8))
        im2[pad:pad + h, pad:pad + w] = img[:]
        img = im2

        img = cv2.GaussianBlur(img, (5, 5), 5)
        im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bw = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY)[1]

        bw = cv2.bitwise_not(bw)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0] + cv2.boundingRect(x)[2])

        self.preds = []
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            self.i += 1
            cropped_img = im[y:y + h, x:x + w]

            if abs(h) > 1.25 * abs(w):
                pad = 3 * (h // w) ** 3
                cropped_img = cv2.copyMakeBorder(cropped_img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=255)

            if abs(w) > 1.1 * abs(h):
                pad = 3 * (w // h) ** 3
                cropped_img = cv2.copyMakeBorder(cropped_img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=255)

            resized_img = cv2.resize(cropped_img, (28, 28))
            padded_img = cv2.copyMakeBorder(resized_img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)

            predi = self.testing(padded_img)
            ind = np.argmax(predi[0])
            acc = predi[0][ind] * 100
            acc = float(f"{acc:.2f}")

            self.preds.append((ind, acc))

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 230, 0), 7)

            yim = y + h + 85 if y < 80 else y - 25
            cv2.putText(img, f"{self.num_to_sym(ind)}", (x, yim), cv2.FONT_HERSHEY_SIMPLEX, 3, (225, 0, 0), 10)
            cv2.putText(img, f"{acc}%", (x + 75, yim), cv2.FONT_HERSHEY_DUPLEX, 1.75, (0, 0, 225), 3)

        cv2.imwrite('Contours.png', img)
        self.img_change()
        self.display_result()

    def img_change(self):
        labimg = Image.open('Contours.png')
        labimg = ctk.CTkImage(dark_image=labimg, size=(WIDTH // 5, HEIGHT // 5))
        self.image_label.configure(image=labimg)
        self.image_label.image = labimg

    def display_result(self):
        result = self.solve_exp(self.preds)  # Get the recognized expression and its result
        self.sol.delete('1.0', ctk.END)
        self.sol.insert(ctk.INSERT, result)  # Display the result


# Function to get face descriptor for a given image
def get_face_descriptor(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) > 0:
        face = faces[0]  # Consider the first face detected
        landmarks = sp(image, face)
        descriptor = facerec.compute_face_descriptor(image, landmarks)
        return np.array(descriptor)
    else:
        return None


# Function to compare two face descriptors
def compare_faces(descriptor1, descriptor2, threshold=0.6):
    distance = np.linalg.norm(descriptor1 - descriptor2)
    return distance < threshold


# Load reference image and compute its descriptor
reference_image = cv2.imread(r"C:\Users\msi-GF63\Desktop\IMG_20241222_235130.jpg")  # Path to your reference image
reference_descriptor = get_face_descriptor(reference_image)


# Function for face recognition and tracking
def face_recognition():
    cap = cv2.VideoCapture(0)  # Capture from default camera

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        # Detect faces
        faces = detector(gray)

        if len(faces) > 0:
            for face in faces:
                landmarks = sp(frame, face)  # Get landmarks
                face_descriptor = facerec.compute_face_descriptor(frame, landmarks)  # Get face descriptor

                # Compare the detected face with the reference face
                if compare_faces(reference_descriptor, np.array(face_descriptor)):
                    # Draw a green rectangle and label when the face is recognized
                    x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Face Recognized", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Call eye tracking function here if face is recognized
                    return True  # Return True when face is recognized
                else:
                    # Draw a red rectangle if the face doesn't match
                    x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Face Not Recognized", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                                2)

        # Show the frame
        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
            break

    cap.release()
    cv2.destroyAllWindows()
    return False  # Return False if no face is recognized


# Eye tracking and click detection
def eye_tracking():
    # Wait for face recognition to complete
    if not face_recognition():
        print("Face not recognized. Exiting...")
        return

    cap = cv2.VideoCapture(0)
    smoothing_factor = 0.2

    smooth_x, smooth_y = screen_width // 2, screen_height // 2
    gaze_origin = None

    def eye_aspect_ratio(eye):
        """
        Calculate the Eye Aspect Ratio (EAR) to detect eye closure.
        """
        vertical_1 = np.linalg.norm(eye[1] - eye[5])
        vertical_2 = np.linalg.norm(eye[2] - eye[4])
        horizontal = np.linalg.norm(eye[0] - eye[3])
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    wink_threshold = 0.2  # Wink EAR threshold
    click_frames = 1  # Frames to detect click
    consec_frames = 2  # Consecutive frames to confirm a wink
    double_click_frames = 3  # Frames to detect double click
    left_wink_counter = 0
    right_wink_counter = 0
    left_hold = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get eye landmarks
                left_eye_pupil = np.array([[
                    face_landmarks.landmark[i].x * frame.shape[1],
                    face_landmarks.landmark[i].y * frame.shape[0]
                ] for i in LEFT_EYE_PUPIL])

                right_eye_pupil = np.array([[
                    face_landmarks.landmark[i].x * frame.shape[1],
                    face_landmarks.landmark[i].y * frame.shape[0]
                ] for i in RIGHT_EYE_PUPIL])

                # Extract eye landmarks
                left_eye = np.array([[face_landmarks.landmark[i].x * frame.shape[1],
                                      face_landmarks.landmark[i].y * frame.shape[0]] for i in LEFT_EYE])
                right_eye = np.array([[face_landmarks.landmark[i].x * frame.shape[1],
                                       face_landmarks.landmark[i].y * frame.shape[0]] for i in RIGHT_EYE])

                # Calculate eye centers
                left_eye_center = np.mean(left_eye, axis=0)
                right_eye_center = np.mean(right_eye, axis=0)

                if gaze_origin is None:
                    gaze_origin = (left_eye_center + right_eye_center) / 2

                # Calculate movement
                gaze_x_movement = (left_eye_center[0] + right_eye_center[0]) / 2 - gaze_origin[0]
                gaze_y_movement = (left_eye_center[1] + right_eye_center[1]) / 2 - gaze_origin[1]

                # Map movement to screen
                move_x = np.interp(gaze_x_movement, [-20, 20], [0, screen_width]) * speed_multiplier
                move_y = np.interp(gaze_y_movement, [-20, 20], [0, screen_height]) * speed_multiplier

                # Smooth cursor movement
                smooth_x += (move_x - smooth_x) * smoothing_factor
                smooth_y += (move_y - smooth_y) * smoothing_factor
                pyautogui.moveTo(smooth_x, smooth_y, duration=0.01)

                # Calculate EAR for both eyes
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)

                # Check for left-eye wink (for click, double-click, or hold)
                if left_ear < wink_threshold < right_ear:  # Left eye wink
                    left_wink_counter += 1
                    if left_wink_counter >= double_click_frames:
                        pyautogui.doubleClick()
                        left_wink_counter = 0
                    elif left_wink_counter >= consec_frames:
                        if not left_hold:
                            left_hold = True
                            pyautogui.mouseDown()
                        else:
                            left_hold = False
                            pyautogui.mouseUp()  # Trigger mouse up if left eye opens
                else:
                    left_wink_counter = 0

                # Check for right-eye wink (for right-click)
                if right_ear < wink_threshold < left_ear:  # Right eye wink
                    right_wink_counter += 1
                    if right_wink_counter >= click_frames:
                        pyautogui.rightClick()  # Trigger right-click
                        right_wink_counter = 0
                else:
                    right_wink_counter = 0

                # Draw unconnected points for eye landmarks
                for (x, y) in left_eye_pupil:
                    cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
                for (x, y) in right_eye_pupil:
                    cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)

                cv2.imshow('Eye Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create and show the overlay and taskbar
    overlay = Overlay()
    taskbar = Taskbar()
    taskbar.speed_signal.connect(overlay.update_bubble_position)

    overlay.show()
    taskbar.show()
    # Show instructions at the beginning of the program in a separate thread
    run_instructions_in_thread()

    # Start eye tracking
    eye_tracking()

    sys.exit(app.exec_())
