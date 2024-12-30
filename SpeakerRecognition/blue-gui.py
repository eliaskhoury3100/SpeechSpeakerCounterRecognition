import os
import sys
import tempfile
import sounddevice as sd
import numpy as np
import wave
tempfile
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QTextEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from speaker_recog_counter import predict_audio
from speaker_recog_no_counter import predict_audio_uncountered

# ----------------------------
# PyQt5 GUI Class
# ----------------------------
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.stream = None
        self.frames = []
        self.initUI()

    def initUI(self):
        # Set Window Properties
        self.setWindowTitle("Speech and Speaker Recognition System")

        # Apply Styling
        self.setStyleSheet("""
        QMainWindow {
            background-color: #2C3E50;  /* Dark Blue */
        }
        QPushButton {
            background-color: #34495E;  /* Slightly lighter dark blue */
            color: #ECF0F1;  /* Light Gray */
            font-size: 14px;
            border: 1px solid #BDC3C7;
            padding: 10px;
            border-radius: 20px;
        }
        QPushButton#page_toggle_btn {
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: #34495E;
            color: #ECF0F1;
            font-size: 20px;
            padding: 10px;
            border-radius: 15px;
            margin-right: 20px;
        }
        QPushButton#upload_btn, QPushButton#record_btn {
            padding-left: 20px;
            padding-right: 20px;
            margin-left: 800px;
            margin-right: 800px;
            height: 25px;
            border-radius: 12.5px;
        }
        QPushButton#upload_btn:hover {
            background-color: #1ABC9C;  /* Green hover color */
        }
        QPushButton#record_btn:hover {
            background-color: #3498DB;  /* Light blue hover color */
        }
        QPushButton#page_toggle_btn:hover {
            background-color: #9B59B6;  /* Purple hover color */
        }
        QLabel {
            color: #ECF0F1;  /* Light Gray */
        }
        QTextEdit {
            background-color: #34495E;  /* Slightly lighter dark blue */
            color: #ECF0F1;  /* Light Gray */
            border: 1px solid #BDC3C7;
            font-size: 14px;
            padding: 1px;
            margin-bottom: 5px;
            border-radius: 20px;
        }
        QTextEdit::placeholder {
            color: #95A5A6;  /* Light gray for placeholder text */
        }
        QVBoxLayout, QHBoxLayout {
            margin: 0;
            padding: 10px;
        }
        """)

        # Main Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Create Pages
        self.speech_page = self.create_speaker_page_not_countered()
        self.speaker_page = self.create_speaker_page_countered()

        # Add Toggle Button (Top-Right corner)
        self.page_toggle_btn = QPushButton("Go to Speaker Recognition - Countered")
        self.page_toggle_btn.setObjectName("page_toggle_btn")
        self.page_toggle_btn.setFont(QFont("Arial", 12))
        self.page_toggle_btn.clicked.connect(self.toggle_page)
        self.layout.addWidget(self.page_toggle_btn, alignment=Qt.AlignTop | Qt.AlignRight)

        # Add Pages
        self.layout.addWidget(self.speech_page)
        self.layout.addWidget(self.speaker_page)

        # Start with the Speech Page
        self.show_speech_page()

        # Show Window Maximized (Fullscreen)
        self.showMaximized()  # This makes the window fullscreen on startup

    def create_speaker_page_not_countered(self):
        """Create the Speaker Recognition Page."""
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.setContentsMargins(20, 30, 20, 10)  # Set margins, top margin is increased to move the title up

        # Title
        title = QLabel("Speaker Recognition - Not Countered")
        title.setFont(QFont("Arial", 50, QFont.Bold))  # Title font size set to 50px
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title, alignment=Qt.AlignCenter)
        layout.addStretch()
        self.speech_output = QTextEdit()
        self.speech_output.setFont(QFont("Arial", 12))
        self.speech_output.setReadOnly(True)
        self.speech_output.setPlaceholderText("NOT Countered Speaker prediction will appear here...")
        self.speech_output.setFixedWidth(600)  # Set the width to 600px
        self.speech_output.setFixedHeight(300)  # Set the height to 200px
        layout.addWidget(self.speech_output, alignment=Qt.AlignCenter)  # Center horizontally

        # Add some space between text area and buttons
        layout.addStretch()

        # Add Button for Running Recognition
        run_recognition_btn = QPushButton("Run Speaker Recognition")
        run_recognition_btn.setFont(QFont("Arial", 12))
        run_recognition_btn.clicked.connect(self.run_speaker_recognition_not_countered) # GHAYER LA TENE FCT L NON COUNTERED
        layout.addWidget(run_recognition_btn)

        # Add stretch below to ensure proper vertical alignment
        layout.addStretch()

        return page
    
    def run_speaker_recognition_not_countered(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.flac *.m4a)")
        if file_path:
            self.speech_output.setText("Processing...")
            try:
                speaker_name = "default_speaker"  # Replace with appropriate logic
                result = predict_audio_uncountered(file_path)
                if result:
                    top_class, confidence = result
                    self.speech_output.setText(f"Predicted Speaker: {top_class}\nConfidence: {confidence:.2f}")
                else:
                    self.speech_output.setText("Error: Unable to make a prediction.")
            except Exception as e:
                self.speech_output.setText(f"Error: {e}")
    

    def create_speaker_page_countered(self):
        """Create the Speaker Recognition Page."""
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.setContentsMargins(20, 30, 20, 10)  # Set margins, top margin is increased to move the title up

        # Title
        title = QLabel("Speaker Recognition - Countered")
        title.setFont(QFont("Arial", 50, QFont.Bold))  # Title font size set to 50px
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title, alignment=Qt.AlignCenter)
        layout.addStretch()
        self.speaker_output = QTextEdit()
        self.speaker_output.setFont(QFont("Arial", 12))
        self.speaker_output.setReadOnly(True)
        self.speaker_output.setPlaceholderText("Countered Speaker prediction will appear here...")
        self.speaker_output.setFixedWidth(600)  # Set the width to 600px
        self.speaker_output.setFixedHeight(300)  # Set the height to 200px
        layout.addWidget(self.speaker_output, alignment=Qt.AlignCenter)  # Center horizontally

        # Add some space between text area and buttons
        layout.addStretch()

        # Add Button for Running Recognition
        run_recognition_btn = QPushButton("Run Speaker Recognition")
        run_recognition_btn.setFont(QFont("Arial", 12))
        run_recognition_btn.clicked.connect(self.run_speaker_recognition)
        layout.addWidget(run_recognition_btn)

        # Add stretch below to ensure proper vertical alignment
        layout.addStretch()

        return page
    
    def run_speaker_recognition(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.flac *.m4a)")
        if file_path:
            self.speaker_output.setText("Processing...")
            try:
                speaker_name = "default_speaker"  # Replace with appropriate logic
                result = predict_audio(file_path, speaker_name)
                if result:
                    top_class, confidence = result
                    self.speaker_output.setText(f"Predicted Speaker: {top_class}\nConfidence: {confidence:.2f}")
                else:
                    self.speaker_output.setText("Error: Unable to make a prediction.")
            except Exception as e:
                self.speaker_output.setText(f"Error: {e}")

    def show_speech_page(self):
        """Show the Speech Recognition Page."""
        self.speech_page.show()
        self.speaker_page.hide()
        self.page_toggle_btn.setText("Go to Speaker Recognition - Countered")

    def show_speaker_page(self):
        """Show the Speaker Recognition Page."""
        self.speech_page.hide()
        self.speaker_page.show()
        
        self.page_toggle_btn.setText("Go to Speaker Recognition - Not Countered")

    def toggle_page(self):
        """Toggle between Speech and Speaker Recognition Pages.""" 
        if self.speech_page.isVisible():
            self.show_speaker_page()
        else:
            self.show_speech_page()

# ----------------------------
# Main Function
# ----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainApp()
    sys.exit(app.exec_())

