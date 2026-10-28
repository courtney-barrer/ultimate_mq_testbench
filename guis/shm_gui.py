import sys
import socket
import subprocess
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
                             QVBoxLayout, QHBoxLayout, QFormLayout, 
                             QLineEdit, QPushButton, QLabel, QGroupBox, 
                             QPlainTextEdit, QCheckBox)
from PyQt5.QtCore import QTimer
import ImageStreamIOWrap

class ServerCameraGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hamamatsu DCAM Server Interface")
        self.resize(1200, 800)

        # Connect to Shared Memory
        self.cam = ImageStreamIOWrap.Image()
        try:
            self.cam.open("dcam0_data")
            self.shm_connected = True
        except Exception as e:
            print(f"Warning: SHM not connected. {e}")
            self.shm_connected = False

        self.setup_ui()
        self.start_timers()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # --- Left Panel: Controls & Logs ---
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)

        # 1. State Controls
        state_group = QGroupBox("Server / Capture State")
        state_layout = QHBoxLayout(state_group)
        self.btn_open = QPushButton("Open")
        self.btn_close = QPushButton("Close")
        self.btn_run = QPushButton("Run (Live)")
        self.btn_stop = QPushButton("Stop")
        
        self.btn_open.clicked.connect(lambda: self.send_command("open"))
        self.btn_close.clicked.connect(lambda: self.send_command("close"))
        self.btn_run.clicked.connect(lambda: self.send_command("run"))
        self.btn_stop.clicked.connect(lambda: self.send_command("stop"))

        state_layout.addWidget(self.btn_open)
        state_layout.addWidget(self.btn_close)
        state_layout.addWidget(self.btn_run)
        state_layout.addWidget(self.btn_stop)
        left_layout.addWidget(state_group)

        # 2. Hardware Settings
        settings_group = QGroupBox("Hardware Settings")
        form = QFormLayout(settings_group)
        
        self.inp_exp = QLineEdit("0.01")
        self.inp_size = QLineEdit("2304 2304")
        self.inp_offset = QLineEdit("0 0")
        
        form.addRow("Exposure (s):", self.inp_exp)
        form.addRow("Size (W H):", self.inp_size)
        form.addRow("Offset (X Y):", self.inp_offset)
        
        # --- NEW: Auto Contrast Toggle ---
        self.chk_auto = QCheckBox("Auto Contrast")
        self.chk_auto.setChecked(True)
        form.addRow("", self.chk_auto)
        
        self.btn_apply = QPushButton("Apply Settings")
        self.btn_apply.clicked.connect(self.apply_settings)
        form.addRow(self.btn_apply)
        left_layout.addWidget(settings_group)

        # 3. Tmux Session Watcher
        log_group = QGroupBox("Live Tmux Session (dcam0_server)")
        log_layout = QVBoxLayout(log_group)
        self.tmux_log = QPlainTextEdit()
        self.tmux_log.setReadOnly(True)
        self.tmux_log.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: monospace;")
        log_layout.addWidget(self.tmux_log)
        left_layout.addWidget(log_group)

        layout.addWidget(left_panel)

        # --- Right Panel: Image Viewer ---
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        layout.addWidget(self.image_view, 1)

        if self.shm_connected:
            frame = np.array(self.cam, copy=False)
            # Let pyqtgraph automatically stretch the contrast on the first frame
            self.image_view.setImage(frame.T, autoLevels=self.chk_auto.isChecked())

    def start_timers(self):
        # High-speed timer for shared memory rendering
        self.img_timer = QTimer()
        self.img_timer.timeout.connect(self.update_frame)
        self.img_timer.start(30)  # ~33 FPS

        # Low-speed timer for scraping the background tmux terminal
        self.tmux_timer = QTimer()
        self.tmux_timer.timeout.connect(self.update_tmux_log)
        self.tmux_timer.start(1000) # 1 update per second

    def update_frame(self):
        if self.shm_connected:
            try:
                frame = np.array(self.cam, copy=False)
                # Apply the contrast levels based on the checkbox state
                do_auto = self.chk_auto.isChecked()
                # autoRange=False prevents the image from zooming out wildly on every frame
                self.image_view.setImage(frame.T, autoLevels=do_auto, autoRange=False)
            except Exception:
                pass

    def update_tmux_log(self):
        """Silently scrapes the background tmux session to provide a live log."""
        try:
            result = subprocess.run(["tmux", "capture-pane", "-p", "-t", "dcam0"], 
                                    capture_output=True, text=True, timeout=0.5)
            if result.returncode == 0:
                new_text = result.stdout.strip()
                if new_text != self.tmux_log.toPlainText():
                    self.tmux_log.setPlainText(new_text)
                    scrollbar = self.tmux_log.verticalScrollBar()
                    scrollbar.setValue(scrollbar.maximum())
            else:
                self.tmux_log.setPlainText("[!] Tmux session 'dcam0' not found or server is offline.")
        except Exception as e:
            self.tmux_log.setPlainText(f"Error querying tmux: {e}")

    def send_command(self, command_string):
        """Native TCP client routing commands to port 18010."""
        print(f"Executing: {command_string}")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect(('127.0.0.1', 18010))
                s.sendall(f"dcam0 {command_string}\r".encode('ascii'))
                response = s.recv(1024).decode('ascii').strip()
                print(f"Server response: {response}")
                return response
        except Exception as e:
            print(f"TCP Error: {e}")
            return None

    def apply_settings(self):
        """Applies the text inputs to the background server sequentially."""
        exp = self.inp_exp.text()
        size = self.inp_size.text()
        offset = self.inp_offset.text()
        
        # Stop capture before applying hardware settings
        self.send_command("stop")
        self.send_command(f"exp {exp}")
        self.send_command(f"size {size}")
        self.send_command(f"offset {offset}")
        self.send_command("run")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = ServerCameraGUI()
    gui.show()
    sys.exit(app.exec_())

########## pyqtgraph with only exposure time control via client 
# import sys
# import socket
# import numpy as np
# import pyqtgraph as pg
# from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
#                              QVBoxLayout, QHBoxLayout, QLineEdit, 
#                              QPushButton, QLabel)
# from PyQt5.QtCore import QTimer
# import ImageStreamIOWrap

# class CameraGUI(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Hamamatsu DCAM Control")
#         self.resize(800, 700)

#         # --- Hardware Connections ---
#         # 1. Connect to the shared memory stream
#         self.cam = ImageStreamIOWrap.Image()
#         self.cam.open("dcam0_data")

#         # --- GUI Layout ---
#         central_widget = QWidget()
#         self.setCentralWidget(central_widget)
#         layout = QVBoxLayout(central_widget)

#         # Top Control Panel
#         control_layout = QHBoxLayout()
        
#         self.exp_label = QLabel("Exposure (s):")
#         self.exp_input = QLineEdit("0.01")
#         self.exp_button = QPushButton("Set Exposure")
        
#         # Connect the button click to our function
#         self.exp_button.clicked.connect(self.update_exposure)
#         # Also allow hitting 'Enter' while typing in the text box
#         self.exp_input.returnPressed.connect(self.update_exposure)
        
#         control_layout.addWidget(self.exp_label)
#         control_layout.addWidget(self.exp_input)
#         control_layout.addWidget(self.exp_button)
#         control_layout.addStretch() # Pushes controls to the left side
        
#         layout.addLayout(control_layout)

#         # Bottom Image Viewer
#         self.image_view = pg.ImageView()
#         layout.addWidget(self.image_view)

#         # Set initial image and stretch the contrast to the noise floor
#         initial_frame = np.array(self.cam, copy=False)
#         # pyqtgraph expects column-major data (x, y), so we transpose (.T) the numpy array
#         self.image_view.setImage(initial_frame.T, autoLevels=False)
#         self.image_view.setLevels(80, 120)

#         # --- Update Loop ---
#         # QTimer runs asynchronously in the background so the GUI doesn't freeze
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.update_frame)
#         self.timer.start(30) # ~33 FPS (30 ms interval)

#     def update_frame(self):
#         """Pulls the latest frame from shared memory and pushes it to the screen."""
#         frame = np.array(self.cam, copy=False)
#         # Update image without resetting the user's zoom or contrast levels every frame
#         self.image_view.setImage(frame.T, autoLevels=False, autoRange=False)

#     def send_dcam_command(self, command_string):
#         """Sends a raw ASCII text command to the background TCP server."""
#         HOST = '127.0.0.1'
#         PORT = 18010
#         full_message = f"dcam0 {command_string}\r"
        
#         try:
#             with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#                 s.settimeout(1.0) # Prevent GUI freezing if the server crashes
#                 s.connect((HOST, PORT))
#                 s.sendall(full_message.encode('ascii'))
#                 response = s.recv(1024).decode('ascii')
#                 return response.strip()
#         except Exception as e:
#             return f"Error: {e}"

#     def update_exposure(self):
#         """Triggered when the button is clicked or Enter is pressed."""
#         new_exp = self.exp_input.text()
#         print(f"Sending command: exp {new_exp}")
        
#         # Send the command through the native socket
#         response = self.send_dcam_command(f"exp {new_exp}")
        
#         print(f"Server response: {response}")

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     gui = CameraGUI()
#     gui.show()
#     sys.exit(app.exec_())


# ######### SIMPLEST VIEWER 

# # import ImageStreamIOWrap
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import matplotlib.animation as animation

# # # Connect to the memory stream
# # cam = ImageStreamIOWrap.Image()
# # cam.open("dcam0_data")

# # #  Setup the Matplotlib Figure
# # fig, ax = plt.subplots()
# # plt.title('Hamamatsu Live Feed')

# # # Grab the initial frame to set up the plot
# # initial_frame = np.array(cam, copy=False)
# # im = ax.imshow(initial_frame, cmap='gray', vmin=80, vmax=120)
# # plt.colorbar(im)

# # #  Define the update loop for the live stream
# # def update_frame(i):
# #     # Grab the absolute latest state of the C++ memory buffer
# #     live_frame = np.array(cam, copy=False)
    
# #     # Push the new pixels to the display
# #     im.set_data(live_frame)
# #     return [im]

# # #  Run the animation at ~20 frames per second (interval in milliseconds)
# # ani = animation.FuncAnimation(fig, update_frame, interval=50, blit=True)

# # # Keep the window open and running
# # plt.show()