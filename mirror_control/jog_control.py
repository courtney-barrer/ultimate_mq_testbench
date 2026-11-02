"""USB jog interface for mirror_control.ino. No motion on connection."""
import time
import tkinter as tk
from tkinter import ttk, messagebox

import serial
from serial.tools import list_ports


class JogApp:
    def __init__(self, root):
        self.root = root
        self.port = None
        self.buffer = bytearray()
        self.last_reply = self.last_poll = 0.0
        self.ready = self.moving = self.referenced = False
        root.title("Mirror motor — USB control")
        root.protocol("WM_DELETE_WINDOW", self.close)
        frame = ttk.Frame(root, padding=16)
        frame.grid(sticky="nsew")
        ttk.Label(frame, text="Mirror motor control", font=("TkDefaultFont", 16, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 12))
        self.device = tk.StringVar()
        self.devices = ttk.Combobox(frame, textvariable=self.device, width=30)
        self.devices.grid(row=1, column=0, columnspan=2)
        ttk.Button(frame, text="Refresh ports", command=self.refresh).grid(row=1, column=2)
        self.connect_button = ttk.Button(frame, text="Connect", command=self.connect)
        self.connect_button.grid(row=1, column=3)
        self.state = tk.StringVar(value="Disconnected")
        ttk.Label(frame, textvariable=self.state).grid(row=2, column=0, columnspan=4, sticky="w", pady=12)
        self.position = tk.StringVar(value="Position: unknown")
        ttk.Label(frame, textvariable=self.position).grid(row=3, column=0, columnspan=4, sticky="w")
        self.idle_buttons = []
        self.motion_buttons = []
        self.zero = self.button(frame, "Set current position to zero", self.set_zero, 4, 0, 4)
        self.idle_buttons.append(self.zero)
        self.jog = self.entry(frame, "Jog size (steps)", "1", 5)
        self.motion_buttons.append(self.button(frame, "← Jog −", lambda: self.move(-1), 5, 2))
        self.motion_buttons.append(self.button(frame, "Jog + →", lambda: self.move(1), 5, 3))
        self.destination = self.entry(frame, "Target (steps)", "0", 6)
        self.motion_buttons.append(self.button(frame, "Go to target", self.goto, 6, 2, 2))
        self.speed = self.entry(frame, "Speed (10–200 steps/s)", "50", 7)
        self.idle_buttons.append(self.button(frame, "Set speed", self.set_speed, 7, 2, 2))
        self.low = self.entry(frame, "Lower limit (steps)", "-100", 8)
        self.high = self.entry(frame, "Upper limit (steps)", "100", 9)
        self.idle_buttons.append(self.button(frame, "Set limits", self.set_limits, 9, 2, 2))
        self.stop = tk.Button(frame, text="STOP  (Esc)", bg="#b42318", fg="white",
                              command=lambda: self.send("STOP"), font=("TkDefaultFont", 13, "bold"))
        self.stop.grid(row=10, column=0, columnspan=4, sticky="ew", pady=12)
        root.bind("<Escape>", lambda event: self.send("STOP"))
        ttk.Label(frame, text="Position counts commanded steps; it is not measured.\n"
                  "Full-step setting: 200 steps/revolution (1.8° per step).\n"
                  "STOP stops pulses; the motor remains energised.").grid(
                      row=11, column=0, columnspan=4, sticky="w")
        self.log = tk.Text(frame, width=72, height=8, state="disabled")
        self.log.grid(row=12, column=0, columnspan=4, pady=(12, 0))
        self.refresh()
        self.update_controls()
        root.after(50, self.tick)

    def button(self, frame, text, callback, row, column, span=1):
        button = ttk.Button(frame, text=text, command=callback)
        button.grid(row=row, column=column, columnspan=span, sticky="ew", padx=3, pady=5)
        return button

    def entry(self, frame, label, initial, row):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
        value = tk.StringVar(value=initial)
        ttk.Entry(frame, textvariable=value, width=12).grid(row=row, column=1, padx=6)
        return value

    def refresh(self):
        ports = [p.device for p in list_ports.comports()]
        self.devices["values"] = ports
        if ports and not self.device.get():
            self.device.set(ports[0])

    def record(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 150:
            self.log.delete("1.0", "30.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def connect(self):
        if self.port:
            self.disconnect()
            return
        try:
            self.port = serial.Serial(self.device.get(), 115200, timeout=0, write_timeout=0.2)
        except (serial.SerialException, ValueError) as error:
            messagebox.showerror("Connection failed", str(error))
            return
        self.buffer.clear()
        self.ready = self.referenced = self.moving = False
        self.last_reply = time.monotonic()
        self.state.set("Connecting; waiting for Arduino (it may reset)…")
        self.connect_button.configure(text="Disconnect")
        self.devices.configure(state="disabled")
        self.update_controls()

    def disconnect(self, reason="Disconnected"):
        if self.port:
            try:
                self.port.write(b"STOP\n")
            except (serial.SerialException, OSError):
                pass
            self.port.close()
        self.port = None
        self.ready = self.referenced = self.moving = False
        self.state.set(reason)
        self.position.set("Position: unknown")
        self.connect_button.configure(text="Connect")
        self.devices.configure(state="normal")
        self.update_controls()

    def send(self, command):
        if not self.port:
            return
        try:
            self.port.write((command + "\n").encode("ascii"))
            if command != "STATUS":
                self.record("> " + command)
        except (serial.SerialException, OSError) as error:
            self.disconnect("Connection lost: " + str(error))

    def integer(self, variable):
        try:
            value = int(variable.get())
            if abs(value) > 1000000:
                raise ValueError
            return value
        except ValueError:
            messagebox.showerror("Invalid value", "Enter a whole number between -1000000 and 1000000.")
            return None

    def set_zero(self):
        if messagebox.askokcancel("Set zero", "Treat this physical position as zero?\n"
                                 "This does not move or home the motor. Travel limits are relative to this zero."):
            self.send("ZERO")

    def move(self, sign):
        amount = self.integer(self.jog)
        if amount is not None and amount > 0:
            self.send(f"MOVE {sign * amount}")

    def goto(self):
        value = self.integer(self.destination)
        if value is not None:
            self.send(f"GOTO {value}")

    def set_speed(self):
        value = self.integer(self.speed)
        if value is not None:
            self.send(f"SPEED {value}")

    def set_limits(self):
        low, high = self.integer(self.low), self.integer(self.high)
        if low is not None and high is not None:
            self.send(f"LIMIT {low} {high}")

    def update_controls(self):
        for button in self.idle_buttons:
            button.configure(state="normal" if self.ready and not self.moving else "disabled")
        for button in self.motion_buttons:
            button.configure(state="normal" if self.ready and self.referenced and not self.moving else "disabled")
        self.stop.configure(state="normal" if self.port else "disabled")

    def receive(self, line):
        if line.startswith("READY MIRROR_CONTROL "):
            self.record(line)
        elif line.startswith("STATE "):
            try:
                pos, target, moving, reference, low, high, speed = map(int, line.split()[1:])
            except ValueError:
                self.record("Invalid state: " + line)
                return
            self.last_reply = time.monotonic()
            self.ready, self.moving, self.referenced = True, bool(moving), bool(reference)
            self.position.set(f"Position: {pos} steps | Target: {target} | Limits: {low} to {high} | Speed: {speed} steps/s")
            self.state.set("Moving" if moving else "Idle" if reference else "Connected — set current position to zero first")
            self.update_controls()
        elif line:
            self.record(line)

    def tick(self):
        if self.port:
            try:
                self.buffer.extend(self.port.read(4096))
                while b"\n" in self.buffer:
                    line, _, rest = self.buffer.partition(b"\n")
                    self.buffer = bytearray(rest)
                    self.receive(line.decode("ascii", errors="replace").strip())
                if len(self.buffer) > 8192:
                    self.buffer.clear()
                now = time.monotonic()
                if now - self.last_reply > (2.0 if self.ready else 6.0):
                    self.disconnect("No response — disconnected. Check firmware and USB.")
                elif now - self.last_poll >= 0.25:
                    self.last_poll = now
                    self.send("STATUS")  # Also keeps the firmware watchdog alive.
            except (serial.SerialException, OSError) as error:
                self.disconnect("Connection lost: " + str(error))
        self.root.after(50, self.tick)

    def close(self):
        self.disconnect()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    JogApp(root)
    root.mainloop()
