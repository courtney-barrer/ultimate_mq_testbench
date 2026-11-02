# Mirror Stepper Motor Control

Control a mirror-steering stepper motor for SLM scanning using either:

- **USB GUI:** manually jog the motor and test positioning.
- **ZeroMQ server and clients:** control the motor from other Python processes, such as scan or acquisition software.

Both use the same Arduino firmware and wiring. Run commands below from this subdirectory inside `mq_testbench`, not the repository root.

## Hardware and wiring

- Arduino UNO and USB cable
- DUINOTECH YM2756 NEMA17 stepper motor
- DFRobot DRI0043 TB6600 stepper driver
- Regulated 12 V DC, 2 A centre-positive plugpack
- Female 5.5 × 2.1 mm barrel-to-screw-terminal adaptor
- Jumper wires for control and thicker wires for driver power

The UNO is powered by USB. The separate 12 V supply powers the driver and motor. The original XC4472 controller is not used.

| From | To |
|---|---|
| Motor blue | Driver A+ |
| Motor red | Driver A− |
| Motor green | Driver B+ |
| Motor black | Driver B− |
| UNO D2 | Driver PUL− |
| UNO D3 | Driver DIR− |
| UNO 5V | Driver PUL+ |
| Driver PUL+ | Driver DIR+ |
| Power adaptor +, thicker wire | Driver VCC |
| Power adaptor −, thicker wire | Driver GND |

Leave **ENA+ and ENA− disconnected**. This common-anode, optically isolated arrangement does not require a connection between UNO GND and driver power GND.

The motor winding pairs were verified as blue/red and green/black. Verify them independently if replacing the motor or cable. Verify supply polarity with a multimeter before connecting the driver.

### Driver settings

| Switch | Setting |
|---|---|
| SW1 | ON |
| SW2 | ON |
| SW3 | OFF |
| SW4 | ON |
| SW5 | OFF |
| SW6 | ON |

These select **200 pulses per revolution**, or **1.8° per full step**, and the **1.0 A current setting** (labelled 1.2 A peak).

The software examples assume this full-step setting. Changing microstepping requires corresponding changes to angle conversion, limits and motion settings.

## Initial setup

Create and activate a Python environment in this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The GUI also requires Tkinter. On Debian/Ubuntu, if it is missing:

```bash
sudo apt install python3-tk
```

With the **12 V supply switched off**, connect the UNO by USB and upload `mirror_control.ino` using Arduino IDE. Select **Arduino Uno** and its USB port. Close Arduino Serial Monitor and Serial Plotter after uploading.

To identify the port:

```bash
python -m serial.tools.list_ports
```

On Linux, it will usually be `/dev/ttyACM0` or `/dev/ttyUSB0`. If unsure, compare the list before and after plugging in the UNO.

## Option 1: Direct USB GUI

Use this for initial testing and manual positioning:

```bash
python jog_control.py
```

1. Select the UNO's port and click **Connect**.
2. Wait for the connection to initialise. Opening USB may reset the UNO.
3. With the motor secured and shaft clear, switch on the 12 V supply.
4. Click **Set current position to zero**.
5. Start with a one-step jog, then try larger moves.

The GUI supports positive/negative jogs, absolute target positions, speed adjustment, software travel limits and **STOP/Esc**.

Default limits are **−100 to +100 steps**, with a maximum speed of **50 steps/s**. These are demonstration limits; adjust them to suit the mechanics.

**Do not connect the USB GUI while the ZeroMQ server is using the port.**

## Option 2: ZeroMQ server and clients

Use this when scan or acquisition processes need to control the motor:

```text
Python clients → ZeroMQ server → USB → UNO → driver → motor
```

The server is the sole owner of the USB connection. No firmware changes are needed when switching from the GUI.

### Start the server

Close the USB GUI and Arduino Serial Monitor, then run:

```bash
source .venv/bin/activate
python zmq_server.py --port /dev/ttyACM0
```

Replace the port with your UNO's actual port. The default endpoint is **`tcp://127.0.0.1:5555`**, accessible from the same computer only. Starting the server does not automatically zero or move the motor.

### Check the connection

In another terminal, enter this directory, activate the environment and run:

```bash
source .venv/bin/activate
python zmq_client.py
```

This prints motor status without commanding movement.

### Use from Python

Make this directory available on your Python import path, or run the example from this directory. Client processes need `pyzmq` installed in their environment.

```python
from zmq_client import MirrorClient

with MirrorClient() as motor:
    print(motor.status())

    # Motor power must be on, and this starting position checked.
    motor.zero()
    motor.set_limits(-20, 20)
    motor.set_speed(30)

    job = motor.move(5)
    motor.wait(job, timeout=10)

    # Add an optical settling delay or measurement here if needed.

    motor.wait(motor.goto(0), timeout=10)
```

`zero()` declares the current physical position to be zero. It does not move or home the motor. Do not call it automatically when an established reference should be preserved.

| Method | Action |
|---|---|
| `status()` | Read position, target, movement state, reference, limits and speed |
| `zero()` | Declare the current position zero |
| `move(steps)` | Start a relative move; return a job ID |
| `goto(position)` | Start an absolute move; return a job ID |
| `wait(job_id, timeout=30)` | Wait for that specific movement to finish |
| `set_speed(steps_per_second)` | Set maximum speed, 10–200 steps/s |
| `set_limits(lower, upper)` | Set inclusive travel limits |
| `stop()` | Cancel movement and stop further pulses |

Movement and configuration requests are rejected with `BUSY` while moving. Any client can request a stop.

Use a separate `MirrorClient` in each process or thread. Clients share position, limits and speed; there is no reservation of an entire scan sequence. Use one process to coordinate motion and configuration during a scan.

### Continuous scan example

With the server running, motor power on and the mechanics clear:

```bash
python scan_example.py --zero
```

This defines the starting position as zero and repeatedly scans **±3.6° of motor-shaft rotation**, with a half-second pause at each end.

For a different amplitude or a finite number of cycles:

```bash
python scan_example.py --amplitude 5.4 --dwell 1 --cycles 3
```

| Option | Meaning |
|---|---|
| `--amplitude` | Degrees on each side of the starting position; default 3.6 |
| `--speed` | Maximum speed in steps/s; default 10 |
| `--dwell` | Pause at each endpoint in seconds; default 0.5 |
| `--cycles` | Complete back-and-forth cycles; 0 means continuous |
| `--zero` | Explicitly declare the starting position zero |

Without `--zero`, the existing reference is preserved and the scan centres on the current commanded position.

Angles are rounded to the nearest **1.8° full step**. These are motor-shaft angles, not reflected-beam angles. The example respects existing travel limits and produces stepped motion with endpoint pauses, rather than a smooth sinusoidal scan.

Press **Ctrl+C** in the scan terminal to request STOP. The motor remains at its stopping position; it does not automatically return to the centre.

## Stopping and shutdown

1. Stop scan scripts so they cannot issue further moves.
2. For GUI control, press **STOP**, then disconnect or close the GUI.
3. For ZeroMQ control, press **Ctrl+C** in the server terminal. The server sends STOP before closing.
4. Wait for motion to stop.
5. Switch off the 12 V supply.
6. Disconnect USB if finished.

STOP stops pulses but does **not** remove holding current. The motor may remain energised until 12 V is switched off.

If software becomes unresponsive, switch off the 12 V supply. Disconnect both power sources before changing wiring or switches. Never disconnect motor wires while the driver is powered.

## Position and communication behaviour

- Position counts **commanded steps**, not measured encoder position.
- A reset loses the reference and restores default speed and limits. Establish zero again before moving.
- Missed steps, manual shaft movement or motor-power interruption can invalidate the position count.
- Do not command movement with motor power off: the UNO would still count pulses.
- Completion means the commanded pulse sequence finished. It does not verify physical position or optical settling.
- The GUI or server polls the UNO regularly. If communication stops during motion, the firmware stops pulses after approximately **1.5 seconds**.
- If a ZeroMQ client exits, an accepted finite move continues because the server remains connected. Closing a client does not automatically stop the motor.
- A request timeout may mean a command was accepted but its reply was lost. Do not blindly retry movement; check status or request STOP.
- A `wait()` timeout does not cancel movement.
- The server exits on a detected Arduino reset or serial communication failure rather than replaying commands automatically.
- The server retains the most recent 128 movement jobs in memory. Restarting it loses that history.

## JSON API

For clients not using `MirrorClient`, use a ZeroMQ REQ socket and send one JSON object per request:

```json
{"op": "status"}
```

Supported requests:

```text
{"op":"zero"}
{"op":"move","steps":5}
{"op":"goto","position":0}
{"op":"speed","value":30}
{"op":"limits","lower":-20,"upper":20}
{"op":"stop"}
{"op":"job","id":"<job_id>"}
```

Successful replies contain `ok: true` and state; movement replies also include `job_id`. Failures contain `ok: false` and `error`. This local-only API has no authentication: any local client can command the motor.

## Tests

These tests use simulated hardware and do not move the physical motor:

```bash
python -m unittest -v test_zmq.py test_scan_example.py
```

They cover ZeroMQ requests, movement completion, busy handling, stopping, input validation, scan endpoints and interruption handling.
