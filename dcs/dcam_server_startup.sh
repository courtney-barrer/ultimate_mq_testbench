#!/bin/bash

echo "=== PHASE A: SYSTEM INITIALIZATION ==="
echo "Mounting high-speed RAM disk..."
sudo mount -t tmpfs -o rw,nosuid,nodev tmpfs /milk/shm
sudo chown -R $USER: /milk/shm

echo "=== PHASE B: STARTING CAMERA PIPELINE ==="
echo "Launching background server..."
dcam0_start

# Wait 2 seconds to ensure the TCP server has opened port 18010
sleep 2 

echo "Connecting to physical sensor..."
dcam0_client dcam0 open

echo "Starting continuous acquisition loop..."
dcam0_client dcam0 run

echo "=== SYSTEM BOOT COMPLETE ==="
echo "Shared memory stream is active. You can now connect your GUI or CACAO loops."