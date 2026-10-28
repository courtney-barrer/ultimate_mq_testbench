#!/bin/bash

echo "=== PHASE D: CLEAN SHUTDOWN SEQUENCE ==="
echo "Halting the sensor acquisition loop..."
dcam0_client dcam0 stop

echo "Releasing Hamamatsu DCAM API hardware locks..."
dcam0_client dcam0 close

echo "Killing the background daemon and tmux session..."
dcam0_stop

echo "=== SHUTDOWN COMPLETE ==="