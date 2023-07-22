#!/bin/bash

# Function to stop the SSH tunnel gracefully
stop_ssh_tunnel() {
  echo "Stopping SSH tunnel..."
  kill $ssh_pid
  echo "SSH tunnel stopped."
}

# Function to send log files
send_logs() {
  local foldername="$1"
  local remote_path="logs/${foldername:-training}"
  echo "Sending logs from $remote_path..."
  rsync --rsync-path=~/usr/bin/rsync -av bournez@slurm-ext:"$remote_path" ./logs/
  echo "Logs sent."
}

# Start the SSH tunnel in the background
ssh -L 16007:127.0.0.1:7860 bournez@slurm-ext &

# Store the SSH tunnel's process ID (PID)
ssh_pid=$!

# Set up the trap to stop the SSH tunnel when the script exits
trap stop_ssh_tunnel EXIT

# Wait for a few seconds (adjust as needed) to allow the SSH tunnel to establish
sleep 5

# Prompt the user for the foldername (optional)
read -p "Enter the foldername (Press Enter for 'training'): " foldername

# Call the function to send log files
send_logs "$foldername"