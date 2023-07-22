# Fichier.sh qui permets de lancer automatiquement  le tensorboard et envoie le fichier
# Function to stop TensorBoard gracefully, MARCHE
stop_tensorboard() {
  echo "Stopping TensorBoard..."
  pkill -f "tensorboard --logdir=runs --host=localhost --port=7860"
  echo "TensorBoard stopped."
}

# Start the SSH tunnel in the background
ssh -L 16007:127.0.0.1:7860 bournez@slurm-ext &

# Store the SSH tunnel's process ID (PID)
ssh_pid=$!

# Wait for a few seconds (adjust as needed) to allow the SSH tunnel to establish
sleep 5

# Run TensorBoard in the background using nohup
nohup tensorboard --logdir=runs --host=localhost --port=7860 > tensorboard.log 2>&1 &

# Store the TensorBoard's process ID (PID)
tensorboard_pid=$!

# Set up the trap to stop TensorBoard when the script exits
trap stop_tensorboard EXIT

# Open a new browser to view TensorBoard, WUINDOWS only
start http://localhost:16007

# Keep the script running in the background to continue monitoring TensorBoard
wait $tensorboard_pid
