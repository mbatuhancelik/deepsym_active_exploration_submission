#!/usr/bin/zsh

# Number of processes to run
num_trainers=3

# Your command to run (replace with your actual command)

# Array to store process PIDs
pids=()
rm -rf council
mkdir council
echo "begin training phase for $2"
# Start processes in the background
for ((i=1; i<=num_trainers; i++)); do
  CUDA_VISIBLE_DEVICES=0 nohup python train.py -c $1 -cid $((i))"_"$2 &
  pids+=($!)  # Store the PID of the background process
done

for ((i=1; i<=num_trainers; i++)); do
   CUDA_VISIBLE_DEVICES=1 nohup python train.py -c $1 -cid $((i+num_trainers))"_"$2  &
  pids+=($!)  # Store the PID of the background process
done

# Wait for all processes to finish
for pid in "${pids[@]}"; do
  wait $pid
done


echo "All processes have finished."
