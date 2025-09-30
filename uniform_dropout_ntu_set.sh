#!/bin/bash

# drop values for random dropout using dropout rates of 0% to 90% in increments of 10%. 
FPS_values=(3 5 10 15 30) 

config_file="/datadrive/GCN_LogsigRNN/config/ntu_sub/test_joint_xset.yaml"
work_dir="/datadrive/GCN_LogsigRNN/work_dir/ntu120/xset/gcn_logsigrnn"

echo "Current working directory: $(pwd)"

if [ ! -f "$config_file" ]; then
    echo "Error: Config file does not exist at $config_file"
    exit 1
else
    echo "Config file exists at $config_file"
fi

for FPS_value in "${FPS_values[@]}"
do
    echo "Running script with FPS: $FPS_value"
    # Update the YAML config file with the new FPS value using sed
    sed -i "s/FPS: [0-9]*/FPS: $FPS_value/" "$config_file"

    # Check if sed updated the file correctly
    updated_value=$(grep 'FPS' "$config_file" | awk '{print $2}')
    if [ "$updated_value" != "$FPS_value" ]; then
        echo "Error: Failed to update FPS to $FPS_value in $config_file"
        exit 1
    else
        echo "Successfully updated FPS to $FPS_value in $config_file"
    fi
    
    # Print the updated config file content
    echo "Updated config file content:"
    grep 'FPS' "$config_file"

    # Run the Python script with the specified command
    python3 main.py --config ./config/ntu_sub/test_joint_xset.yaml  --device 0 
    echo "Finished running script with FPS: $FPS_value"
    echo "---------------------------------------------"
done