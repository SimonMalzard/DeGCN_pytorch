#!/bin/bash

# Define the list of values for miss_amount
miss_values=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
config_file="/datadrive3/DeGCN_pytorch/work_dir/ntu120/cset/degcn/config.yaml"
work_dir="work_dir/ntu120/cset/degcn"

echo "Current working directory: $(pwd)"
if [ ! -f "$config_file" ]; then
    echo "Error: Config file does not exist at $config_file"
    exit 1
else
    echo "Config file exists at $config_file"
fi

for miss_value in "${miss_values[@]}"
do
    echo "Running script with random_miss_amount: $miss_value"
    
    # Update the YAML config file with the new miss_amount value using sed
    #sed -i "s/miss_amount: [0-9]\+/miss_amount: $miss_value/" "$config_file"
    sed -i "s/miss_amount: [0-9]*\.[0-9]\+/miss_amount: $miss_value/" "$config_file"

    # Check if sed updated the file correctly
    updated_value=$(grep 'random_miss_amount' "$config_file" | awk '{print $2}')
    echo "updated_value: $updated_value"
    if [ "$updated_value" != "$miss_value" ]; then
        echo "Error: Failed to update random_miss_amount to $miss_value in $config_file"
        exit 1
    else
        echo "Successfully updated random_miss_amount to $miss_value in $config_file"
    fi
    
    # Print the updated config file content
    echo "Updated config file content:"
    grep 'random_miss_amount' "$config_file"

    # Run the Python script with the specified command
    python3 main.py --config "$config_file" --work-dir "$work_dir" --phase test --save-score True --weights "$work_dir/20250429_211735/epoch_80_68080.pt" --device 0

    echo "Finished running script with random_miss_amount: $miss_value"
    echo "---------------------------------------------"
done