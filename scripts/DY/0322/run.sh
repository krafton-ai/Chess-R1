#!/bin/bash

# List the scripts to run in order
scripts=("scripts/DY/0322/bs128n4_addengsysprompt_defaultkl.sh" "scripts/DY/0322/bs128n4_addengsysprompt_strongerkl.sh")

# Loop through each script and execute it
for script in "${scripts[@]}"; do
    echo "Executing $script..."
    bash "$script"
    
    # Check if the script executed successfully
    if [ $? -ne 0 ]; then
        echo "Error: $script failed. Stopping further execution."
        exit 1
    fi
done

echo "All scripts executed successfully."