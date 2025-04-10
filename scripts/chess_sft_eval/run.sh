#!/bin/bash

# Function to run a script and exit if it fails
run_script() {
  script_name="$1"
  echo "Running ${script_name}..."
  ./"${script_name}"
  if [ $? -ne 0 ]; then
    echo "Error: ${script_name} failed to execute correctly."
    exit 1
  fi
}

# Run each script consecutively
run_script "script1.sh"
run_script "script2.sh"
run_script "script3.sh"

echo "All scripts executed successfully!"