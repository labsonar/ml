"""
Example Application
===================

This script provides an example application demonstrating the usage of the `example` module
from the `project_name` package. 

Application files should contain:
- Imports of necessary standard and project-specific modules.
- Function definitions for core functionalities.
- A `main` function to serve as the entry point for the application.
- Argument parsing to allow for configurable execution.

This example demonstrates the structure and usage of an application file.
"""

import os
import time
import argparse
import project_name.example as lib

def str_format_time(n_seconds: float) -> str:
    """
    Format a time duration given in seconds into a more readable string format.
    As an example of function only present in the application file.

    Args:
        n_seconds (float): The time duration in seconds.

    Returns:
        str: The formatted time duration as a string.
    """
    if n_seconds < 60:
        return f"{n_seconds:.2f} seconds"

    if n_seconds < 3600:
        minutes = n_seconds / 60
        return f"{minutes:.2f} minutes"

    hours = n_seconds / 3600
    return f"{hours:.2f} hours"

def main(sleep: float):
    """
    Main function to demonstrate the usage of the `sample_function` and sleep functionality.

    Args:
        sleep (float): The duration to sleep in seconds.
    """
    print(lib.sample_function())
    time.sleep(sleep)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f'{os.path.basename(__file__)} application')
    parser.add_argument('-s', '--sleep', type=float, default=0.5, help='Sleep duration in seconds')
    args = parser.parse_args()

    start_time = time.time()
    main(sleep=args.sleep)
    print(f"Elapsed time: {str_format_time(time.time() - start_time)}")
