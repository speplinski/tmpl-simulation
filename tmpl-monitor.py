#!/usr/bin/env python3

import os
import time
from ast import literal_eval
import sys

def clear_screen():
    """Clear terminal screen for different OS"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_last_state(filename='tmpl.log'):
    """Read and return the last state from file"""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            if lines:
                return literal_eval(lines[-1].strip())
            return None
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def main():
    filename = 'tmpl.log'
    last_modified = 0
    last_state = None

    print("TMPL Log Monitor Started")
    print("Waiting for updates...")

    while True:
        try:
            # Check if file exists
            if not os.path.exists(filename):
                if last_state is not None:
                    clear_screen()
                    print(f"Waiting for {filename}...")
                    last_state = None
                time.sleep(1)
                continue

            # Get file's last modification time
            current_modified = os.path.getmtime(filename)

            # If file was modified
            if current_modified != last_modified:
                current_state = get_last_state(filename)
                
                # If state changed and is valid
                if current_state != last_state and current_state is not None:
                    clear_screen()
                    print(f"Current State: {current_state}")
                    last_state = current_state
                
                last_modified = current_modified

            time.sleep(0.1)  # Small delay to prevent high CPU usage

        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
