#!/usr/bin/env python3

import os
import time
from ast import literal_eval
import sys
import cv2
import numpy as np

class TMPLMonitor:
    def __init__(self):
        # Monitor settings
        self.filename = 'tmpl.log'
        self.last_modified = 0
        self.last_state = None
        
        # Image processing settings
        self.base_dir = './sequence/0145_220'
        self.directories = [os.path.join(self.base_dir, f"0145_220_{i:02}") for i in range(1, 6)]
        self.output_dir = './results'
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize path cache
        print("\nInitializing BMP path cache...")
        self.path_cache = self.initialize_cache()
        print("Cache initialized")

    def initialize_cache(self):
        """Cache only file paths, not images"""
        cache = [{} for _ in range(5)]
        files_count = 0
        
        for i, directory in enumerate(self.directories):
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    if filename.endswith('.bmp'):
                        frame_num = int(filename.split('_')[-1].split('.')[0])
                        filepath = os.path.join(directory, filename)
                        cache[i][frame_num] = filepath
                        files_count += 1
                print(f"Directory {i+1}: {files_count} files indexed")
        
        return cache

    def get_last_state(self):
        """Read and return the last state from file"""
        try:
            with open(self.filename, 'r') as f:
                lines = f.readlines()
                if lines:
                    return literal_eval(lines[-1].strip())
                return None
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

    def get_image(self, dir_index, frame_number):
        """Load image on demand"""
        try:
            filepath = self.path_cache[dir_index].get(frame_number)
            if filepath and os.path.exists(filepath):
                return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            print(f"Error loading frame {frame_number} from dir {dir_index+1}: {e}")
        return None

    def process_state(self, state):
        """Process images based on current state"""
        if not state:
            return

        # Collect valid overlays
        load_start = time.time()
        overlays = []
        for i, frame_number in enumerate(state):
            if frame_number > 0:
                overlay = self.get_image(i, frame_number)
                if overlay is not None:
                    overlays.append(overlay)
        load_time = time.time() - load_start

        if not overlays:
            return

        try:
            # Merge overlays
            merge_start = time.time()
            result = overlays[0].copy()
            for overlay in overlays[1:]:
                result = np.maximum(result, overlay)
            merge_time = time.time() - merge_start

            # Save result
            save_start = time.time()
            output_path = os.path.join(self.output_dir, "current_state.bmp")
            cv2.imwrite(output_path, result)
            save_time = time.time() - save_start

            # Print performance metrics
            print(f"\nState: {state}")
            print(f"Load time: {load_time:.3f}s")
            print(f"Merge time: {merge_time:.3f}s")
            print(f"Save time: {save_time:.3f}s")
            print(f"Total time: {load_time + merge_time + save_time:.3f}s")

        except Exception as e:
            print(f"Error in processing: {e}")

    def run(self):
        print("\nTMPL Monitor Started")
        print("Waiting for updates...")

        while True:
            try:
                if not os.path.exists(self.filename):
                    time.sleep(0.1)
                    continue

                current_modified = os.path.getmtime(self.filename)

                if current_modified != self.last_modified:
                    current_state = self.get_last_state()
                    
                    if current_state != self.last_state and current_state is not None:
                        self.process_state(current_state)
                        self.last_state = current_state
                    
                    self.last_modified = current_modified

                time.sleep(0.01)

            except KeyboardInterrupt:
                print("\nMonitor stopped.")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(0.1)

if __name__ == "__main__":
    monitor = TMPLMonitor()
    monitor.run()
