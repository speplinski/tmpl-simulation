#!/usr/bin/env python3

import os
import time
from ast import literal_eval
import sys
import cv2
import numpy as np
from datetime import datetime
import json

class TMPLMonitor:
    def __init__(self):
        # Monitor settings
        self.filename = 'tmpl.log'
        self.last_modified = 0
        self.last_state = None
        
        # Performance tracking
        self.performance_log = {
            'png': {'load_times': [], 'total_times': []},
            'bmp': {'load_times': [], 'total_times': []}
        }
        
        # Create output directory
        self.output_dir = './output_frames'
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize caches for both formats
        print("Initializing caches for both formats...")
        self.caches = self.initialize_caches()
        
    def initialize_caches(self):
        formats = {
            'png': './frames_bw_resized_png',
            'bmp': './frames_bw_resized_bmp'
        }
        
        caches = {}
        
        for fmt, base_dir in formats.items():
            cache_start = time.time()
            print(f"\nInitializing {fmt.upper()} cache...")
            
            cache = [{} for _ in range(5)]
            files_count = 0
            
            directories = [os.path.join(base_dir, f"0145_220_{i:02}") for i in range(1, 6)]
            
            for i, directory in enumerate(directories):
                dir_start = time.time()
                if os.path.exists(directory):
                    for filename in os.listdir(directory):
                        if filename.endswith(f'.{fmt}'):
                            frame_num = int(filename.split('_')[-1].split('.')[0])
                            filepath = os.path.join(directory, filename)
                            cache[i][frame_num] = filepath
                            files_count += 1
                dir_time = time.time() - dir_start
                print(f"Directory {i+1} processed in {dir_time:.3f} seconds")
            
            cache_time = time.time() - cache_start
            print(f"Total {fmt.upper()} files cached: {files_count}")
            print(f"{fmt.upper()} cache initialized in {cache_time:.3f} seconds")
            
            caches[fmt] = {'directories': directories, 'cache': cache}
            
        return caches

    def get_last_state(self):
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

    def get_image(self, fmt, dir_index, frame_number):
        load_start = time.time()
        try:
            filepath = self.caches[fmt]['cache'][dir_index].get(frame_number)
            if filepath and os.path.exists(filepath):
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                load_time = time.time() - load_start
                self.performance_log[fmt]['load_times'].append(load_time)
                return img, load_time
        except Exception as e:
            print(f"Error loading {fmt} image for dir {dir_index}, frame {frame_number}: {e}")
        return None, 0

    def process_state(self, state):
        if not state:
            return

        print(f"\nProcessing state: {state}")
        
        # Process both formats
        for fmt in ['png', 'bmp']:
            process_start = time.time()
            print(f"\nProcessing {fmt.upper()}:")
            
            # Collect valid overlays
            overlays = []
            total_load_time = 0
            
            for i, frame_number in enumerate(state):
                if frame_number > 0:
                    overlay, load_time = self.get_image(fmt, i, frame_number)
                    if overlay is not None:
                        overlays.append(overlay)
                        total_load_time += load_time
            
            if not overlays:
                print(f"No valid {fmt} overlays found")
                continue

            try:
                # Process overlays
                merge_start = time.time()
                result = overlays[0].copy()
                for overlay in overlays[1:]:
                    result = np.maximum(result, overlay)
                merge_time = time.time() - merge_start

                # Save result
                save_start = time.time()
                output_path = os.path.join(self.output_dir, f"current_state_{fmt}.{fmt}")
                cv2.imwrite(output_path, result)
                save_time = time.time() - save_start

                total_time = time.time() - process_start
                self.performance_log[fmt]['total_times'].append(total_time)

                print(f"{fmt.upper()} Performance:")
                print(f"Total load time: {total_load_time:.3f} seconds")
                print(f"Merge time: {merge_time:.3f} seconds")
                print(f"Save time: {save_time:.3f} seconds")
                print(f"Total time: {total_time:.3f} seconds")

            except Exception as e:
                print(f"Error in {fmt} image processing: {e}")

    def save_performance_stats(self):
        stats = {fmt: {
            'average_load_time': np.mean(data['load_times']),
            'min_load_time': np.min(data['load_times']) if data['load_times'] else 0,
            'max_load_time': np.max(data['load_times']) if data['load_times'] else 0,
            'average_total_time': np.mean(data['total_times']),
            'min_total_time': np.min(data['total_times']) if data['total_times'] else 0,
            'max_total_time': np.max(data['total_times']) if data['total_times'] else 0,
            'operations_count': len(data['total_times'])
        } for fmt, data in self.performance_log.items()}
        
        with open('format_performance_comparison.json', 'w') as f:
            json.dump(stats, f, indent=4)
        
        print("\nPerformance Comparison:")
        for fmt, data in stats.items():
            print(f"\n{fmt.upper()}:")
            print(f"Average load time: {data['average_load_time']:.3f} seconds")
            print(f"Average total time: {data['average_total_time']:.3f} seconds")
            print(f"Operations count: {data['operations_count']}")

    def run(self):
        print("\nTMPL Monitor Started")
        print("Waiting for updates...")

        try:
            while True:
                if not os.path.exists(self.filename):
                    time.sleep(0.1)
                    continue

                current_modified = os.path.getmtime(self.filename)

                if current_modified != self.last_modified:
                    current_state = self.get_last_state()
                    
                    if current_state != self.last_state and current_state is not None:
                        print(f"\nNew state detected: {current_state}")
                        self.process_state(current_state)
                        self.last_state = current_state
                    
                    self.last_modified = current_modified

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nSaving performance statistics...")
            self.save_performance_stats()
            print("\nMonitor stopped.")

if __name__ == "__main__":
    monitor = TMPLMonitor()
    monitor.run()
