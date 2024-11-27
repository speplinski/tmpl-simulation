#!/usr/bin/env python3

import os
import time
from ast import literal_eval
import cv2
import numpy as np
from pathlib import Path
import mmap
import psutil
from datetime import datetime

def get_memory_usage():
    """Returns memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info().rss
    return memory_info / 1024 / 1024

def print_memory_status():
    """Prints current memory status"""
    memory_mb = get_memory_usage()
    total_memory = psutil.virtual_memory().total / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")
    print(f"Total system memory: {total_memory:.1f} MB")
    print(f"Memory usage percentage: {(memory_mb/total_memory)*100:.1f}%")

class TMPLMonitor:
    def __init__(self, panorama_id, gray_values, gray_indexes):
        self.filename = 'tmpl.log'
        self.last_modified = 0
        self.last_state = None
        
        # Basic configuration
        self.panorama_id = panorama_id
        self.gray_values = gray_values
        self.gray_indexes = gray_indexes
        
        # Paths
        self.base_dir = Path(f'./landscapes/{panorama_id}/sequences')
        self.output_dir = Path(f'./landscapes/{panorama_id}')
        self.results_dir = Path('./results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Memory caches
        self.mask_cache = {}
        self.sequence_frames = {}
        self.current_mask_220 = None
        self.results_index = 0
        
        # Initialize system
        self._initialize_system()
        
    def _initialize_system(self):
        """Initialize system with memory monitoring"""
        print("Initializing system...")
        print("\nInitial memory status:")
        print_memory_status()
        
        # Load static masks
        print("\nLoading static masks...")
        for gray_value in self.gray_values:
            if gray_value != 220:
                mask_path = self.output_dir / f"{self.panorama_id}_{gray_value}.bmp"
                if mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        self.mask_cache[gray_value] = mask
                        print(f"Loaded mask {gray_value}")
        
        print("\nMemory status after loading masks:")
        print_memory_status()
        
        # Load sequence frames
        print("\nLoading sequence frames...")
        total_frames = 0
        for seq_num in range(1, 6):
            seq_dir = self.base_dir / f"{seq_num:02}_{self.panorama_id}_220"
            if seq_dir.exists():
                self.sequence_frames[seq_num] = {}
                seq_frames = 0
                for frame_path in seq_dir.glob('*.bmp'):
                    try:
                        frame_num = int(frame_path.stem.split('_')[-1])
                        frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
                        if frame is not None:
                            self.sequence_frames[seq_num][frame_num] = frame
                            seq_frames += 1
                    except ValueError:
                        continue
                total_frames += seq_frames
                print(f"Sequence {seq_num}: {seq_frames} frames loaded")
                print_memory_status()
        
        print(f"\nTotal frames loaded: {total_frames}")
        print("\nFinal memory status:")
        print_memory_status()

    def get_last_state(self):
        """Read last state from file using mmap"""
        try:
            with open(self.filename, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    last_line = mm.readline()
                    while True:
                        next_line = mm.readline()
                        if not next_line:
                            break
                        last_line = next_line
                    return literal_eval(last_line.decode().strip())
        except Exception:
            return None

    def update_mask_220(self, state):
        """Update mask 220 in memory"""
        active_frames = []
        
        for seq_idx, frame_number in enumerate(state, 1):
            if frame_number > 0:
                frame = self.sequence_frames.get(seq_idx, {}).get(frame_number)
                if frame is not None:
                    active_frames.append(frame)
        
        if not active_frames:
            self.current_mask_220 = None
            return False
            
        self.current_mask_220 = np.maximum.reduce(active_frames)
        return True

    def combine_masks(self):
        """Combine all masks in memory"""
        first_mask = next(iter(self.mask_cache.values()))
        combined = np.full(first_mask.shape, 255, dtype=np.uint8)
        
        for gray_value, mask in self.mask_cache.items():
            if gray_value in self.gray_indexes:
                combined[mask > 0] = self.gray_indexes[gray_value]
        
        if self.current_mask_220 is not None and 220 in self.gray_indexes:
            combined[self.current_mask_220 > 0] = self.gray_indexes[220]
        
        return combined

    def process_state(self, state):
        """Process state - all operations in memory"""
        if not state or not any(state):
            return

        process_start = time.time()
        current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"\nStarting processing at: {current_time}")
        print("Memory before processing:")
        print_memory_status()

        try:
            # Update mask 220 in memory
            mask_220_start = time.time()
            print("Updating mask 220 in memory...")
            if not self.update_mask_220(state):
                print("No valid frames to process")
                return
            mask_220_time = time.time() - mask_220_start
            print(f"Mask 220 update time: {mask_220_time:.3f}s")
            
            # Create combined mask
            combine_start = time.time()
            print("Combining masks...")
            combined_mask = self.combine_masks()
            combine_time = time.time() - combine_start
            print(f"Masks combination time: {combine_time:.3f}s")
            
            # Save result
            save_start = time.time()
            next_index = self.results_index + 1
            result_path = self.results_dir / f"{next_index}.bmp"
            cv2.imwrite(str(result_path), combined_mask)
            save_time = time.time() - save_start
            
            total_time = time.time() - process_start
            current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            self.results_index = next_index
            
            print(f"\nProcessing completed at: {current_time}")
            print(f"Result saved as: {result_path}")
            print(f"\nTiming breakdown:")
            print(f"- Mask 220 update: {mask_220_time:.3f}s")
            print(f"- Masks combination: {combine_time:.3f}s")
            print(f"- Save result: {save_time:.3f}s")
            print(f"Total processing time: {total_time:.3f}s")
            
            print("\nMemory after processing:")
            print_memory_status()
            
        except Exception as e:
            print(f"Error in processing: {e}")

    def run(self):
        """Main program loop"""
        print("\nTMPL Monitor Started")
        print("Waiting for updates...")
        
        try:
            while True:
                try:
                    if not os.path.exists(self.filename):
                        time.sleep(0.1)
                        continue

                    current_modified = os.path.getmtime(self.filename)
                    if current_modified != self.last_modified:
                        current_state = self.get_last_state()
                        if current_state and current_state != self.last_state:
                            print(f"\nProcessing state: {current_state}")
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
                    
        except KeyboardInterrupt:
            print("\nShutting down...")

if __name__ == "__main__":
    panorama_id = "0145"
    gray_values = [250, 245, 220, 200, 195, 55, 38, 35]
    gray_indexes = {
        35: 1, 38: 2, 55: 4, 195: 9,
        200: 10, 220: 11, 245: 12, 250: 13
    }
    
    print("\nStarting memory status:")
    print_memory_status()
    
    monitor = TMPLMonitor(panorama_id, gray_values, gray_indexes)
    monitor.run()