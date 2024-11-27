#!/usr/bin/env python3

import os
import time
from ast import literal_eval
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

class TMPLMonitor:
    def __init__(self, panorama_id, gray_values, gray_indexes):
        # Monitor settings
        self.filename = 'tmpl.log'
        self.last_modified = 0
        self.last_state = None
        self.use_preview = False

        # Image processing settings
        self.panorama_id = panorama_id
        self.gray_values = gray_values
        self.gray_indexes = gray_indexes
        self.base_dir = f'./landscapes/{panorama_id}/sequences'
        self.directories = [os.path.join(self.base_dir, f"{i:02}_{panorama_id}_220") for i in range(1, 6)]
        self.output_dir = f'./landscapes/{panorama_id}'
        self.results_dir = './results'
        self.preview_results_dir = './preview_results'
        self.results_index = 0

        # Initialize caches
        self.image_cache = {}
        self.cache_ttl = 300
        self.executor = ThreadPoolExecutor(max_workers=5)

        # Verify directories
        print("\nChecking directories:")
        for dir in self.directories:
            print(f"{dir}: {'exists' if os.path.exists(dir) else 'MISSING'}")

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.preview_results_dir, exist_ok=True)

        # Initialize path cache
        print("\nInitializing BMP path cache...")
        self.path_cache = self.initialize_cache()
        print("Cache initialized")

        # Pre-load and resize masks
        print("\nPre-loading and resizing masks...")
        self.cached_masks = self.preload_masks()
        print("Masks loaded")

        print(f"Starting from index: {self.results_index}")

    def preload_masks(self, target_size=(3840, 1280)):
        """Pre-load and resize all masks"""
        masks = {}
        for gray_value in self.gray_values:
            mask_path = os.path.join(self.output_dir, f"{self.panorama_id}_{gray_value}.bmp")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
                masks[gray_value] = mask
        return masks

    def initialize_cache(self):
        """Cache file paths and their existence"""
        from pathlib import Path
        cache = [{} for _ in range(5)]
        total_files = 0

        for i, directory in enumerate(self.directories):
            dir_files = 0
            directory_path = Path(directory)
            if directory_path.exists():
                for filepath in directory_path.glob('*.bmp'):
                    try:
                        frame_num = int(filepath.stem.split('_')[-1])
                        cache[i][frame_num] = (str(filepath), True)
                        dir_files += 1
                    except ValueError as e:
                        print(f"Error parsing filename {filepath.name}: {e}")
                total_files += dir_files
                print(f"Sequence {i+1}: {dir_files} BMP files indexed")

        print(f"Total files indexed: {total_files}")
        return cache

    def get_last_state(self):
        """Read and return the last state from file with file locking"""
        max_retries = 3
        retry_delay = 0.1

        for attempt in range(max_retries):
            try:
                with open(self.filename, 'r') as f:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)

                    try:
                        lines = f.readlines()
                        if lines:
                            return literal_eval(lines[-1].strip())
                        return None
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            except (IOError, OSError) as e:
                if attempt < max_retries - 1:
                    print(f"Retry {attempt + 1}/{max_retries}: {e}")
                    time.sleep(retry_delay)
                    continue
                print(f"Error reading file after {max_retries} attempts: {e}")
                return None
            except Exception as e:
                print(f"Error parsing file content: {e}")
                return None

    def get_missing_states(self, prev_state, current_state):
        """Generate sequence of intermediate states"""
        if not prev_state or not current_state:
            return []

        max_steps = 0
        for i in range(len(prev_state)):
            if prev_state[i] != current_state[i]:
                steps = abs(current_state[i] - prev_state[i])
                max_steps = max(max_steps, steps)

        if max_steps <= 1:
            return []

        missing_states = []
        for step in range(1, max_steps):
            temp_state = list(prev_state)
            for i in range(len(prev_state)):
                if prev_state[i] != current_state[i]:
                    direction = 1 if current_state[i] > prev_state[i] else -1
                    if prev_state[i] != 0 or current_state[i] != 0:
                        temp_state[i] = prev_state[i] + (direction * step)
                        if direction > 0:
                            temp_state[i] = min(temp_state[i], current_state[i])
                        else:
                            temp_state[i] = max(temp_state[i], current_state[i])
            missing_states.append(temp_state)

        return missing_states

    def get_image(self, dir_index, frame_number):
        """Load image on demand with optimized caching"""
        try:
            filepath_info = self.path_cache[dir_index].get(frame_number)
            if filepath_info:
                filepath, exists = filepath_info
                if exists:
                    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    return image
                else:
                    print(f"File marked as non-existent: {filepath}")
            else:
                print(f"No path for frame {frame_number} in sequence {dir_index+1}")
        except Exception as e:
            print(f"Error loading frame {frame_number} from sequence {dir_index+1}: {e}")
        return None

    def save_file(self, path, image):
        """Helper method for saving files"""
        cv2.imwrite(str(path), image)

    def combine_colored_masks(self):
        """Combine pre-loaded masks using vectorized operations"""
        target_size = (3840, 1280)
        combined_image = np.full((target_size[1], target_size[0]), 255, dtype=np.uint8)

        for gray_value in self.gray_values:
            if gray_value == 220:
                mask_path = os.path.join(self.output_dir, f"{self.panorama_id}_{gray_value}.bmp")
                if os.path.exists(mask_path):
                    print(f"Loading current mask {gray_value}: {mask_path}")
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            else:
                mask = self.cached_masks.get(gray_value) # Use cache

            color_index = self.gray_indexes.get(gray_value)
            if mask is not None and color_index is not None:
                binary_mask = (mask > 0)
                combined_image[binary_mask] = color_index
                print(f"Applied mask {gray_value} -> index {color_index}")

        return combined_image

    def create_viridis_preview(self, mask):
        """
        Create a colored preview using the viridis colormap and save it as BMP.

        Args:
            mask (np.ndarray): Grayscale mask with indexed values.

        Returns:
            None
        """
        normalized_mask = (mask - mask.min()) / (mask.max() - mask.min())
        viridis_colored = plt.cm.viridis(normalized_mask)
        viridis_image = (viridis_colored[:, :, :3] * 255).astype(np.uint8)
        return cv2.cvtColor(viridis_image, cv2.COLOR_RGB2BGR)

    def process_state(self, state):
        """Process images based on current state"""
        if not state or not any(state):
            print("Invalid state - skipping processing")
            return

        missing_states = self.get_missing_states(self.last_state, state)
        if missing_states:
            print(f"Found missing states:")
            for missing_state in missing_states:
                print(f"Processing intermediate state: {missing_state}")
                self._process_single_state(missing_state)

        self._process_single_state(state)
        self.last_state = state

    def _process_single_state(self, state):
        """Process a single state"""
        load_start = time.time()
        overlays = []
        active_frames = []

        print(f"\nProcessing state: {state}")
        for seq_idx, frame_number in enumerate(state):
            sequence_name = f"{(seq_idx + 1):02}_{self.panorama_id}_220"
            if frame_number > 0:
                print(f"* {sequence_name} > loading frame {frame_number}")
                image = self.get_image(seq_idx, frame_number)
                if image is not None:
                    overlays.append(image)
                    active_frames.append(f"{seq_idx+1}:{frame_number}")
            else:
                print(f"* {sequence_name} > skipping")

        load_time = time.time() - load_start

        if not overlays:
            print("No valid frames to process. Skipping.")
            return

        try:
            merge_start = time.time()
            result = np.maximum.reduce(overlays)
            merge_time = time.time() - merge_start

            save_start = time.time()
            
            output_path = os.path.join(self.output_dir, f"{self.panorama_id}_220.bmp")
            self.save_file(output_path, result)
            print(f"Update mask 220: {output_path}")

            # Combined mask
            combined_image = self.combine_colored_masks()
        
            # Only increment index after we know we have valid data
            next_index = self.results_index + 1
        
            # Try to save all files - if any fails, don't increment the index
            try:
                # Save combined mask
                combined_output_path = os.path.join(self.results_dir, f"{next_index}.bmp")
                self.save_file(combined_output_path, combined_image)
                print(f"Saving combined mask: {combined_output_path}")

                # Create and save preview if enabled
                if self.use_preview:
                    preview_output_path = os.path.join(self.preview_results_dir, f"{next_index}.bmp")
                    preview_image = self.create_viridis_preview(combined_image)
                    self.save_file(preview_output_path, preview_image)
                    print(f"Adding preview: {preview_output_path}")

                # Only increment index if all saves were successful
                self.results_index = next_index
            
                save_time = time.time() - save_start

                print(f"\nActive frames: {', '.join(active_frames)}")
                print(f"Load time: {load_time:.3f}s")
                print(f"Merge time: {merge_time:.3f}s")
                print(f"Save time: {save_time:.3f}s")
                print(f"Total time: {load_time + merge_time + save_time:.3f}s")
                print(f"Successfully saved all files for index {self.results_index}")

            except Exception as e:
                print(f"Error saving results for index {next_index}: {e}")
                raise  # Re-raise the exception to be caught by outer try-except

        except Exception as e:
            print(f"Error in processing: {e}")
            print("Skipping this state due to error")

    def run(self):
        try:
            print("\nTMPL Monitor Started")
            print("Waiting for updates...")

            while True:
                try:
                    if not os.path.exists(self.filename):
                        time.sleep(0.1)
                        continue

                    current_modified = os.path.getmtime(self.filename)

                    if current_modified != self.last_modified:
                        time.sleep(0.05)

                        current_state = self.get_last_state()

                        print(f"\nPrevious state: {self.last_state}")
                        print(f"Current state: {current_state}")

                        if current_state and current_state != self.last_state:
                            print(f"\nPrevious state: {self.last_state}")
                            print(f"Current state: {current_state}")
                            self.process_state(current_state)
                            self.last_state = current_state
                        else:
                            print("State unchanged or invalid - skipping")

                        self.last_modified = current_modified

                    time.sleep(0.01)

                except KeyboardInterrupt:
                    print("\nMonitor stopped.")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(0.1)
        finally:
            self.executor.shutdown()

if __name__ == "__main__":
    panorama_id = "0145"
    gray_values = [250, 245, 220, 200, 195, 55, 38, 35]
    gray_indexes = {
        35: 1,
        38: 2,
        55: 4,
        195: 9,
        200: 10,
        220: 11,
        245: 12,
        250: 13
    }
    monitor = TMPLMonitor(panorama_id, gray_values, gray_indexes)
    monitor.run()
