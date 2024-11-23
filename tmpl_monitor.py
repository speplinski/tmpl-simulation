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
        self.preview_path = os.path.join(self.output_dir, f"{panorama_id}_preview.bmp")

        # Initialize caches
        self.image_cache = {}
        self.cache_ttl = 300

        self.executor = ThreadPoolExecutor(max_workers=5)

        # Verify directories
        print("\nChecking directories:")
        for dir in self.directories:
            print(f"{dir}: {'exists' if os.path.exists(dir) else 'MISSING'}")

        mask_dir = self.output_dir
        print("\nFiles in output directory:")
        for filename in os.listdir(mask_dir):
            print(filename)

        # Initialize path cache
        print("\nInitializing BMP path cache...")
        self.path_cache = self.initialize_cache()
        print("Cache initialized")

        # Pre-load and resize masks
        print("\nPre-loading and resizing masks...")
        self.cached_masks = self.preload_masks()
        print("Masks loaded")

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
            mask = self.cached_masks.get(gray_value)
            color_index = self.gray_indexes.get(gray_value)
            if mask is not None and color_index is not None:
                binary_mask = (mask > 0)
                combined_image[binary_mask] = color_index

        # Create preview
        if self.use_preview:
            self.create_viridis_preview(combined_image, self.preview_path)

        return combined_image

    def create_viridis_preview(self, mask, output_path):
        """
        Create a colored preview using the viridis colormap and save it as BMP.

        Args:
            mask (np.ndarray): Grayscale mask with indexed values.
            output_path (str): Path to save the preview image.

        Returns:
            None
        """

        normalized_mask = (mask - mask.min()) / (mask.max() - mask.min())
        viridis_colored = plt.cm.viridis(normalized_mask)
        viridis_image = (viridis_colored[:, :, :3] * 255).astype(np.uint8)
        viridis_bgr = cv2.cvtColor(viridis_image, cv2.COLOR_RGB2BGR)

        # Save the image using OpenCV in BMP format
        cv2.imwrite(output_path, viridis_bgr)
        print("Preview saved")

    def process_state(self, state):
        """Process images based on current state"""
        if not state or not any(state):
           return

        # Collect valid overlays
        load_start = time.time()
        overlays = []
        active_frames = []

        for i, frame_number in enumerate(state):
           if frame_number > 0:
               # Try to get from cache first
               cache_key = f"{i}_{frame_number}"
               current_time = time.time()
    
               if cache_key in self.image_cache:
                   image, timestamp = self.image_cache[cache_key]
                   if current_time - timestamp < self.cache_ttl:
                       print(f"Cache hit: S{i+1}:F{frame_number}")
                       overlays.append(image)
                       active_frames.append(f"S{i+1}:F{frame_number}")
                       continue

               # Load from disk if not in cache or expired
               print(f"Loading from disk: S{i+1}:F{frame_number}")
               image = self.get_image(i, frame_number)
               if image is not None:
                   self.image_cache[cache_key] = (image, current_time)
                   overlays.append(image)
                   active_frames.append(f"S{i+1}:F{frame_number}")
                   print(f"Cached: S{i+1}:F{frame_number}")

        load_time = time.time() - load_start

        if not overlays:
           print("No valid frames to process. Skipping.")
           return

        try:
            # Merge overlays
            merge_start = time.time()
            result = np.maximum.reduce(overlays)
            merge_time = time.time() - merge_start

            # Save all results in parallel
            save_start = time.time()
            
            # Prepare all save operations
            save_tasks = []
            
            # Original result
            output_path = os.path.join(self.output_dir, f"{self.panorama_id}_220.bmp")
            save_tasks.append((output_path, result))
            
            # Resize and crop
            original_height, original_width = result.shape
            aspect_ratio = original_width / original_height
            target_width = 3840
            target_height = int(target_width / aspect_ratio)
            resized_image = cv2.resize(result, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            crop_top = (resized_image.shape[0] - 1280) // 2
            cropped_result = resized_image[crop_top:crop_top + 1280, :3840]
            
            # Combined mask
            combined_image = self.combine_colored_masks()
            combined_output_path = os.path.join(self.output_dir, f"{self.panorama_id}_mask.bmp")
            save_tasks.append((combined_output_path, combined_image))
            
            # Execute all saves in parallel
            futures = [
                self.executor.submit(self.save_file, path, img)
                for path, img in save_tasks
            ]
            
            # Wait for all saves to complete
            for future in futures:
                future.result()
                
            save_time = time.time() - save_start

            # Print performance metrics
            print(f"\nState: {state}")
            print(f"Active frames: {', '.join(active_frames)}")
            print(f"Load time: {load_time:.3f}s")
            print(f"Merge time: {merge_time:.3f}s")
            print(f"Save time: {save_time:.3f}s")
            print(f"Total time: {load_time + merge_time + save_time:.3f}s")

        except Exception as e:
            print(f"Error in processing: {e}")

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