#!/usr/bin/env python3

import os
import time
from ast import literal_eval
import sys
import cv2
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt

class TMPLMonitor:
    def __init__(self, panorama_id, gray_values, gray_indexes):
        # Monitor settings
        self.filename = 'tmpl.log'
        self.last_modified = 0
        self.last_state = None
        
        # Image processing settings
        self.panorama_id = panorama_id
        self.gray_values = gray_values
        self.gray_indexes = gray_indexes
        self.base_dir = f'./landscapes/{panorama_id}/sequences'
        self.directories = [os.path.join(self.base_dir, f"{i:02}_{panorama_id}_220") for i in range(1, 6)]
        self.output_dir = f'./landscapes/{panorama_id}'

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

    def initialize_cache(self):
        """Cache only file paths, not images"""
        cache = [{} for _ in range(5)]
        total_files = 0
        
        for i, directory in enumerate(self.directories):
            dir_files = 0
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    if filename.endswith('.bmp'):
                        try:
                            frame_num = int(filename.split('_')[-1].split('.')[0])
                            filepath = os.path.join(directory, filename)
                            cache[i][frame_num] = filepath
                            dir_files += 1
                        except ValueError as e:
                            print(f"Error parsing filename {filename}: {e}")
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
        """Load image on demand"""
        try:
            filepath = self.path_cache[dir_index].get(frame_number)
            if filepath and os.path.exists(filepath):
                return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            else:
                if filepath:
                    print(f"File not found: {filepath}")
                else:
                    print(f"No path for frame {frame_number} in sequence {dir_index+1}")
        except Exception as e:
            print(f"Error loading frame {frame_number} from sequence {dir_index+1}: {e}")
        return None

    def process_state(self, state):
        """Process images based on current state"""
        if not state:
            return

        # Collect valid overlays
        load_start = time.time()
        overlays = []
        active_frames = []
    
        for i, frame_number in enumerate(state):
            if frame_number > 0:
                overlay = self.get_image(i, frame_number)
                if overlay is not None:
                    overlays.append(overlay)
                    active_frames.append(f"S{i+1}:F{frame_number}")
                else:
                    print(f"Failed to load sequence {i+1}, frame {frame_number}")
    
        load_time = time.time() - load_start

        if not overlays:
            print("No valid overlays found for state")
            return

        try:
            # Merge overlays
            merge_start = time.time()
            result = overlays[0].copy()
            for overlay in overlays[1:]:
                result = np.maximum(result, overlay)
            merge_time = time.time() - merge_start

            # Resize and crop the merged result (optional, only if needed later)
            resize_crop_start = time.time()

            # Get the original size of the image
            original_height, original_width = result.shape
            aspect_ratio = original_width / original_height

            # Calculate the new dimensions while maintaining the aspect ratio
            target_width = 3840
            target_height = int(target_width / aspect_ratio)

            # Resize the image using OpenCV
            resized_image = cv2.resize(result, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

            # Crop the resized image to 3840x1280 dimensions, centered vertically
            crop_top = (resized_image.shape[0] - 1280) // 2
            cropped_image = resized_image[crop_top:crop_top + 1280, :3840]
            resize_crop_time = time.time() - resize_crop_start

            # Combine all masks
            combine_start = time.time()
            mask_files = [os.path.join(self.output_dir, f"{self.panorama_id}_{value}.bmp") for value in self.gray_values]
            combined_image = combine_colored_masks(
                mask_files, self.gray_values, self.gray_indexes, self.panorama_id, self.output_dir
            )

            # Resize and crop the combined image
            combined_resized_image = cv2.resize(combined_image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            combined_cropped_image = combined_resized_image[
                (combined_resized_image.shape[0] - 1280) // 2:(combined_resized_image.shape[0] - 1280) // 2 + 1280, :3840
            ]

            combined_output_path = os.path.join(self.output_dir, f"{self.panorama_id}_mask.bmp")
            cv2.imwrite(combined_output_path, combined_cropped_image)
            combine_time = time.time() - combine_start

            # Print performance metrics
            print(f"\nState: {state}")
            print(f"Active frames: {', '.join(active_frames)}")
            print(f"Load time: {load_time:.3f}s")
            print(f"Merge time: {merge_time:.3f}s")
            print(f"Resize and crop time: {resize_crop_time:.3f}s")
            print(f"Mask combine time: {combine_time:.3f}s")
            print(f"Total time: {load_time + merge_time + resize_crop_time + combine_time:.3f}s")

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

def load_mask(filepath):
    return np.array(Image.open(filepath)).astype(bool)

def combine_colored_masks(mask_files, gray_values, gray_indexes, panorama_id, output_dir, target_size=(3840, 1280)):
    """
    Combine masks and scale the final image to the target size.

    Args:
        mask_files (list): List of file paths to individual masks.
        gray_values (list): List of grayscale values corresponding to masks.
        gray_indexes (dict): Mapping of grayscale values to indexes.
        panorama_id (str): Identifier for the panorama.
        output_dir (str): Directory where results will be saved.
        target_size (tuple): Target size for the output image (width, height).

    Returns:
        np.ndarray: Resized combined mask.
    """
    # Pre-load all masks into a list (all masks have the same dimensions)
    masks = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in mask_files]

    # Initialize combined image with white (255)
    combined_image = np.full_like(masks[0], 255, dtype=np.uint8)

    # Apply masks in descending order of grayscale values
    for gray_value, mask in zip(gray_values, masks):
        color_index = gray_indexes.get(gray_value, None)
        if color_index is not None:
            combined_image[mask > 0] = color_index

    # Resize the combined mask to the target size
    target_width, target_height = target_size
    resized_combined_image = cv2.resize(combined_image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    # Save a colored preview using viridis if requested
    #preview_path = os.path.join(output_dir, f"{panorama_id}_preview.bmp")
    #create_viridis_preview(resized_combined_image, preview_path)

    return resized_combined_image

def create_viridis_preview(mask, output_path):
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
    print(f"Viridis preview saved to {output_path}")

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