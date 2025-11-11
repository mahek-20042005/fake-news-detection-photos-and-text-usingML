
import os
import cv2
import random
import shutil

# --- Configuration ---
# How many frames to skip between captures.
# A value of 15 means we save 1 frame for every 15 frames in the video.
# This prevents having thousands of nearly identical images and saves space.
FRAME_INTERVAL = 15
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# The rest will be the test set

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'dataset', 'original_data')
OUTPUT_DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'dataset', 'final_dataset')


def extract_and_split(source_dir, output_dir):
    """
    Finds videos, extracts frames, and splits them into train/validation/test sets.
    """
    # Clean up any previous runs by deleting the old output directory
    if os.path.exists(output_dir):
        print(f"Removing existing directory: {output_dir}")
        shutil.rmtree(output_dir)

    # Create the full directory structure
    print("Creating new directory structure...")
    for split in ['train', 'validation', 'test']:
        for class_name in ['real', 'fake']:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

    # Process each class ('real' and 'fake')
    for class_name in ['real', 'fake']:
        class_source_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_source_path):
            print(f"Warning: Class directory not found at {class_source_path}")
            continue

        print(f"\nProcessing videos for class: '{class_name}'")

        # Get a list of all video files and shuffle them
        video_files = [f for f in os.listdir(class_source_path) if f.endswith(('.mp4', '.avi', '.mov'))]
        random.shuffle(video_files)

        if not video_files:
            print(f"  -> No video files found in {class_source_path}")
            continue

        # Split the LIST of videos into train, validation, and test sets
        total_videos = len(video_files)
        train_end = int(total_videos * TRAIN_RATIO)
        val_end = train_end + int(total_videos * VAL_RATIO)

        video_splits = {
            'train': video_files[:train_end],
            'validation': video_files[train_end:val_end],
            'test': video_files[val_end:]
        }

        # Now, process each video and save its frames to the correct destination
        for split_name, videos in video_splits.items():
            print(f"  -> Extracting frames for '{split_name}' set...")
            destination_dir = os.path.join(output_dir, split_name, class_name)

            for video_name in videos:
                video_path = os.path.join(class_source_path, video_name)

                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                saved_frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break  # End of video

                    if frame_count % FRAME_INTERVAL == 0:
                        # Construct a unique name for each frame
                        frame_filename = f"{os.path.splitext(video_name)[0]}frame{saved_frame_count}.png"
                        output_path = os.path.join(destination_dir, frame_filename)
                        cv2.imwrite(output_path, frame)
                        saved_frame_count += 1

                    frame_count += 1

                cap.release()
                print(f"    - Extracted {saved_frame_count} frames from {video_name}")

    print("\nFrame extraction and splitting completed successfully!")


if __name__ == '__main__':
    # Make sure the source data exists
    if not os.path.isdir(SOURCE_DATA_DIR):
        print(f"Error: Source directory not found at '{SOURCE_DATA_DIR}'")
        print(
            "Please make sure your 'original_data' folder with 'real' and 'fake' subfolders (containing videos) exists.")
    else:
        extract_and_split(SOURCE_DATA_DIR, OUTPUT_DATA_DIR)