import os
import cv2
import json
import csv
import numpy as np

class FixationFrameExtractor:
    def __init__(self, patient_id, patient_dir):
        """
        video_path: Path to the video file (e.g., data/raw_videos/subject001/world.mp4)
        fixation_path: Path to fixation data file (CSV or JSON).
        output_dir: Root directory to save extracted frames (e.g., data/extracted_frames/)
        """
        self.patient_id = patient_id
        self.patient_dir = patient_dir
        self.video_path = os.path.join(patient_dir, "world.mp4")
        self.fixation_path = os.path.join(patient_dir, f"{patient_id}_fixations.csv")
        self.output_dir = patient_dir
        self.fixation_data = []
        self.video_capture = None

    def load_video(self):
        """Load the video into a capture object."""
        self.video_capture = cv2.VideoCapture(self.video_path)
        if not self.video_capture.isOpened():
            raise ValueError(f"Error opening video file: {self.video_path}")

    def load_fixation_data(self):
        """Load fixation data from a CSV or JSON file."""
        if self.fixation_path.endswith(".csv"):
            self.fixation_data = []
            with open(self.fixation_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    start = int(row["start_frame_index"])   # adapt to your CSV column names
                    end = int(row["end_frame_index"])
                    self.fixation_data.append({"start": start, "end": end})
        elif self.fixation_path.endswith(".json"):
            with open(self.fixation_path, "r") as f:
                self.fixation_data = json.load(f)
        else:
            raise ValueError("Unsupported fixation file format. Must be .csv or .json")

    def select_frames(self, start_frame, end_frame):
        """Select representative frames based on fixation length."""
        fixation_length = end_frame - start_frame + 1

        if fixation_length <= 3:
            return [start_frame + fixation_length // 2]
        elif fixation_length <= 9:
            return [start_frame, (start_frame + end_frame) // 2, end_frame]
        else:
            return [
                start_frame,
                start_frame + fixation_length // 4,
                (start_frame + end_frame) // 2,
                start_frame + (3 * fixation_length) // 4,
                end_frame
            ]

    def extract_and_save_frames(self):
        """Main function to extract frames and save them, and save metadata."""
        if not self.video_capture:
            self.load_video()

        if not self.fixation_data:
            self.load_fixation_data()

        os.makedirs(self.output_dir, exist_ok=True)

        # Use the parent folder of video (subject ID) as subject name
        subject_name = self.patient_id
        subject_output_dir = os.path.join(self.output_dir, "extracted_frames")
        os.makedirs(subject_output_dir, exist_ok=True)

        # Open a metadata CSV file
        metadata_path = os.path.join(subject_output_dir, "frames_metadata.csv")
        with open(metadata_path, mode="w", newline="") as metafile:
            writer = csv.writer(metafile)
            # Write header
            writer.writerow(["fixation_id", "frame_idx", "saved_filename"])

            for idx, fixation in enumerate(self.fixation_data):
                start_frame = fixation["start"]
                end_frame = fixation["end"]

                middle_frame = (start_frame + end_frame) // 2
                frame_indices = [middle_frame]

                for frame_idx in frame_indices:
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    success, frame = self.video_capture.read()

                    if success:
                        filename = f"fix{idx}.png"
                        save_path = os.path.join(subject_output_dir, filename)
                        cv2.imwrite(save_path, frame)

                        writer.writerow([idx, frame_idx, filename])
                    else:
                        print(f"Warning: Could not read frame {frame_idx} in fixation {idx}.")

        print(f"✅ Frames and metadata saved for subject '{subject_name}' at: {subject_output_dir}")

    def release(self):
        """Release the video capture object."""
        if self.video_capture:
            self.video_capture.release()