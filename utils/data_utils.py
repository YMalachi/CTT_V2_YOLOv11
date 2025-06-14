import os
import pandas as pd

# dictionary that contains the class number, and it's corresponding ball:
class_ball_dict = {
 1: 'ball_01_pink',   2: 'ball_02_pink',   3: 'ball_02_yellow',  4: 'ball_03_pink',   5: 'ball_03_yellow',
 6: 'ball_04_pink',   7: 'ball_04_yellow', 8: 'ball_05_pink',    9: 'ball_05_yellow',10: 'ball_06_pink',
11: 'ball_06_yellow',12: 'ball_07_pink', 13: 'ball_07_yellow', 14: 'ball_08_pink',  15: 'ball_08_yellow',
16: 'ball_09_pink',  17: 'ball_09_yellow',18: 'ball_10_pink',  19: 'ball_10_yellow',20: 'ball_11_pink',
21: 'ball_11_yellow',22: 'ball_12_pink', 23: 'ball_12_yellow',24: 'ball_13_pink',  25: 'ball_13_yellow',
26: 'ball_14_pink',  27: 'ball_14_yellow',28: 'ball_15_pink',  29: 'ball_15_yellow',30: 'ball_16_pink',
31: 'ball_16_yellow',32: 'ball_17_pink', 33: 'ball_17_yellow',34: 'ball_18_pink',  35: 'ball_18_yellow',
36: 'ball_19_pink',  37: 'ball_19_yellow',38: 'ball_20_pink',  39: 'ball_20_yellow',40: 'ball_21_pink',
41: 'ball_21_yellow',42: 'ball_22_pink', 43: 'ball_22_yellow',44: 'ball_23_pink',  45: 'ball_23_yellow',
46: 'ball_24_pink',  47: 'ball_24_yellow',48: 'ball_25_pink',  49: 'ball_25_yellow',50: 'ball_unknown', 51: 'user_cursor'}

class DataExtraction:
    def __init__(self, patient_id, patient_dir):
        """
        :param patient_id: Patient ID (e.g. AN755)
        :param patient_dir: A path to the patient folder.
        """
        self.patient_id = patient_id
        self.patient_dir = patient_dir
        self.predictions_dir = os.path.join(self.patient_dir, "predictions") # path to YOLOv11 prediction data directory (contain a .txt file for each fixation)
        self.fixation_data = os.path.join(patient_dir, f"{patient_id}_fixations.csv") # path to Pupil Labs' exported fixation data.

    def _find_file(self, fixation_id): # INTERNAL FUNCTION!
        """
        Finds fixation file in prediction_dir and returns the file path.
        :param fixation_id: Fixation ID (e.g. fix1)
        :return: Fixation path.
        """
        pred_path = None
        for filename in os.listdir(self.predictions_dir):
            if filename.startswith(fixation_id):
                pred_path = os.path.join(self.predictions_dir, filename)
                break

        if pred_path is None:
            raise FileNotFoundError(
                f"No prediction file found for fixation ID '{fixation_id}' in {self.predictions_dir}")
        return pred_path

    def extract_frame_predictions(self, fixation_id:str) -> pd.DataFrame:
        """
        Extract prediction data from an individual fixation frames. NOTE: this function ONLY extracts data predicted
        by the model, and not Pupil Labs or Unity's data! use 'extract_fixation_data' function to extract PL's data.
        :param fixation_id: Fixation number (e.g. fix101)
        :return: A dataframe with columns: ball, ball_id, center, size. A row for each ball detected in the frame.
        """
        # Initiate dataframe to store the data
        columns = ["ball", "ball_id", "center", "size"]
        fixation_df = pd.DataFrame(columns=columns)

        # Find frame prediction txt file path
        pred_path = self._find_file(fixation_id=fixation_id)

        # Read the .txt file
        # Expected line format in YOLOv11-style .txt file:
        # <class_id> <x_center> <y_center> <width> <height>   (all normalized)
        with open(pred_path, 'r') as f:
            for line in f:
                parts = line.strip().split()  # split by space
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # adding data to the dataframe:
                new_row = {
                    "ball" : class_ball_dict[class_id],
                    "ball_id": class_id,
                    "center" : (x_center, y_center), # a column for (x,y) ball's center coordinates.
                    "size" : width * height # using the area of the bounding box to estimate ball's distance later
                }

                fixation_df = pd.concat([fixation_df, pd.DataFrame([new_row])], ignore_index=True)

        return fixation_df

    def extract_fixations_data(self):
        """
        Extracts Pupil Labs fixation metadata from the patient's CSV file.

        This function loads the Pupil Labs exported fixation data and returns a filtered DataFrame
        containing only the most relevant columns for downstream analysis. The selected columns include:
        - fixation ID
        - start time of fixation
        - start and end frame indices
        - normalized gaze position (X and Y)
        - fixation duration

        :return: pd.DataFrame with columns:
                 ["id", "start_times", "start_frame_index", "end_frame_index", "norm_pos_x", "norm_pos_y", "duration"]
        """
        df = pd.read_csv(self.fixation_data)
        selected_columns = ["id", "start_times", "start_frame_index", "end_frame_index", "norm_pos_x", "norm_pos_y",
                            "duration"]
        filtered_df = df[selected_columns]
        return filtered_df





