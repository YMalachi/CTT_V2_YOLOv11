import os
from utils.video_utils import FixationFrameExtractor

def clean_path(p):
    """Clean user input: strip spaces and surrounding quotes."""
    return p.strip().strip('"').strip("'")

def get_valid_directory(prompt_message):
    """Prompt user until they provide a valid existing directory path (or create it)."""
    while True:
        path = clean_path(input(prompt_message))
        if os.path.isdir(path):
            return path
        else:
            create = input(f"⚠️ Directory does not exist. Create it? (y/n): ").strip().lower()
            if create == 'y':
                os.makedirs(path, exist_ok=True)
                print(f"✅ Directory created: {path}")
                return path
            else:
                print("🔁 Let's try again.\n")

def main():
    print("\n🎯 Welcome to the Frame Extraction Tool 🎯\n")

    patient_id = input("🔹 Enter patient ID (e.g., AN755):\n> ").strip()
    patient_dir = get_valid_directory("🔹 Enter path to the patient directory (e.g., data/processed/AN755/):\n> ")

    extractor = FixationFrameExtractor(
        patient_id=patient_id,
        patient_dir=patient_dir
    )

    try:
        extractor.extract_and_save_frames()

    finally:
        extractor.release()

if __name__ == "__main__":
    main()
