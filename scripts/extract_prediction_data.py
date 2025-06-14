from utils.data_utils import DataExtraction

def clean_path(p):
    """Clean user input: strip spaces and surrounding quotes."""
    return p.strip().strip('"').strip("'")

def main():
    print("\n🎯 Prediction Data Extractor 🎯\n")

    patient_id = input("🔹 Enter patient ID (e.g., AN755):\n> ").strip()
    patient_dir = clean_path(input("🔹 Enter path to patient folder:\n> "))
    fixation_id = input("🔹 Enter fixation ID (e.g., fix101):\n> ").strip()

    extractor = DataExtraction(patient_id, patient_dir)

    try:
        fixation_df = extractor.extract_frame_predictions(fixation_id)
        print("\n✅ Prediction data extracted!\n")
        print(fixation_df)
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()