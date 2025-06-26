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
        prediction_df = extractor.parse_frame_predictions(fixation_id)
        print("prediction df: \n", prediction_df, "\n")
        print(prediction_df.columns)
    except Exception as e:
        print(f"\n❌ Error: {e}")

    try:
        fixation_df = extractor.parse_fixations_data()
        print("fixation df: \n", fixation_df, "\n")
        print(fixation_df.columns)
    except Exception as e:
        print(f"\n❌ Error: {e}")

    try:
        unity_df = extractor.parse_unity_log()
        print("unity df: \n", unity_df, "\n")
        print(unity_df.columns)
    except Exception as e:
        print(f"\n❌ Error: {e}")
if __name__ == "__main__":
    main()