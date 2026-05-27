import os
import yaml
from ..utils.logger import logger

class GestureCalibrator:
    """An interactive command-line interface wizard to tune thresholds for gaze, face, and gestures."""
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def _save_config(self):
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(self.config, f)
        logger.info("Successfully updated config.yaml thresholds.")

    def run(self):
        print("=== AXON THRESHOLD TUNING WIZARD ===")
        print("This helper helps you tune thresholds to prevent false positive triggers.")
        print("1. Tune Gaze Wink Threshold")
        print("2. Tune Head Tilt Scroll Threshold")
        print("3. Tune Pinch Zoom / Click Threshold")
        print("4. Toggle GPU Acceleration")
        print("5. Exit and Save")
        
        while True:
            choice = input("\nEnter choice (1-5): ").strip()
            if choice == "1":
                curr = self.config["gaze"].get("ear_wink_threshold", 0.21)
                print(f"Current Wink EyeBlink/EAR Threshold: {curr}")
                val = input("Enter new value (e.g. 0.25 for less sensitive, 0.18 for more sensitive): ").strip()
                if val:
                    self.config["gaze"]["ear_wink_threshold"] = float(val)
                    print("Updated in memory.")
            elif choice == "2":
                curr = self.config["face"].get("head_tilt_scroll_threshold_deg", 15)
                print(f"Current Head Tilt Scroll Threshold: {curr} degrees")
                val = input("Enter new value (e.g. 20 for less sensitive scroll, 10 for more sensitive): ").strip()
                if val:
                    self.config["face"]["head_tilt_scroll_threshold_deg"] = int(val)
                    print("Updated in memory.")
            elif choice == "3":
                curr = self.config["gestures"].get("pinch_threshold", 0.05)
                print(f"Current Pinch Distance Threshold: {curr}")
                val = input("Enter new value (e.g. 0.03 for tighter pinch, 0.07 for looser pinch): ").strip()
                if val:
                    self.config["gestures"]["pinch_threshold"] = float(val)
                    print("Updated in memory.")
            elif choice == "4":
                curr = self.config["advanced"].get("use_gpu", True)
                print(f"GPU Acceleration: {'ENABLED' if curr else 'DISABLED'}")
                confirm = input("Toggle GPU? (y/n): ").strip().lower()
                if confirm == 'y':
                    self.config["advanced"]["use_gpu"] = not curr
                    print(f"GPU set to: {self.config['advanced']['use_gpu']}")
            elif choice == "5":
                self._save_config()
                print("Config saved. Exiting wizard.")
                break
            else:
                print("Invalid choice.")


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.yaml")
    wizard = GestureCalibrator(config_file)
    wizard.run()
