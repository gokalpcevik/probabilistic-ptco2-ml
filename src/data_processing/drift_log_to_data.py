import os
import json
import random
import re

INPUT_DIR = "./data/raw/logs/drift"  
OUTPUT_DIR = "./data/processed/" 
OXYGEN_PERCENTAGE = 20.95
ATM_PRESSURE = 76.0  # cmHg
PO2 = ATM_PRESSURE * (OXYGEN_PERCENTAGE / 100)

# Dark data for subtraction
dark_data_tom003f_12 = [
    6771, 6760, 6758, 6755, 6757, 6756, 6756, 6755, 6754, 6755, 6756, 6754, 6755, 6755, 6754, 6753,
    6754, 6755, 6755, 6755, 6755, 6755, 6754, 6754, 6755, 6757, 6755, 6754, 6753, 6755, 6754, 6754,
    6753, 6754, 6754, 6754, 6756, 6756, 6755, 6756, 6754, 6754, 6753, 6754, 6754, 6754, 6755, 6754,
    6755, 6757, 6757, 6756, 6755, 6753, 6753, 6754, 6752, 6754, 6753, 6755, 6753, 6755, 6754, 6754,
    6755, 6754, 6755, 6755, 6755, 6754, 6755, 6753, 6754, 6753, 6754, 6754, 6755, 6753, 6756, 6755
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def subtract_dark_data(decay_curve):
    corrected = []
    for i in range(min(len(decay_curve), len(dark_data_tom003f_12))):
        value = decay_curve[i] - dark_data_tom003f_12[i]
        corrected.append(max(value, 0))
    return corrected

def parse_log_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    measurements = []
    measurement = None
    decay_curve = []
    pulses_applied = 0
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("Measurement"):
            if measurement:
                measurement["decay_curve"] = subtract_dark_data(decay_curve)
                measurement["pulses_applied"] = pulses_applied
                measurements.append(measurement)
                pulses_applied += 10

            measurement = {
                "measurement_number": len(measurements) + 1,
                "timestamp": 0.0,
                "temperature": 0.0,
                "g_x": 0.0,
                "g_y": 0.0,
                "g_z": 0.0,
                "film_id": 0,
                "photobleaching": 2515000,
                "pulses_applied": 0,
            }
            decay_curve = []

        elif line.startswith("Temperature"):
            temp_match = re.search(r"([-+]?\d*\.\d+)", line)
            if temp_match:
                measurement["temperature"] = float(temp_match.group(1))

        elif line.startswith("g_x"):
            gx_match = re.search(r"([-+]?\d*\.\d+)", line)
            if gx_match:
                measurement["g_x"] = float(gx_match.group(1))

        elif line.startswith("g_y"):
            gy_match = re.search(r"([-+]?\d*\.\d+)", line)
            if gy_match:
                measurement["g_y"] = float(gy_match.group(1))

        elif line.startswith("g_z"):
            gz_match = re.search(r"([-+]?\d*\.\d+)", line)
            if gz_match:
                measurement["g_z"] = float(gz_match.group(1))

        elif re.match(r"^-?\d+$", line):
            decay_curve.append(int(line))

    if measurement:
        measurement["decay_curve"] = subtract_dark_data(decay_curve)
        measurement["pulses_applied"] = pulses_applied
        measurements.append(measurement)

    return measurements

def process_all_logs():
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".log"):
            filepath = os.path.join(INPUT_DIR, filename)
            measurements = parse_log_file(filepath)

            output_data = {
                "oxygen_percentage": OXYGEN_PERCENTAGE,
                "po2": PO2,
                "measurements": measurements
            }

            film_id = random.randint(0, 10000)
            for m in output_data["measurements"]:
                m["film_id"] = film_id

            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            with open(output_path, 'w') as out_f:
                json.dump(output_data, out_f, indent=2)

            print(f"Processed {filename} -> {output_filename}")

if __name__ == "__main__":
    process_all_logs()
