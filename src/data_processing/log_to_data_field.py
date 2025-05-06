import os
import re
import ast
import json
import random

LOG_DIR = './data/raw/logs/new/Field'
OUT_DIR = './data/processed/Field'
ATM_PRESSURE = 76.0  # cmHg

def remove_ansi(text):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

def parse_measurements(log_text):
    clean_text = remove_ansi(log_text)

    measurement_pattern = re.compile(
        r"=+ +Measurement #(?P<measurement_number>\d+) +={5,}.*?"
        r"Luminescence decay curve: *\[(?P<decay_curve>.*?)\].*?"
        r"Lifetime \[us\]: *(?P<lifetime>[0-9.eE+-]+).*?"
        r"Measurement timestamp \[ms\]: *(?P<timestamp>[0-9.eE+-]+).*?"
        r"Temperature \[C\]: *(?P<temperature>[0-9.eE+-]+).*?"
        r"g_x \[m/s\^2\]: *(?P<g_x>[0-9.eE+-.]+).*?"
        r"g_y \[m/s\^2\]: *(?P<g_y>[0-9.eE+-.]+).*?"
        r"g_z \[m/s\^2\]: *(?P<g_z>[0-9.eE+-.]+).*?"
        r"Film ID: *(?P<film_id>\d+).*?"
        r"Photobleaching exposure amount: *(?P<photobleaching>\d+).*?"
        r"# Pulses applied: *(?P<pulses_applied>\d+).*?"
        r"Predicted PtcO2 \[cmHg\]: *(?P<ptco2>[0-9.eE+-]+)",
        re.DOTALL
    )

    measurements = []
    film_id = None
    for match in measurement_pattern.finditer(clean_text):
        data = match.groupdict()
        data['measurement_number'] = int(data['measurement_number'])
        data['decay_curve'] = ast.literal_eval(f"[{data['decay_curve']}]")
        data['lifetime'] = float(data['lifetime'])
        data['timestamp'] = float(data['timestamp'])
        data['temperature'] = float(data['temperature'])
        data['g_x'] = float(data['g_x'])
        data['g_y'] = float(data['g_y'])
        data['g_z'] = float(data['g_z'])
        data['film_id'] = int(data['film_id'])
        data['photobleaching'] = int(data['photobleaching'])
        data['pulses_applied'] = int(data['pulses_applied'])
        data['ptco2'] = float(data['ptco2'])
        film_id = data['film_id']
        measurements.append(data)

    return measurements, film_id

def extract_o2_percent(filename):
    match = re.search(r'(\d+(?:\.\d+)?)%\.log$', filename)
    if match:
        return float(match.group(1))
    return None

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for filename in os.listdir(LOG_DIR):
        if filename.endswith('.log'):
            log_path = os.path.join(LOG_DIR, filename)
            print(f"Processing {log_path}...")

            o2_percent = extract_o2_percent(filename)
            if o2_percent is None:
                print(f"Warning: Could not extract O₂ % from filename: {filename}")
                continue

            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                log_text = f.read()

            measurements, film_id = parse_measurements(log_text)

            if film_id is None:
                print(f"Warning: No film ID found in {filename}")
                continue

            po2 = (o2_percent / 100.0) * ATM_PRESSURE

            output_data = {
                "oxygen_percentage": o2_percent,
                    "po2": po2,
                    "measurements": measurements
                }

            rand_id = random.randint(10, 99)
            out_filename = f"{film_id}-{o2_percent:.1f}-{rand_id}.json"
            out_path = os.path.join(OUT_DIR, out_filename)

            with open(out_path, 'w', encoding='utf-8') as f_out:
                json.dump(output_data, f_out, indent=2)

    print("Done!")

if __name__ == "__main__":
    main()
