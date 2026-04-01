import json
import pandas as pd
import os

folder_path = "data/new_delhi_traffic_dataset/probe_counts/geojson"

all_rows = []

# Loop through all files
for file in os.listdir(folder_path):
    if file.endswith(".geojson"):

        file_path = os.path.join(folder_path, file)

        # Extract date from filename
        date = file.split("__")[1].split("_to_")[0]

        with open(file_path) as f:
            data = json.load(f)

        features = data["features"]

        for feature in features[1:]:
            props = feature["properties"]

            for entry in props["segmentProbeCounts"]:
                time_set = entry["timeSet"]
                probe_count = entry["probeCount"]

                hour = time_set - 2
                if hour < 0:
                    hour = 0

                date_time = f"{date} {hour:02d}:00:00"

                all_rows.append({
                    "date_time": date_time,
                    "traffic_volume": probe_count
                })

# Create DataFrame
df = pd.DataFrame(all_rows)

df["date_time"] = pd.to_datetime(df["date_time"])

# Aggregate across roads
df = df.groupby("date_time").agg({
    "traffic_volume": "sum"
}).reset_index()

df = df.sort_values("date_time")

# Save final dataset
df.to_csv("data/traffic.csv", index=False)

print("✅ Full dataset created!")
print(df.head())
print(df.tail())