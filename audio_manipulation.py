#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 19:36:29 2025

@author: kissmarcell
"""

import os
import pandas as pd
import subprocess
import yt_dlp

# Example DataFrame

stamps=pd.read_excel("Data/timestamps_raw.xlsx",sheet_name="Timestamps")
links=stamps[stamps['order']==1]
links=links[['date','link']].sort_values(by='date')

FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"

# Output directory
output_dir = "fomc_audio"
os.makedirs(output_dir, exist_ok=True)

# Function to download and convert audio
def download_audio(date, url):
    # Temporary audio file
    temp_audio = os.path.join(output_dir, f"{date}")
    output_filename = os.path.join(output_dir, f"{date}.wav")  # Final WAV file

    # yt-dlp options (downloads best audio only)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_audio,  # Save temp file
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',  # First download in M4A format
        }],
        'ffmpeg_location': "/opt/homebrew/bin/ffmpeg",  # Adjust if needed
        'quiet': True
    }

    # Download audio
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    temp_audio = os.path.join(output_dir, f"{date}.m4a")
    # Convert to WAV (16kHz, mono, 64kbps)
    command_convert = [
        FFMPEG_PATH , "-i", temp_audio, "-ar", "16000", "-ac", "1", "-b:a", "64k", output_filename
    ]
    subprocess.run(command_convert, check=True)

    # Remove temporary file
    os.remove(temp_audio)
    print(f"✅ Processed: {output_filename}")

# Process each row in DataFrame
for _, row in links.iterrows():
    download_audio(row['date'], row['link'])

print("✅ All audio files processed successfully!")


# Identify the introductory statements (first entry for each conference)
intro_statements = stamps.groupby("date").first().reset_index()

# List to store new timestamp entries
new_timestamps = []

for _, row in stamps.iterrows():
    if row["start"] == intro_statements.loc[intro_statements["date"] == row["date"], "start"].values[0]:
        # Introductory statement found
        total_duration = row["length"]
        part_duration = total_duration // 3  # Split into 3 equal parts

        # Generate three new timestamp segments
        new_timestamps.append({"date": row["date"], "start_sec": row["start_sec"], "end_sec": row["start_sec"] + part_duration, "length": part_duration})
        new_timestamps.append({"date": row["date"], "start_sec": row["start_sec"] + part_duration, "end_sec": row["start_sec"] + 2 * part_duration, "length": part_duration})
        new_timestamps.append({"date": row["date"], "start_sec": row["start_sec"] + 2 * part_duration, "end_sec": row["end_sec"], "length": row["end_sec"] - (row["start_sec"] + 2 * part_duration)})
    else:
        # Keep other timestamps unchanged
        new_timestamps.append(row.to_dict())

# Convert to DataFrame
updated_timestamps = pd.DataFrame(new_timestamps)
updated_timestamps=updated_timestamps[updated_timestamps['length']>4]
updated_timestamps=updated_timestamps.drop(columns=['link','start','end'])
updated_timestamps["order"] = updated_timestamps.groupby("date").cumcount() + 1
updated_timestamps["id"] = updated_timestamps["date"].astype(str) + "_" + updated_timestamps["order"].apply(lambda x: f"{x:02}")
updated_timestamps=updated_timestamps.sort_values(by=['date','order'])

# Save back to Excel
updated_timestamps.to_excel("Data/timestamps_clean.xlsx", sheet_name="Timestamps", index=False)

print("✅ Timestamps updated successfully!")

import librosa
import soundfile as sf
import os

# Directory where your audio files are stored
audio_dir = "Data/fomc_audio"
audio_cut_dir="Data/fomc_audio_cut"
# Loop through each row in updated_timestamps to process the audio files
for _, row in updated_timestamps.iterrows():
    # Define the output path using the 'id' column
    output_path = os.path.join(audio_cut_dir, f"{row['id']}.wav")
    
    # Path to the original audio file in the 'fomc_audio' directory
    filename = os.path.join(audio_dir, f"{row['date']}.wav")
    
    # Load the audio file (Librosa automatically converts to mono by default)
    y, sr = librosa.load(filename, sr=None)  # sr=None keeps the original sample rate
    
    # Get the start and end time in seconds
    start_time = row['start_sec']
    end_time = row['end_sec']
    
    # Convert the start and end times to samples
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    # Trim the audio based on the calculated sample indices
    trimmed_audio = y[start_sample:end_sample]
    
    # Save the trimmed audio to the output path
    sf.write(output_path, trimmed_audio, sr)  # Use output_path variable instead of the string "output_path"
    
    print(f"✅ Trimmed audio saved as {row['id']}.wav")

    
    