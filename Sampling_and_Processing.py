import librosa
import soundfile as sf
import os
import pandas as pd
from natsort import natsort_keygen
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def init_metadata(metadata, csv_data, merge_window):

    for clip, group in csv_data.groupby('ID'):
        gap = group['Timestamp'].diff() != 1
        gap_ids = gap.cumsum()
        interval = group.groupby(gap_ids)['Timestamp'].agg(['min','max'])
        metadata[clip] = list(interval.itertuples(index = False, name = None))



    for key, lst in metadata.items():
        
        if not lst:
            continue
            
        merged = []
        
        current_start, current_end = lst[0]
        
        for next_start, next_end in lst[1:]:
            
            if next_start - current_end < merge_window:
                current_end = max(current_end, next_end) 
            else:
                merged.append((current_start, current_end))
                current_start, current_end = next_start, next_end
                
        merged.append((current_start, current_end))
        
        metadata[key][:] = merged

        return metadata


def normalize_tuple(tuples_list):
    if not tuples_list:
        return []

    global_min = 0.0  
    global_max = tuples_list[-1][1]

    normalized = []

    for a, b in tuples_list:
        diff = b - a

        if diff == 4.0:
            normalized.append((a, b))

        elif diff < 4.0:
            pad_total = 4.0 - diff
            pad_each = pad_total / 2.0

            new_a = a - pad_each
            new_b = b + pad_each

            if new_a < global_min:
                shortfall = global_min - new_a
                new_a = global_min
                new_b += shortfall

            elif new_b > global_max:
                excess = new_b - global_max
                new_b = global_max
                new_a -= excess

            normalized.append((new_a, new_b))

        else:
            curr_a = a
            while True:
                curr_b = curr_a + 4.0

                if curr_b > global_max:
                    normalized.append((global_max - 4.0, global_max))
                    break
                
                elif curr_b >= b:
                    normalized.append((curr_a, curr_b))
                    break
                
                else:
                    normalized.append((curr_a, curr_b))
                    curr_a += 2.5  

    return normalized



def splice_tuple(tuples_list):
    spliced_chunks = []
    
    for start, end in tuples_list:
        current_start = start
        
        while current_start + 4.0 <= end:
            spliced_chunks.append((current_start, current_start + 4.0))
            
            current_start += 5.0
            
    return spliced_chunks


def process_metadata(metadata, method):
    for key in metadata.keys():
        if(method == 0.0):
            metadata[key] = normalize_tuple(metadata.get(key))
        if(method == 1.0):
            metadata[key] = splice_tuple(metadata.get(key))

    return metadata


def sort_csv(clip_filter):
    df = pd.read_csv('train.csv')
    cols = df[['Id','Expected']]

    sub = cols[cols['Expected'] == clip_filter]

    id_col = sub['Id'].drop_duplicates()

    sorted_cols = id_col.sort_values(key=natsort_keygen())

    final_cols = sorted_cols.str.extract(r'(.*\.wav)(\d+)')

    final_cols.columns = ['ID', 'Timestamp']

    final_cols['Timestamp'] = final_cols['Timestamp'].astype(float)
    if(clip_filter == 1.0):
        final_cols.to_csv('positive_ids.csv', index=False)
        return final_cols

    if(clip_filter == 0.0):
        final_cols.to_csv('negative_ids.csv', index = False)
        return final_cols

skipped_counter = 0
processed_counter = 0
short_clip_counter = 0

def chop_and_resample(audio_dict, input_dir, output_dir):
    TARGET_SR = 32000  

    for filename, segments in audio_dict.items():
        input_path = os.path.join(input_dir, filename)
        
        base_name = os.path.splitext(filename)[0]

        if not os.path.exists(input_path):
            print(f"'{input_path}' not found. Skipping.")
            global skipped_counter
            skipped_counter = skipped_counter + 1
            continue

       
 
        try:
            print(f"Loading and resampling {filename} into memory...")
            y_full, sr = librosa.load(input_path, sr=TARGET_SR)

            for start, end in segments:
                start_sample = int(start * TARGET_SR)
                
                end_sample = int((end + 1.0) * TARGET_SR)
                if end_sample > len(y_full):
                    print(f"  -> Skipping clip at {start}s: Source file too short for 5 full seconds.")
                    global short_clip_counter
                    short_clip_counter += 1
                    continue
                y_segment = y_full[start_sample:end_sample]

                start_str = f"{int(start)}" if start.is_integer() else f"{start}".replace('.', '_')
                output_filename = f"{base_name}__{start_str}.wav"
                output_path = os.path.join(output_dir, output_filename)

                sf.write(output_path, y_segment, TARGET_SR)
                print(f"  -> Successfully saved: {output_filename}")
                global processed_counter
                processed_counter += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")

def split_tuple(input_tuples):
    result = []
    step = 2.5
    size = 4.0

    for start, end in input_tuples:
        current_start = float(start)
        end = float(end)
        added = False
        
        while current_start + size <= end:
            result.append((current_start, current_start + size))
            added = True
            current_start += step
            
        if not added:
            result.append((end - size, end))
        else:
            last_end = (current_start - step) + size
            if last_end < end:
                result.append((end - size, end))

    return result


def splice_video(p):
    duration = librosa.get_duration(path = p)
    audio_dict = {}
    audio_dict[p] = split_tuple([(0.0, duration)])
    return audio_dict

def convert_audio_to_spectrograms(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    valid_extensions = ('.wav', '.mp3', '.m4a', '.flac', '.ogg')
    audio_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    print(f"Processing {len(audio_files)} files from '{input_dir}'...")
    
    for file_name in tqdm(audio_files):
        input_path = os.path.join(input_dir, file_name)
        
        base_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(output_dir, f"{base_name}.png")
        
        try:
            y, sr = librosa.load(input_path, sr=None)
            
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
            
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
            
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.axis('off')
            
            librosa.display.specshow(S_dB, sr=sr, hop_length=512, cmap='viridis', ax=ax)
            
            plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close(fig) 
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
