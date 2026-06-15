import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)

sys.path.insert(1, parent_dir)

import os
import numpy as np
import soundfile as sf
import joblib
import tensorflow_hub as hub
import tensorflow as tf
import Sampling_and_Processing
import embeddings_extractor


path = '/home/masood/Desktop/code/marine_acoustics/inference/inference_input/'
name = 'OS_7_05_2019_08_24_00_.wav'
o_path = '/home/masood/Desktop/code/marine_acoustics/inference/inference_output/'
 

t_path = path+name


def load_model():
    clf = joblib.load('./LR_model.joblib')
    scaler = joblib.load('./scaler.joblib')
    return clf, scaler


def start_pipeline(inference_dict, inference_function, clf, scaler, spliced_audio_folder = '/home/masood/Desktop/code/marine_acoustics/inference/inference_input/'):
    print("\nStarting Orca detection pipeline...")
    print("-" * 40)

    for video_name, chunks in inference_dict.items():
        for start, end in chunks:
            base_name = os.path.basename(video_name)

            if base_name.lower().endswith('.wav'):
                base_name = base_name[:-4]

            if start == int(start):
                start_str = str(int(start))     
            else:
                start_str = str(start).replace('.', '_') 
            
            filename = f"{base_name}__{start_str}.wav"
            filepath = spliced_audio_folder+filename
            
            if not os.path.exists(filepath):
                print(f"Warning: Could not find {filename}")
                continue
            
            audio, sr = sf.read(filepath)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1) 
            waveform = audio.astype(np.float32)
        
            model_outputs = inference_function(inputs=waveform[np.newaxis, :])
            raw_embedding = model_outputs['embedding'].numpy()[0]
        
            scaled_embedding = scaler.transform(raw_embedding.reshape(1, -1))
        
            prediction = clf.predict(scaled_embedding)[0]
            probabilities = clf.predict_proba(scaled_embedding)[0]
        
            if prediction == 1:
                confidence = probabilities[1] * 100
                print(f"ORCA detected with confidence of {confidence:.1f}%! ")
                print(f"   Video: {video_name}")
                print(f"   Timestamp: {start}s - {end}s")


tf.experimental.numpy.experimental_enable_numpy_behavior()
inference_function = embeddings_extractor.load_perch()
X, Y = load_model()
while(True):
    x = input("enter the name")

    if(x.casefold() == 'exit'):
        break

    Path = path+x

    inference_dict = {}

    inference_dict = Sampling_and_Processing.splice_video(Path)

    y = input("Does the file need to be spliced?")
    if(y.casefold() == 'yes'):
        Sampling_and_Processing.chop_and_resample(inference_dict, path, o_path)

    start_pipeline(inference_dict, inference_function, X, Y)











