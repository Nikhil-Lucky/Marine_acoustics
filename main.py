import pandas as pd
from natsort import natsort_keygen
import Sampling_and_Processing
import embeddings_extractor
import numpy as np

normalize = 0.0
splice = 1.0

wav_file_path = '/home/masood/Desktop/code/marine_acoustics/train_audio_files/'
normalized_positive_clip_file_path = '/home/masood/Desktop/code/marine_acoustics/normalized_positive_clips/'
spliced_negative_clip_file_path = '/home/masood/Desktop/code/marine_acoustics/spliced_negative_clips/'

positive_metadata = {}
negative_metadata = {}

'''
df_positive = Sampling_and_Processing.sort_csv(1.0)
df_negative = Sampling_and_Processing.sort_csv(0.0)

negative_metadata = Sampling_and_Processing.init_metadata(negative_metadata, df_negative, 0.0)
print(negative_metadata.get('64025.wav'))

negative_metadata = Sampling_and_Processing.process_metadata(negative_metadata, splice)
print(negative_metadata.get('64025.wav'))


print()

positive_metadata = Sampling_and_Processing.init_metadata(positive_metadata, df_positive, 2.1)
print(positive_metadata.get('64025.wav'))

positive_metadata = Sampling_and_Processing.process_metadata(positive_metadata, normalize)
print(positive_metadata.get('64025.wav'))

Sampling_and_Processing.chop_and_resample(positive_metadata, wav_file_path, normalized_positive_clip_file_path)
Sampling_and_Processing.chop_and_resample(negative_metadata, wav_file_path, spliced_negative_clip_file_path)

inf = embeddings_extractor.load_perch()
embeddings,labels = embeddings_extractor.extract_embeddings(normalized_positive_clip_file_path, spliced_negative_clip_file_path, inf)

np.save('Embeddings.npy', embeddings)
np.save('Labels.npy', labels)
'''

embeddings = np.load('Embeddings.npy')
labels = np.load('Labels.npy')
embeddings_extractor.train_Lregression_model(embeddings, labels)


