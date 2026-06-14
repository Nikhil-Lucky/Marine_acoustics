import os
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def load_perch(model_url='https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1'):
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    print("Loading Perch 2.0 model...")
    perch_model = hub.load(model_url)
    print("Model loaded successfully!")
    return perch_model.signatures['serving_default']

def feed_perch(folder_path, label, infer_fn):
    embeddings = []
    labels = []
    
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith('.wav'):
            continue
            
        filepath = os.path.join(folder_path, filename)
        
        audio, sr = sf.read(filepath)
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1) 
            
        waveform = audio.astype(np.float32)
        
        model_outputs = infer_fn(inputs=waveform[np.newaxis, :])
        embedding = model_outputs['embedding'].numpy()[0]
        
        embeddings.append(embedding)
        labels.append(label)
        
    return embeddings, labels

def extract_embeddings(positive_path, negative_path, inference_function):

    pos_emb, pos_lbl = feed_perch(positive_path, 1, inference_function)
    neg_emb, neg_lbl = feed_perch(negative_path, 0, inference_function)

    x = np.array(pos_emb + neg_emb)
    y = np.array(pos_lbl + neg_lbl)

    print(f"Extraction complete! {x.shape[0]} embeddings of dimension {x.shape[1]}.")
    return x,y


def train_Lregression_model(embeddings, labels, train_split = 0.8, test_split = 0.2):

    x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42, stratify=labels)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    print("Training the classifier...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(x_train_scaled, y_train)
    print("Training complete!\n")

    print("Evaluating on test data...")
    y_pred = clf.predict(x_test_scaled)

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['Negative (0)', 'Orca (1)']))
