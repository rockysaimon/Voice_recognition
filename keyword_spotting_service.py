import os
import librosa
import tensorflow as tf
import numpy as np

SAVED_MODEL_PATH = "modelvoicerecog.h5"
SAMPLES_TO_CONSIDER = 22050

class _Keyword_Spotting_Service:
    model = None
    _mapping = [
        "down",
        "off",
        "on",
        "no",
        "yes",
        "stop",
        "up",
        "right",
        "left",
        "go"
    ]
    _instance = None

    def predict(self, file_path):
       
        
        MFCCs = self.preprocess(file_path)
        
        if MFCCs is None:
            return None

        
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword

    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
       

        try:
            
            signal, sample_rate = librosa.load(file_path)

            if len(signal) >= SAMPLES_TO_CONSIDER:
                
                signal = signal[:SAMPLES_TO_CONSIDER]

                
                MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                return MFCCs.T
            else:
                print(f"Audio file {file_path} is too short: {len(signal)} samples.")
                return None

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

def Keyword_Spotting_Service():
   
    
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance

if __name__ == "__main__":
    
    kss = Keyword_Spotting_Service()

    
    test_directory = "test"
    
    
    for filename in os.listdir(test_directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(test_directory, filename)
            keyword = kss.predict(file_path)
            print(f"File: {file_path}, Predicted keyword: {keyword}")
