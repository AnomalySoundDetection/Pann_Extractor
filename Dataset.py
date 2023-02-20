from torch.utils.data import Dataset
import librosa
import torch

def load_audio(audio_path, sample_rate):
    audio_data, _ = librosa.core.load(audio_path, sr=sample_rate)
    audio_data = torch.FloatTensor(audio_data)

    return audio_data

class AudioDataset(Dataset):
    def __init__(self, data, _id, root, sample_rate):
        self.data = [sample for sample in data if _id in sample]
        self.root = root
        self.sample_rate = sample_rate

    def __getitem__(self, index):
        file_path = self.data[index]
        audio_data = load_audio(f"{file_path}", sample_rate=self.sample_rate)
        
        return audio_data
    
    def __len__(self):
        return len(self.data)