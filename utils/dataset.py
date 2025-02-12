import torch
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import nltk
from collections import Counter
import json

nltk.download('punkt')

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def build_vocabulary(self, sentences):
        counter = Counter()
        for sentence in sentences:
            counter.update(nltk.word_tokenize(sentence.lower()))

        for word, freq in counter.items():
            if freq >= self.freq_threshold:
                idx = len(self.itos)
                self.itos[idx] = word
                self.stoi[word] = idx

    def numericalize(self, text):
        return [self.stoi.get(word, self.stoi["<UNK>"]) for word in nltk.word_tokenize(text.lower())]

class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.df["caption"].tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_id = self.df.iloc[index, 0]
        caption = self.df.iloc[index, 1]

        img_path = os.path.join(self.root_dir, "Images", img_id)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        numerical_caption = [self.vocab.stoi["<SOS>"]] + self.vocab.numericalize(caption) + [self.vocab.stoi["<EOS>"]]

        return image, torch.tensor(numerical_caption)

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions

# Define Transformations for Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

if __name__ == "__main__":
    dataset = Flickr8kDataset(root_dir="data/Flickr8k", captions_file="data/captions.txt", transform=transform)
    print(f"Dataset Size: {len(dataset)}")
    print(f"Sample Data: {dataset[0]}")
