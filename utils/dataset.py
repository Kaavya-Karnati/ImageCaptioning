import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import nltk
from collections import Counter
import torchvision.transforms as transforms

nltk.download('punkt')

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            tokens = nltk.word_tokenize(sentence.lower())
            frequencies.update(tokens)
        
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = len(self.stoi)
                self.itos[len(self.itos)] = word

    def numericalize(self, text):
        tokens = nltk.word_tokenize(text.lower())
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]

class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file, header=0)  # Treat the first row as header
        self.transform = transform

        # Initialize and build the vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.df["caption"].tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        
        # Ensure the image name has the correct extension
        if not img_name.endswith(".jpg"):
            img_name += ".jpg"
        
        # Construct the image path
        img_path = os.path.join(self.root_dir, img_name)
        
        # Check if the file exists
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # Load and transform the image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Convert caption to numerical tokens
        caption = self.df.iloc[idx, 1]
        numerical_caption = [self.vocab.stoi["<SOS>"]] + self.vocab.numericalize(caption) + [self.vocab.stoi["<EOS>"]]

        return image, torch.tensor(numerical_caption)
