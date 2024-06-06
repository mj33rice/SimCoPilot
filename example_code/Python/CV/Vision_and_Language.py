import torch, os, json
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer
from PIL import Image

class MovieDataset(torch.utils.data.Dataset):
    def __init__(self, folder = 'dataset-resized-256max', split = 'dev',
                 image_transform = None,
                 tokenizer = RobertaTokenizer.from_pretrained('roberta-base')):
        self.json_dir = os.path.join(folder, split, 'metadata')
        self.image_dir = os.path.join(folder, split, 'images')
        self.image_transform = image_transform
        self.tokenizer = tokenizer

        # Category definitions of movies.
        self.categories = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography',
                           'Comedy', 'Crime', 'Documentary', 'Drama',
                           'Family', 'Fantasy', 'Film-Noir', 'History',
                           'Horror', 'Music', 'Musical', 'Mystery', 'News',
                           'Reality-TV', 'Romance', 'Sci-Fi', 'Short',
                           'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']
        self.categories2ids = {category: id for (id, category)in enumerate(self.categories)}

        # Load JSON files.
        print('Loading %s ...' % self.json_dir, end = '')
        fdir = os.listdir(self.json_dir)
        self.metadata = [(fname[:-5], json.load(open(os.path.join(self.json_dir, fname))))
                     for fname in sorted(fdir) if not fname.startswith('.')]
        print(' finished')

        # Pre-tokenizing all sentences.
        # See documentation for what encode_plus does and each of its parameters.
        print('Tokenizing...', end = '')
        self.tokenized_plots = list()
        self.texts = list()
        for i in range(0, len(self.metadata)):
            text = self.metadata[i][1]['plot'][0]
            if self.tokenizer:
              encoded_text = self.tokenizer.encode_plus(
                  text, add_special_tokens = True, truncation = True,
                  max_length = 256, padding = 'max_length',
                  return_attention_mask = True,
                  return_tensors = 'pt')
              self.tokenized_plots.append(encoded_text)
            self.texts.append(text)
        print(' finished')

    def __getitem__(self, index: int):
        # Load images on the fly.
        filename, movie_data = self.metadata[index]
        img_path = os.path.join(self.image_dir, filename + '.jpeg')
        image = Image.open(img_path).convert('RGB')

        if self.tokenizer:
          text = self.tokenized_plots[index]['input_ids'][0]
          text_mask = self.tokenized_plots[index]['attention_mask'][0]
        else:
          text = self.texts[index]

        genres = movie_data['genres']

        if self.image_transform: image = self.image_transform(image)

        # Encode labels in a binary vector.
        label_vector = torch.zeros((len(self.categories)))
        label_ids = [self.categories2ids[cat] for cat in genres]
        label_vector[label_ids] = 1

        if self.tokenizer:
          return image, text, text_mask, label_vector
        else:
          return image, text, label_vector

    def load_image_only(self, index: int):
        filename, movie_data = self.metadata[index]
        img_path = os.path.join(self.image_dir, filename + '.jpeg')
        image = Image.open(img_path).convert('RGB')
        return image

    def get_metadata(self, index: int):
        _, movie_data = self.metadata[index]
        return movie_data

    def __len__(self):
        return len(self.metadata)
    
# Loading the dataloader.
val_data = MovieDataset(split = 'dev')
print('Data size: %d samples' % len(val_data))