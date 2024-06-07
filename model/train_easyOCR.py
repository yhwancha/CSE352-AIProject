import easyocr
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import io
import lmdb
import numpy as np

# Define function to convert LMDB buffer to PIL Image
def buf2PIL(txn, key, type='RGB'):
    imgbuf = txn.get(key)
    buf = io.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    im = Image.open(buf).convert(type)
    return im

# Define custom dataset class for LMDB
class LMDBDataset(Dataset):
    def __init__(self, root, transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        if not self.env:
            raise RuntimeError(f"Cannot create LMDB from {root}")
        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get(b'num-samples'))
        self.transform = transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index < len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        img_lr_key = b'image_lr-%09d' % index  # Adjust key according to your dataset
        word = str(txn.get(label_key).decode())
        img_lr = buf2PIL(txn, img_lr_key, 'RGB')
        if self.transform:
            img_lr = self.transform(img_lr)
        return img_lr, word

# Define transformations
transform = transforms.Compose([
    transforms.Resize((32, 128)),  # Resize to fixed size for consistency
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
dataset_path = '/Users/yhwancha/Desktop/CSE352/ai_project/dataset/easy'
dataset = LMDBDataset(root=dataset_path, transform=transform)

# Create DataLoader
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the character set, including punctuation and whitespace
characters = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?"\'#&()[]{}-:; _/+%@*'

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load EasyOCR model
reader = easyocr.Reader(['en'], gpu=device.type == 'cuda')

# Prepare the model for training
model = reader.recognizer  # Get the recognizer model from EasyOCR
model = model.to(device)  # Move model to the device

# Freeze all layers except the last fully connected layer
for param in model.parameters():
    param.requires_grad = False
model.model[-1].weight.requires_grad = True
model.model[-1].bias.requires_grad = True

# Training loop
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
criterion = torch.nn.CTCLoss(blank=0)

# Create a label converter
class LabelConverter:
    def __init__(self, characters):
        self.characters = characters
        self.char_to_index = {char: i + 1 for i, char in enumerate(characters)}
        self.index_to_char = {i + 1: char for i, char in enumerate(characters)}
        self.index_to_char[0] = '-'  # CTC blank token

    def encode(self, text):
        try:
            return [self.char_to_index[char] for char in text]
        except KeyError as e:
            print(f"Unexpected character encountered in text: {e}")
            return []

    def decode(self, indices):
        return ''.join([self.index_to_char[index] for index in indices if index != 0])

label_converter = LabelConverter(characters)

for epoch in range(10):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        optimizer.zero_grad()
        images = images.to(device)  # Move images to the device

        labels_encoded = [label_converter.encode(label) for label in labels]
        valid_indices = [i for i, label in enumerate(labels_encoded) if label]

        if len(valid_indices) != len(images):
            images = images[valid_indices]
            labels_encoded = [labels_encoded[i] for i in valid_indices]

        if not labels_encoded:
            continue  # Skip if no valid labels

        labels_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(label) for label in labels_encoded], batch_first=True, padding_value=0
        ).to(device)  # Move labels to the device
        label_lengths = torch.tensor([len(label) for label in labels_encoded], dtype=torch.long).to(device)  # Move label lengths to the device

        # Forward pass
        outputs = model(images)
        batch_size, seq_len, _ = outputs.size()
        output_lengths = torch.full((batch_size,), seq_len, dtype=torch.long).to(device)

        # Ensure the number of sequences in outputs matches the batch size
        if batch_size != len(label_lengths):
            print(f"Skipping batch due to mismatch: batch_size={batch_size}, len(label_lengths)={len(label_lengths)}")
            continue

        # Adjust the shape of outputs to match the requirements of CTCLoss
        outputs = outputs.permute(1, 0, 2)  # (seq_len, batch_size, num_classes)

        loss = criterion(outputs.log_softmax(2), labels_padded, output_lengths, label_lengths)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')

# Save the trained model
torch.save(model.state_dict(), 'fine_tuned_easyocr_model.pth')