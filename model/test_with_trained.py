import cv2
import pytesseract
import io
import lmdb
import torch
from torch.utils.data import Dataset
from PIL import Image
import sys
import random
import torchvision.transforms as transforms
import torchvision.models as models

# Define text extraction functions for each OCR model
def extract_text_tesseract(image):
    return pytesseract.image_to_string(image, lang='eng+kor')

def extract_text_crnn(image, model, label_converter, transform):
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
    output = output.permute(1, 0, 2)  # (seq_len, batch_size, num_classes)
    _, predictions = output.max(2)
    predictions = predictions.squeeze(1).cpu().numpy().tolist()
    decoded_text = label_converter.decode(predictions)
    return decoded_text

# Define function to evaluate accuracy
def evaluate_accuracy(extracted_text, ground_truth):
    return extracted_text.strip() == ground_truth.strip()

# Define function to convert LMDB buffer to PIL Image
def buf2PIL(txn, key, type='RGB'):
    imgbuf = txn.get(key)
    buf = io.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    im = Image.open(buf).convert(type)
    return im

# Load LMDB dataset
class lmdbDataset_real(Dataset):
    def __init__(self, root, max_len=30):
        super(lmdbDataset_real, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)
        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get(b'num-samples'))
        self.max_len = max_len

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        img_HR_key = b'image_hr-%09d' % index  # 128*32
        img_lr_key = b'image_lr-%09d' % index  # 64*16
        try:
            img_HR = buf2PIL(txn, img_HR_key, 'RGB')
            img_lr = buf2PIL(txn, img_lr_key, 'RGB')
        except IOError or len(word) > self.max_len:
            return self[index + 1]
        return img_HR, img_lr, word

# Use the same character set used during training
characters = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?"\'#&()[]{}-:; '

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
        if isinstance(indices, int):  # Handle the case where indices is an int
            indices = [indices]
        return ''.join([self.index_to_char[index] for index in indices if index != 0])

label_converter = LabelConverter(characters)

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model class matching the trained model's architecture
class CRNNModel(torch.nn.Module):
    def __init__(self):
        super(CRNNModel, self).__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1')  # Use pretrained weights
        modules = list(resnet.children())[:-2]  # Remove the last FC layer and average pool
        self.cnn = torch.nn.Sequential(*modules)
        self.rnn = torch.nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.fc = torch.nn.Linear(512, 87)  # +1 for CTC blank token

    def forward(self, x):
        x = self.cnn(x)  # Output shape: (batch_size, 512, h, w)
        batch_size, channels, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # Shape: (batch_size, w, channels, h)
        x = x.reshape(batch_size, w, channels * h)  # Shape: (batch_size, w, channels * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# Define main function to run the experiment
def main(dataset_path, model_path, num_samples=30):
    # Load dataset
    dataset = lmdbDataset_real(root=dataset_path)
    indices = random.sample(range(len(dataset)), num_samples)
    tesseract_correct = 0
    crnn_correct = 0

    # Load CRNN model
    model = CRNNModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    for i in indices:
        img_HR, img_lr, ground_truth = dataset[i]
        img_lr_pil = img_lr.convert('RGB')

        # Extract text using different OCR models
        text_tesseract = extract_text_tesseract(img_lr_pil)
        text_crnn = extract_text_crnn(img_lr_pil, model, label_converter, transform)

        # Evaluate accuracy
        if evaluate_accuracy(text_tesseract, ground_truth):
            tesseract_correct += 1

        if evaluate_accuracy(text_crnn, ground_truth):
            crnn_correct += 1

        # Print results for each image
        print(f"Image Index: {i}")
        print(f"Tesseract Accuracy: {evaluate_accuracy(text_tesseract, ground_truth)}, Extracted Text: {text_tesseract}")
        print(f"CRNN Accuracy: {evaluate_accuracy(text_crnn, ground_truth)}, Extracted Text: {text_crnn}")
        print("")

    # Calculate and print final accuracy percentages
    tesseract_accuracy_percentage = (tesseract_correct / num_samples) * 100
    crnn_accuracy_percentage = (crnn_correct / num_samples) * 100

    print(f"Final Tesseract Accuracy: {tesseract_accuracy_percentage:.2f}%")
    print(f"Final CRNN Accuracy: {crnn_accuracy_percentage:.2f}%")

if __name__ == "__main__":
    dataset_path = '/Users/yhwancha/Desktop/CSE352/ai_project/dataset/easy'
    model_path = 'crnn_low_res_model.pth'
    main(dataset_path, model_path)