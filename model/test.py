import cv2
import pytesseract
import easyocr
import io
import lmdb
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import sys
import random

# Define text extraction functions for each OCR model
def extract_text_tesseract(image):
    return pytesseract.image_to_string(image, lang='eng+kor')

def extract_text_easyocr(image):
    reader = easyocr.Reader(['en', 'ko'])
    results = reader.readtext(np.array(image))
    return ' '.join([result[1] for result in results])

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

# Define main function to run the experiment
def main(dataset_path, num_samples=30):
    dataset = lmdbDataset_real(root=dataset_path)
    indices = random.sample(range(len(dataset)), num_samples)
    tesseract_correct = 0
    easyocr_correct = 0

    for i in indices:
        img_HR, img_lr, ground_truth = dataset[i]
        img_lr_pil = img_lr.convert('RGB')

        # Extract text using different OCR models
        text_tesseract = extract_text_tesseract(img_lr_pil)
        text_easyocr = extract_text_easyocr(img_lr_pil)

        # Evaluate accuracy
        if evaluate_accuracy(text_tesseract, ground_truth):
            tesseract_correct += 1

        if evaluate_accuracy(text_easyocr, ground_truth):
            easyocr_correct += 1

        # Print results for each image
        print(f"Image Index: {i}")
        print(f"Tesseract Accuracy: {evaluate_accuracy(text_tesseract, ground_truth)}, Extracted Text: {text_tesseract}")
        print(f"EasyOCR Accuracy: {evaluate_accuracy(text_easyocr, ground_truth)}, Extracted Text: {text_easyocr}")
        print("")

    # Calculate and print final accuracy percentages
    tesseract_accuracy_percentage = (tesseract_correct / num_samples) * 100
    easyocr_accuracy_percentage = (easyocr_correct / num_samples) * 100

    print(f"Final Tesseract Accuracy: {tesseract_accuracy_percentage:.2f}%")
    print(f"Final EasyOCR Accuracy: {easyocr_accuracy_percentage:.2f}%")

if __name__ == "__main__":
    dataset_path = '/Users/yhwancha/Desktop/CSE352/ai_project/dataset/easy'  # Path to the LMDB dataset
    main(dataset_path)
