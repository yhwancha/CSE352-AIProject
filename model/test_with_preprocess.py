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

# Define preprocessing techniques
def preprocess_image(image, technique):
    if technique == 'grayscale':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif technique == 'histogram_equalization':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)
    elif technique == 'adaptive_thresholding':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif technique == 'gaussian_blur':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)
    elif technique == 'median_blur':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.medianBlur(gray, 5)
    elif technique == 'sharpening':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    elif technique == 'resize':
        return cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    else:
        return image

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
def main(dataset_path, techniques, num_samples=50):
    dataset = lmdbDataset_real(root=dataset_path)
    indices = random.sample(range(len(dataset)), num_samples)
    tesseract_correct = {technique: 0 for technique in techniques}
    easyocr_correct = {technique: 0 for technique in techniques}

    for i in indices:
        img_HR, img_lr, ground_truth = dataset[i]
        img_lr_array = np.array(img_lr)

        for technique in techniques:
            # Preprocess image
            processed_image = preprocess_image(img_lr_array, technique)

            # Convert back to PIL image for OCR models that require it
            processed_image_pil = Image.fromarray(processed_image)

            # Extract text using different OCR models
            text_tesseract = extract_text_tesseract(processed_image_pil)
            text_easyocr = extract_text_easyocr(processed_image_pil)

            # Evaluate accuracy
            if evaluate_accuracy(text_tesseract, ground_truth):
                tesseract_correct[technique] += 1

            if evaluate_accuracy(text_easyocr, ground_truth):
                easyocr_correct[technique] += 1

            # Print results for each image and technique
            print(f"Image Index: {i}, Technique: {technique}")
            print(f"Tesseract Accuracy: {evaluate_accuracy(text_tesseract, ground_truth)}, Extracted Text: {text_tesseract}")
            print(f"EasyOCR Accuracy: {evaluate_accuracy(text_easyocr, ground_truth)}, Extracted Text: {text_easyocr}")
            print("")

    # Calculate and print final accuracy percentages for each technique
    for technique in techniques:
        tesseract_accuracy_percentage = (tesseract_correct[technique] / num_samples) * 100
        easyocr_accuracy_percentage = (easyocr_correct[technique] / num_samples) * 100

        print(f"Final Tesseract Accuracy with {technique}: {tesseract_accuracy_percentage:.2f}%")
        print(f"Final EasyOCR Accuracy with {technique}: {easyocr_accuracy_percentage:.2f}%")

if __name__ == "__main__":
    dataset_path = '/Users/yhwancha/Desktop/CSE352/ai_project/dataset/easy'
    techniques = ['grayscale', 'histogram_equalization', 'adaptive_thresholding', 'gaussian_blur', 'median_blur', 'sharpening', 'resize']
    main(dataset_path, techniques)
