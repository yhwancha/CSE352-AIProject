import cv2
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'


def extract_text_from_image(image):
    text = pytesseract.image_to_string(image, lang='kor+eng')
    return text

def extract_credit_card_info(text):
    card_number_pattern = r'\b(?:\d[ -]*?){13,16}\b'
    matches = re.findall(card_number_pattern, text)
    return matches

def main(image_path):
    text = extract_text_from_image(image_path)
    print('Extracted Text:', text)
    card_numbers = extract_credit_card_info(text)
    return card_numbers

if __name__ == "__main__":
    image_path = '../data/card3_lower_resolution.jpg'
    card_numbers = main(image_path)
    print("Extracted Credit Card Numbers:", card_numbers)
