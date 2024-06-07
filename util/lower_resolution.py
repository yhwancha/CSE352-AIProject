import cv2


def convert_to_blurry_image(input_image_path, output_image_path, blur_strength=81):
    image = cv2.imread(input_image_path)
    # Apply Gaussian blur to the image
    blurry_image = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    # Save the blurry image
    cv2.imwrite(output_image_path, blurry_image)

    print(f"Blurry image saved to: {output_image_path}")


# Example usage
input_image_path = '../data/card3.jpg'
output_image_path = '../data/card3_lower_resolution_3.jpg'
convert_to_blurry_image(input_image_path, output_image_path, blur_strength=61)
