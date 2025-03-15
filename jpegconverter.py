import os
from PIL import Image
import pillow_heif

def convert_heif_to_jpeg(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Register HEIF opener
    pillow_heif.register_heif_opener()
    
    # List all files in input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        # Check if the file is HEIF/HEIC
        if filename.lower().endswith(('.heic', '.heif')):
            try:
                # Read HEIF file
                heif_file = pillow_heif.open_heif(input_path, convert_hdr_to_8bit=True)
                
                # Convert to PIL Image
                image = Image.frombytes(
                    heif_file.mode,
                    heif_file.size,
                    heif_file.data,
                    "raw",
                    heif_file.mode,
                    heif_file.stride,
                )
                
                # Create output path with .jpg extension
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, base_name + '.jpg')
                
                # Save as JPEG
                image.save(output_path, format='JPEG', quality=90)
                print(f"Converted: {filename} -> {base_name}.jpg")
                
            except Exception as e:
                print(f"Error converting {filename}: {str(e)}")

if __name__ == "__main__":
    # Set your input and output folders here
    input_folder = "input_images"
    output_folder = "output_images"
    
    convert_heif_to_jpeg(input_folder, output_folder)