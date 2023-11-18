import os
import random
import shutil

def reduce_image_count(input_folder, output_folder, target_count):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Check if the target count is greater than the total number of images
    if target_count >= len(image_files):
        print("Target count is greater than or equal to the total number of images. No reduction needed.")
        return

    # Randomly select images to keep
    selected_images = random.sample(image_files, target_count)

    # Copy selected images to the output folder
    for image in selected_images:
        source_path = os.path.join(input_folder, image)
        destination_path = os.path.join(output_folder, image)
        shutil.copyfile(source_path, destination_path)

    print(f"{target_count} images selected and copied to {output_folder}.")

# Example usage:
input_folder_path = "path/to/your/input/folder"
output_folder_path = "path/to/your/output/folder"
target_image_count = 50  # Set your desired number of images

reduce_image_count(input_folder_path, output_folder_path, target_image_count)
