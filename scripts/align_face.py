import argparse
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.seed import set_seed
from utils.image_utils import list_image_files
from utils.shape_predictor import align_face


def alignment_from_path(input_path, output_path, replace_cropped=False):
    output_path.mkdir(parents=True, exist_ok=True)
    image_files = list_image_files(input_path)

    for img_path in image_files:
        img_path_jpg = Path(img_path).with_suffix('.jpg').name
        if (output_path / img_path_jpg).is_file() and not replace_cropped:
            # print(f"Aligned image for {img_path} already exists, skipping.")
            continue

        image = Image.open(input_path / img_path)

        try:
            # returns the largest face
            crop_image = align_face(image, return_tensors=False)
        except Exception as e:
            print(f"Warning: Could not align face for image {img_path}: {e}")
            continue
        
        if len(crop_image) == 1:
            print(f"Saving aligned image to {output_path / img_path_jpg}")
            crop_image[0].save(output_path / img_path_jpg)
        elif len(crop_image) > 1:
            raise NotImplementedError("Multiple faces detected, not yet implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align faces')

    parser.add_argument('-unprocessed_dir', type=Path, default='/workspace/outputs/image', help='directory with unprocessed images')
    parser.add_argument('-output_dir', type=Path, default='/workspace/outputs/aligned_image', help='output directory')
    parser.add_argument('-replace_cropped', action='store_true')

    args = parser.parse_args()

    set_seed(3407)
    alignment_from_path(args.unprocessed_dir, args.output_dir, args.replace_cropped)
