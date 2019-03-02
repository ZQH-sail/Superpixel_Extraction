from skimage.segmentation import slic
import cv2  # much faster image loading than skimage
import cv2.cvtColor as changeColorSpace
import cv2.COLOR_BGR2GRAY as RGB2GRAY
import numpy as np
import glob
import pandas as pd
from tqdm import tqdm
import subprocess
from argparse import ArgumentParser


class ImageOperations(object):
    """docstring for ImageOperations
    Handles all the image related operations
    :ivar THRESHOLD: Threshold for deciding if the label of a super-pixel should be 0 or 1
                Currently if 2% or more of the super-pixel has a mask, it is labelled true
    :ivar SEGMENTS: Number of segments
    :ivar COMPACTNESS: <Fill this in>
    """

    def __init__(self, img_path, mask_path, b_sav, p_sav, counter):
        self.THRESHOLD = 0.02
        self.SEGMENTS = 200
        self.COMPACTNESS = 35

        self.disk_io_handler = ReaderWriter(img_path, mask_path, b_sav, p_sav, counter)
        self.img_paths = self.disk_io_handler.get_all_img_paths()
        self.mask_paths = self.disk_io_handler.get_all_mask_paths()

        self.super_pixels = []
        self.labels = []

    def process_images(self):
        for iterator, (img_addr, mask_addr) in tqdm(enumerate(zip(self.img_paths, self.mask_paths))):
            # Load
            im = self.disk_io_handler.load_image(img_addr)
            mask = self.disk_io_handler.load_image(mask_addr)

            # Process
            mask = changeColorSpace(mask, RGB2GRAY)
            super_pixel_map = self.get_super_pixels(im)
            new_band = self._get_new_super_pixel_band(super_pixel_map, im, mask)

            # Write
            self.disk_io_handler.save_new_band(img_addr, new_band)
            self._save_patches_on_interval(iterator, 50)
        self.disk_io_handler.save_patches(self.super_pixels, self.labels)

    def get_super_pixels(self, img):
        """ :return 2D array: Same size as image. Each pixel has an integer value denoting which super-pixel it
                            belongs to.
        """
        return slic(img, n_segments=self.SEGMENTS, compactness=self.COMPACTNESS)

    def _get_new_super_pixel_band(self, super_pixel_map, im, mask):
        new_band = np.zeros_like(im[:, :, 0])
        for sp_index in range(np.max(super_pixel_map) + 1):
            # Get the rows and column of the pixels in current super-pixel
            rows, cols = np.where(super_pixel_map == sp_index)
            self._generate_cropped_super_pixel(im, rows, cols)
            mask_segment = self._extract_mask_portion(im, mask, rows, cols)
            label = self._calculate_and_save_label(mask_segment, rows)

            # Calculate the new band
            if label == 1:
                new_band[rows, cols] += 255
        return new_band

    def _generate_cropped_super_pixel(self, im, rows, cols):
        super_pixel = np.zeros_like(im)
        super_pixel[rows, cols, :] += im[rows, cols, :]
        super_pixel = self.crop_img(super_pixel)
        super_pixel = cv2.resize(super_pixel, (128, 128), cv2.INTER_AREA)
        self.super_pixels.append(super_pixel)

    @staticmethod
    def crop_img(img):
        im = changeColorSpace(img, RGB2GRAY)
        mask = im > 0
        return img[np.ix_(mask.any(1), mask.any(0))]

    @staticmethod
    def _extract_mask_portion(im, mask, rows, cols):
        mask_segment = np.zeros_like(im[:, :, 0])
        mask_segment[rows, cols] += mask[rows, cols]
        return mask_segment

    def _calculate_and_save_label(self, mask_segment, rows):
        num_pixels = rows.shape[0]
        num_true = np.sum(mask_segment) / 255
        fraction_true = float(num_true) / num_pixels
        label = 1 if fraction_true > self.THRESHOLD else 0
        self.labels.append(label)
        return label

    def _save_patches_on_interval(self, iterator, interval):
        if iterator % interval == interval - 1:
            self.disk_io_handler.save_patches(self.super_pixels, self.labels)
            self.super_pixels, self.labels = [], []


class ReaderWriter(object):
    """Handles all kinds of required DiskIO"""

    def __init__(self, path_to_imgs, path_to_masks, path_to_save_bands, path_to_save_patches, counter=None):
        self.path_to_imgs = path_to_imgs
        self.path_to_masks = path_to_masks
        self.path_to_save_bands = path_to_save_bands
        self.path_to_save_patches = path_to_save_patches

        if counter:
            self.save_counter = counter
        else:
            self.save_counter = 0

        self.csv_path = 'Labels.csv'
        self._init_csv()

    def _init_csv(self):
        if self.save_counter > 0:
            df = pd.read_csv(self.csv_path)
            self.csv = list(df.itertuples(index=False))
        else:
            self.csv = []

    def save_patches(self, super_pixels, labels):
        for super_pixel, label in zip(super_pixels, labels):
            patch_addr = self._get_addr_string_new_patch()
            patch_name = self._get_name_new_patch()
            self.write_image(patch_addr, super_pixel)
            self.csv.append([patch_name, label])
            self.save_counter += 1
        csv_df = pd.DataFrame(np.array(self.csv))
        csv_df.to_csv(self.csv_path)

    def get_all_img_paths(self):
        return sorted(glob.glob(self.path_to_imgs + "/*"))

    def get_all_mask_paths(self):
        return sorted(glob.glob(self.path_to_masks + "/*"))

    def save_new_band(self, img_addr, new_band):
        band_addr = self._get_addr_string_new_bands(img_addr)
        cv2.imwrite(band_addr, new_band)

    @staticmethod
    def load_image(img_path):
        return cv2.imread(img_path)

    @staticmethod
    def write_image(img_path, img):
        cv2.imwrite(img_path, img)

    def _get_addr_string_new_bands(self, img_addr):
        return self.path_to_save_bands + '/Band4_' + (img_addr.split('/')[-1])[:-5] + '.png'

    def _get_addr_string_new_patch(self):
        return self.path_to_save_patches + 'train/Patch_' + str(self.save_counter) + '.png'

    def _get_name_new_patch(self):
        return self.path_to_save_patches + 'train/Patch_' + str(self.save_counter)


class ArgumentParserHandler(object):
    """Simple class to handle operations pertaining to Python's argument parser"""

    def __init__(self):
        self.parser = ArgumentParser(
            description="Script for generating superpixel training masks, from actual masks."
                        " Also outputs cropped superpixels extracted from images and their labels.")
        self._init_argument_parser()
        self.args = None
        self._parse_args()

    def _init_argument_parser(self):
        self.parser.add_argument('-i', '--image_path', help='Path to directory with images', required=True)
        self.parser.add_argument('-m', '--mask_path', help='Path to directory with masks', required=True)
        self.parser.add_argument('-b', '--band_path', help='Path to directory to save new bands', required=True)
        self.parser.add_argument('-p', '--patch_path', help='Path to directory to save patches', required=True)
        self.parser.add_argument('-c', '--counter', help='Initial value of counter. Use if save path already has '
                                                         'patches saved', required=False, default=0)
        self.parser.add_argument('-s', '--shutdown', help='Shutdown VM after script execution completes',
                                 required=False, default=None)

    def _parse_args(self):
        self.args = self.parser.parse_args()

    def get_paths(self):
        return self.args.image_path, self.args.mask_path, self.args.band_path, self.args.patch_path

    def get_counter(self):
        return self.args.counter

    def shut_down(self):
        if self.args.shutdown:
            subprocess.call(['shutdown', '-h', 'now'])


def main():
    args_parser_handler = ArgumentParserHandler()
    path_to_images, path_to_masks, path_to_save_bands, path_to_save_patches = args_parser_handler.get_paths()
    counter = args_parser_handler.get_counter()

    sp_op = ImageOperations(path_to_images, path_to_masks, path_to_save_bands, path_to_save_patches, counter)
    sp_op.process_images()

    args_parser_handler.shut_down()


if __name__ == '__main__':
    main()
