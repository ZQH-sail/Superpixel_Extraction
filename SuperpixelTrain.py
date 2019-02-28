# from skimage.filters import sobel
from skimage.segmentation import slic, mark_boundaries
import cv2 #much faster image loading than skimage
import numpy as np
import glob
# import sys
import pandas as pd
from tqdm import tqdm
import subprocess
from argparse import ArgumentParser


# Threshold for deciding if the label of a superpixel should be 0 or 1
# Currently if 2% or more of the superpixel has a mask, it is labelled true
THRESHOLD = 0.02

# Pretty optimal values. Found using trial-and-error
SEGMENTS = 200
COMPACTNESS = 35

def crop_img(img):
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = im>0
    return img[np.ix_(mask.any(1),mask.any(0))]

def save_patches(sp_list,labels,counter,csv=[],p_sav=''):
    for j in range(len(sp_list)):
        cv2.imwrite(p_sav+'train/Patch_'+str(counter)+'.png',sp_list[j])
        csv.append([p_sav+'train/Patch_'+str(counter),labels[counter]])
        counter+=1
    csv_df = pd.DataFrame(np.array(csv))
    csv_df.to_csv('Labels.csv')
    return counter,csv

def main():    
    parser = ArgumentParser(
        description="Script for generating superpixel training masks, from actual masks."
        " Also outputs cropped superpixels extracted from images and their labels.")
    parser.add_argument('-i','--image_path',help='Path to directory with images', required=True)
    parser.add_argument('-m','--mask_path',help='Path to directory with masks', required=True)
    parser.add_argument('-b','--band_path',help='Path to directory to save new bands', required=True)
    parser.add_argument('-p','--patch_path',help='Path to directory to save patches', required=True)
    parser.add_argument('-c','--counter',help='Initial value of counter. Use if save path already has patches saved', required=False, default=0)
    parser.add_srgument('-s','--shutdown',help='Shutdown VM after script execution completes', required=False, default=None)
    args = parser.parse_args()
    
    im = args.image_path
    msk = args.mask_path
    b_sav = args.band_path
    p_sav = args.patch_path

    im_addr = sorted(glob.glob(im+"/*"))
    msk_addr = sorted(glob.glob(msk+"/*"))

    sp_list = []
    labels = []
    csv = []
    counter = args.counter
    if(counter>0):
        df = pd.read_csv('Labels.csv')
        csv = list(df.itertuples(index=False))
    
    for i in tqdm(range(len(im_addr))):
        im = cv2.imread(im_addr[i])

        mask = cv2.imread(msk_addr[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Returns 2D array with same size as image.
        # Each pixel has an integer value denoting which
        # superpixel it belongs to.
        seg = slic(im, n_segments=SEGMENTS, compactness=COMPACTNESS)

        new_band = np.zeros_like(im[:,:,0])
        for sp_index in range(np.max(seg)+1):
            # Get the rows and column of the pixels in current spixel
            rows,cols = np.where(seg==sp_index) 
            # Number of pixels in current spixel
            num_pixels = rows.shape[0] 
            super_pixel = np.zeros_like(im)
            mask_segment =  np.zeros_like(im[:,:,0])
                        
            # Generate cropped spixel
            super_pixel[rows,cols,:] += im[rows,cols,:]        
            super_pixel = crop_img(super_pixel)
            super_pixel = cv2.resize(super_pixel,(128,128),cv2.INTER_AREA)

            # Extract portion of mask
            mask_segment[rows,cols] += mask[rows,cols]

            # Calculate label
            num_true = np.sum(mask_segment)/255
            fraction_true = float(num_true)/num_pixels
            label = 1 if fraction_true>THRESHOLD else 0
            
            sp_list.append(super_pixel)
            labels.append(label)

            # Calculate the new band
            if(label==1):
                new_band[rows,cols] += 255

        # Save the new band
        cv2.imwrite(b_sav+'/Band4_'+(im_addr[i].split('/')[-1])[:-5]+'.png',new_band)

        # Save patches every 50 images processed
        # Delete patches from memory after saving
        if(i%50 == 49):
            counter,csv = save_patches(sp_list,labels,counter,csv,p_sav) 
            sp_list = []

    save_patches(sp_list,labels,counter,csv)
    if args.shutdown:
        subprocess.Popen('sudo shutdown -h now')
        
if __name__ == '__main__':
    main()






