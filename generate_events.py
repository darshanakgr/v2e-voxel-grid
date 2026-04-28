import argparse
import glob
import logging
import os
import sys
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from v2ecore.slomo_mem import SuperSloMoMem
from v2ecore.emulator import EventEmulator
from v2ecore.v2e_utils import all_images, read_image


def read_image_folder(input_folder):
    image_files = []
    image_files.extend(glob.glob(os.path.join(input_folder, "*.jpg")))
    
    image_files = sorted(image_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    return np.array(image_files)


def read_images(image_files):
    images = []
    # output_width, output_height = 320, 240

    for image_file in tqdm(image_files, desc="RGB2GRAY"):
        frame = cv2.imread(image_file)
        # frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        images.append(frame)
        
    return np.array(images)


def load_superslomo_model(upsampling_factor=5, batch_size=16):
    slomo = SuperSloMoMem(model_path="input/SuperSloMo39.ckpt", upsampling_factor=upsampling_factor, batch_size=batch_size)
    return slomo


def main():
    input_folder = "data/scene0711_00/color"
    up_sampling_factor = 5
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    img_files = read_image_folder(input_folder)
    images = read_images(img_files)
    
    if len(images) == 0:
        print("No images found in the input folder.")
        return
    
    output_height, output_width = images[0].shape
    print(f"--- Image Resolution: {output_width}x{output_height}")
    
    slomo = None
    
    for bs in [32, 16, 8, 4, 2, 1]:
        try:
            slomo = load_superslomo_model(upsampling_factor=up_sampling_factor, batch_size=bs)
            out_frames, interp_times = slomo.interpolate(images)
            break
        except torch.OutOfMemoryError:
            print(f"Batch size {bs} is too large for the available GPU memory. Trying smaller batch size.")
            
            if bs == 1:
                print("Batch size 1 is still too large.")
                return
            
            del slomo
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            continue
        
    if slomo is not None:
        del slomo
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # adding last frame and time
    last_frame = np.expand_dims(images[-1], axis=0)
    out_frames = np.concatenate([out_frames, last_frame], axis=0)
    
    interp_times = np.concatenate([interp_times, [interp_times[-1] + 1 / up_sampling_factor]], axis=0)
    
    emulator = EventEmulator(
        pos_thres=0.2,
        neg_thres=0.2,
        sigma_thres=0.03,
        cutoff_hz=0,
        leak_rate_hz=0.1,
        shot_noise_rate_hz=0.001,
        refractory_period_s=0.0005,
        seed=0,
        output_width=output_width,
        output_height=output_height,
        device=device
    )
    
    n_frames = out_frames.shape[0]
    
    voxel_grid = np.zeros((5, output_height, output_width), dtype=np.float32)

    emulator.reset()

    with tqdm(total=n_frames, desc='dvs', unit='fr') as pbar:
        with torch.no_grad():
            for i in range(n_frames):
                events = emulator.generate_events(out_frames[i], interp_times[i])
                pbar.update(1)

                if i == 0:
                    continue
                
                if events is not None and events.shape[0] > 0:
                    np.add.at(voxel_grid, ((i - 1) % 5, events[:, 2].astype(int), events[:, 1].astype(int)), events[:, 3])

                if i % 5 == 0:
                    np.savez_compressed(f"data/dump/{os.path.splitext(os.path.basename(img_files[i // 5]))[0]}.npz", voxel_grid=voxel_grid)
                    voxel_grid = np.zeros((5, output_height, output_width), dtype=np.float32)
    
if __name__ == "__main__":
    main()