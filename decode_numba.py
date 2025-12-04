# Istituto Nazionale di Fisica Nucleare (INFN)
# High Performance DAQ for Imaging Project
#
# Authors:
#  - A. Bortone
#  - T. Corna
#  - V. Pagliarino
#  - E. Petrini
#  - E. Posteraro
#
# Description:
# Parse data from the binary file from DMA and store to uncompressed TIFF files.
#

#%%
import argparse
from tqdm import tqdm
from numba import njit
import numpy as np
from PIL import Image
import tifffile as tiff
import os 

# Numba is a Just-In-Time (JIT) compiler for Python that translates
# a subset of Python and NumPy code into optimized machine code at runtime.
# It improves performance by compiling numerical functions on-the-fly,
# enabling execution speeds comparable to low-level languages such as C or C++,
# while maintaining the simplicity and flexibility of Python.

@njit
def process_events(words, timestamps):
    x_list = []
    y_list = []
    i = 0

    for w, t in zip(words, timestamps):
        
        # Report status
        i = i + 1
        if i % 1000000 == 0:
            print(f"Processed ", int(i/1000000), "M words...")
            
        # Elaborate data
        col = w & 0xF
        sec = (w >> 4) & 0xF
        dummy = (w >> 16) & 0x1
        hitmap = (w >> 17) & 0xFF
        pr = (w >> 25) & 0x7F

        # Reverse bits 8-bit
        rev = 0
        tmp = hitmap
        for _ in range(8):
            rev = (rev << 1) | (tmp & 1)
            tmp >>= 1

        for bit in range(8):
            if (rev >> bit) & 1:
                # Calcolo X
                px = 1 if bit in (1, 2, 5, 6) else 0
                x_list.append(px + 2 * col + 32 * sec)

                # Calcolo Y
                if bit in (0, 1):
                    py = 1
                elif bit in (2, 3):
                    py = 0
                elif bit in (4, 5):
                    py = 3 if dummy == 0 else -1
                else:
                    py = 2 if dummy == 0 else -2

                y_list.append(py + 4 * pr)

    return np.array(x_list, dtype=np.int32), np.array(y_list, dtype=np.int32)


def parse_binary_file(filename):
    data = np.fromfile(filename, dtype=np.uint8).reshape(-1, 8)

    # Conversione su dtype nativo (little-endian)
    words = np.frombuffer(data[:, :4].tobytes(), dtype='>u4').astype(np.uint32)
    timestamps = np.frombuffer(data[:, 4:].tobytes(), dtype='<u4').astype(np.uint32)
    return words, timestamps

@njit
def fast_hist2d_numba(x, y, bins_x=512, bins_y=512):
    H = np.zeros((bins_x, bins_y), dtype=np.int64)
    for i in range(x.size):
        xi = x[i]
        yi = y[i]
        if 0 <= xi < bins_x and 0 <= yi < bins_y:
            H[xi, yi] += 1
    return H

def decode_raw(filename):
    print("\nDECODING RAW DATA: precompiling Numba JIT...")
    filename_no_ext = os.path.splitext(os.path.basename(filename))[0]
    tiff16 = os.path.join(os.path.dirname(filename), filename_no_ext + "_tiff16.tiff")
    tiff32 = os.path.join(os.path.dirname(filename), filename_no_ext + "_tiff32.tiff")
    words, timestamps = parse_binary_file(filename)
    x, y = process_events(words, timestamps)
    print("Binning data into 2D matrix...")
    # H, x_hist, y_hist = np.histogram2d(x, y, bins=(512, 512), range=((0, 512), (0, 512)))
    H = fast_hist2d_numba(x, y, bins_x=512, bins_y=512)

    #hist_norm = (H / H.max() * 65535).astype(np.uint16)
    #img = Image.fromarray(hist_norm)
    img = Image.fromarray(H.astype(np.uint16)) # Directly save the counts without normalization (sami)
    print("Writing uncompressed TIFF...")
    img.save(tiff16, compression=None)
    tiff.imwrite(tiff32, H.astype(np.float32), compression=None)
    print("Completed.")
    return H 


# Entry point da linea di comando
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decodifica un file RAW e salva tiff16/tiff32.")
    parser.add_argument("file_path", help="Percorso del file RAW da decodificare")
    args = parser.parse_args()

    if not os.path.isfile(args.file_path):
        print(f"Errore: il file '{args.file_path}' non esiste!")
    else:
        decode_raw(args.file_path)
        
