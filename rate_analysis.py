#%%
import tifffile as tiff
import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.optimize import curve_fit


# --------------------------
# Read TIFF files
# --------------------------
tiff_files = sorted(glob.glob("/Users/samiullahkhan/Downloads/tiff files/tiff/*.tiff"))
matrix_list = []

for f in tiff_files:
    img = tiff.imread(f).astype(np.float32)
    matrix_list.append(img)

# Stack into 3D array: (num_frames, rows, cols)
matrix_3d = np.stack(matrix_list, axis=0)

# --------------------------
# Parameters
# --------------------------
acq_time = 100  # seconds per frame
num_frames, num_rows, num_cols = matrix_3d.shape

# --------------------------
# 1D Histogram: Frame rate over time
# --------------------------
#frame_rates = np.sum(matrix_3d, axis=(4,2)) / acq_time  # sum all pixels per frame
#plt.figure(figsize=(10,5))
#plt.plot(frame_rates, marker='o', color='blue')
#plt.xlabel("Frame Number")
#plt.ylabel("Rate (hits/s)")
#plt.title("1D Histogram: Total Rate per Frame")
#plt.grid(True)
#plt.show()

# --------------------------
# 1D Histogram: Row and Column rates
# --------------------------
#row_rates = np.sum(matrix_3d, axis=(0,2)) / acq_time  # sum over columns and frames
#col_rates = np.sum(matrix_3d, axis=(0,1)) / acq_time  # sum over rows and frames

# --------------------------
# Inject a test pulse
# --------------------------
frame_to_test = 0        # frame index where you want to inject
row_to_test = 10         # row index
col_to_test = 5          # column index
pulse_value = 100        # value of test pulse

matrix_3d[frame_to_test, row_to_test, col_to_test] += pulse_value
print(f"Test pulse added at frame {frame_to_test}, row {row_to_test}, col {col_to_test}")

# --------------------------
# 1D Histogram: Row and Column rates
# --------------------------
row_rates = np.sum(matrix_3d, axis=(0,2)) / acq_time  # sum over columns and frames
col_rates = np.sum(matrix_3d, axis=(0,1)) / acq_time  # sum over rows and frames

# Compute power
row_power = row_rates**2
col_power = col_rates**2

window_size = 10
# ---- Exponential fitting for Rate per Row ----
x = np.arange(len(row_power))
y = np.array(row_power)

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c


col_to_test = 275  # column index
# Sum over rows and frames for that column
col_counts = np.sum(matrix_3d[:, :, col_to_test])
col_rate = col_counts / acq_time
print(f"Column {col_to_test} total counts: {col_counts}, rate: {col_rate} hits/s")


popt, pcov = curve_fit(exponential, x, y, p0=[y.max(), -0.001, y.min()])

plt.figure(figsize=(10,5))
#plt.plot(x, y, 'o', label='Data', color='purple')
#plt.plot(x, exponential(x, *popt), '-', label='Exponential Fit', color='black')
plt.xlabel("Row Number")
plt.ylabel("Power (rate^2)")
plt.title("1D Histogram (Rate per Row) - Exponential Fit")
plt.legend()
plt.grid(True)
plt.show()

print("Fit parameters for Row Power:")
print(f"a = {popt[0]:.4f}, b = {popt[1]:.6f}, c = {popt[2]:.4f}")

window_size = 10
plt.figure(figsize=(10,5))
#plt.plot(col_power, marker='o', color='orange')
plt.xlabel("Column Number")
plt.ylabel("Power (rate^2)")
plt.title("1D Histogram (Rate per Column) - Original")
plt.grid(True)
plt.show()
# --------------------------
# 2D Histogram: Sum over frames
# --------------------------
hit_map = np.sum(matrix_3d, axis=0)  # sum over frames
plt.figure(figsize=(8,6))
plt.imshow(hit_map, cmap='hot', origin='lower')
plt.colorbar(label='Total Hits')
plt.title("2D Histogram: Hits per Pixel (summed over frames)")
plt.xlabel("Column")
plt.ylabel("Row")
plt.show()

# --------------------------
# 3D Histogram: Surface plot of hits per pixel
# --------------------------
X, Y = np.meshgrid(np.arange(num_cols), np.arange(num_rows))
Z = hit_map  # same as 2D histogram but for 3D surface plot
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('Column')
ax.set_ylabel('Row')
ax.set_zlabel('Total Hits')
plt.title("3D Histogram: Hits per Pixel")
plt.show()
print("Row 10 total counts (after test pulse):", row_rates[10] * acq_time)
print("Column 5 total counts (after test pulse):", col_rates[5] * acq_time)








#%%
import tifffile as tiff
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --------------------------
# Helper Functions
# --------------------------

def read_tiff_files(folder_path):
    """Read all TIFF files and stack into 3D array (frames, rows, cols)."""
    tiff_files = sorted(glob.glob(folder_path + "/*.tiff"))
    matrix_list = [tiff.imread(f).astype(np.float32) for f in tiff_files]
    return np.stack(matrix_list, axis=0)

def exponential(x, a, b, c):
    """Exponential function for curve fitting."""
    return a * np.exp(b * x) + c

def fit_exponential_to_column(matrix, frame_index, column_index):
    """Fit exponential to selected column in a single frame."""
    col_values = matrix[frame_index, :, column_index]  # row-wise counts
    rows = np.arange(len(col_values))
    popt, _ = curve_fit(exponential, rows, col_values, p0=[col_values.max(), -0.001, col_values.min()])
    return rows, col_values, popt

def plot_exponential_fit(rows, popt, column_index, frame_index):
    """Plot only the exponential fit for a given column."""
    plt.figure(figsize=(10,5))
    plt.plot(rows, exponential(rows, *popt), '-', color='red', label='Exponential Fit')
    plt.xlabel("Row Number")
    plt.ylabel("Counts (Fitted)")
    plt.title(f"Exponential Fit for Column {column_index} (Frame {frame_index})")
    plt.grid(True)
    plt.legend()
    plt.show()
    print("Fit parameters:")
    print(f"a = {popt[0]:.4f}, b = {popt[1]:.6f}, c = {popt[2]:.4f}")

def plot_exponential_fit(rows, popt, column_index, frame_index):
    """Plot only the exponential fit for a given column."""


def plot_2d_hitmap(matrix):
    """Plot 2D histogram (sum over frames)."""
    hit_map = np.sum(matrix, axis=0)
    plt.figure(figsize=(8,6))
    plt.imshow(hit_map, cmap='hot', origin='lower')
    plt.colorbar(label='Total Hits')
    plt.title("2D Histogram: Hits per Pixel (summed over frames)")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()
    return hit_map

def plot_3d_surface(hit_map):
    """Plot 3D surface histogram of hits per pixel."""
    num_rows, num_cols = hit_map.shape
    X, Y = np.meshgrid(np.arange(num_cols), np.arange(num_rows))
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, hit_map, cmap='viridis')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_zlabel('Total Hits')
    plt.title("3D Histogram: Hits per Pixel")
    plt.show()

# --------------------------
# Main Script
# --------------------------

# Read all TIFF files
folder_path = "/Users/samiullahkhan/Downloads/tiff files/tiff2"
matrix_3d = read_tiff_files(folder_path)
num_frames, num_rows, num_cols = matrix_3d.shape
print(f"Loaded {num_frames} frames of size {num_rows}x{num_cols}")

# Parameters for analysis
frame_to_test = 0      # single frame to analyze
col_to_test = 190      # specific column to analyze

# Fit exponential and plot
rows, col_values, popt = fit_exponential_to_column(matrix_3d, frame_to_test, col_to_test)
plot_exponential_fit(rows, popt, col_to_test, frame_to_test)

# Optional: 2D and 3D visualizations
hit_map = plot_2d_hitmap(matrix_3d)
plot_3d_surface(hit_map)



# %%
import tifffile as tiff
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --------------------------
# Helper Functions
# --------------------------

def read_tiff_files(folder_path):
    """Read all TIFF files and stack into 3D array (frames, rows, cols)."""
    tiff_files = sorted(glob.glob(folder_path + "/*.tiff"))
    matrix_list = [tiff.imread(f).astype(np.float32) for f in tiff_files]
    return np.stack(matrix_list, axis=0)

def read_tiff_files_multiple(folder_path):
    """Read all TIFF files and stack into 3D array (frames, rows, cols)."""
    tiff_files = sorted(glob.glob(folder_path + "/*.tiff"))
    print(tiff_files)
    matrix_list = [tiff.imread(f).astype(np.float32) for f in tiff_files]
    return matrix_list

def exponential(x, a, b, c):
    """Exponential function for curve fitting."""
    return a * np.exp(b * x) + c

def fit_exponential_to_column(matrix, frame_index, column_index):
    """Fit exponential to selected column in a single frame."""
    col_values = matrix[frame_index, :, column_index]  # row-wise counts
    rows = np.arange(len(col_values))
    popt, _ = curve_fit(exponential, rows, col_values, p0=[col_values.max(), -0.001, col_values.min()])
    return rows, col_values, popt

def plot_exponential_fit(rows, popt, column_index, frame_index):
    """Plot only the exponential fit for a given column."""
    plt.figure(figsize=(10,5))
    plt.plot(rows, exponential(rows, *popt), '-', color='red', label='Exponential Fit')
    plt.xlabel("Row Number")
    plt.ylabel("Counts (Fitted)")
    plt.title(f"Exponential Fit for Column {column_index} (Frame {frame_index})")
    plt.grid(True)
    plt.legend()
    plt.show()
    print("Fit parameters:")
    print(f"a = {popt[0]:.4f}, b = {popt[1]:.6f}, c = {popt[2]:.4f}")

def plot_exponential_fit(rows, popt, column_index, frame_index):
    """Plot only the exponential fit for a given column."""


def plot_2d_hitmap(matrix):
    """Plot 2D histogram (sum over frames)."""
    hit_map = np.sum(matrix, axis=0)
    plt.figure(figsize=(8,6))
    plt.imshow(hit_map, cmap='hot', origin='lower')
    plt.colorbar(label='Total Hits')
    plt.title("2D Histogram: Hits per Pixel (summed over frames)")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()
    return hit_map

def plot_3d_surface(hit_map):
    """Plot 3D surface histogram of hits per pixel."""
    num_rows, num_cols = hit_map.shape
    X, Y = np.meshgrid(np.arange(num_cols), np.arange(num_rows))
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, hit_map, cmap='viridis')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_zlabel('Total Hits')
    plt.title("3D Histogram: Hits per Pixel")
    plt.show()

# --------------------------
# Main Script
# --------------------------

folder_path = "/Users/samiullahkhan/Downloads/tiff files/"
matrix_list = read_tiff_files_multiple(folder_path)

col_to_test = 190  # column index for fitting

plt.figure(figsize=(10,6))
plt.title(f"Exponential Fits for Column {col_to_test}")
plt.xlabel("Row Number")
plt.ylabel("Counts")
plt.grid(True)

for i, matrix_2d in enumerate(matrix_list):
    rows, col_values, popt = fit_exponential_to_column(matrix_2d, 0, col_to_test)
    plot_exponential_fit(rows, col_values, popt, label=f"Frame {i}")

plt.legend()
plt.show()
# %%
import tifffile as tiff
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# --------------------------
# Helper Functions
# --------------------------

def read_tiff_files_multiple(folder_path,files):
    """Read all TIFF files and return list of 2D numpy arrays."""
    tiff_files = []
    
    for file in files:
        f = folder_path + file
        print(" -", os.path.basename(f))
        tiff_files.append(f)
    matrix_list = [tiff.imread(f).astype(np.float32) for f in tiff_files]
    return matrix_list

def exponential(x, a, b, c):
    """Exponential model: a * exp(b*x) + c"""
    return a * np.exp(b * x) + c

def fit_exponential_to_column(matrix, frame_index, column_index):
    """Fit exponential to selected column in a single frame (matrix)."""
    col_values = matrix[:, column_index]  # vertical slice (rows)
    rows = np.arange(len(col_values))
    popt, _ = curve_fit(exponential, rows, col_values,
                        p0=[col_values.max(), -0.001, col_values.min()],
                        maxfev=5000)
    return rows, col_values, popt

def fit_exponential_to_column_multiple(matrix, frame_index, column_indexes):
    """Fit exponential to selected column in a single frame (matrix)."""
    col_values_multiple = []
    for column_index in column_indexes:
        print(column_index)
        col_values = matrix[:, int(column_index)]  # vertical slice (rows)
        col_values_multiple.append(col_values)
    
    col_values = np.mean(np.stack(col_values_multiple, axis=0), axis=0)
    rows = np.arange(len(col_values))
    popt, _ = curve_fit(exponential, rows, col_values,
                        p0=[col_values.max(), -0.001, col_values.min()],
                        maxfev=5000)
    return rows, col_values, popt

def plot_exponential_fit(rows, col_values, popt, label):
    """Plot the exponential fit for a given column."""
    plt.plot(rows, col_values, 'o', alpha=0.5, label=f"{label}kV data")
    plt.plot(rows, exponential(rows, *popt), '-', label=f"{label}kV fit")

# --------------------------
# Main Script
# --------------------------

folder_path = "/Users/samiullahkhan/Downloads/tiff files/tiff3/"
col_to_test = 190  # column index for fitting
# col_to_tests = np.linspace(434,458, 458-434+1)
col_to_tests = [i for i in range(434,459)]
print(col_to_tests)


print(col_to_tests)

file_energy = [
     "22_10_25_edge_40kV_tiff16.tiff",
 "22_10_25_edge_60kV_tiff16.tiff",
 "22_10_25_edge_80kV_tiff16.tiff",
    "22_10_25_edge_100kV_tiff16.tiff",
 "22_10_25_edge_120kV_tiff16.tiff",
 "22_10_25_edge_140kV_tiff16.tiff"

]
labels_energy = [
    40,
    60,
    80,
    100,
    120,
    140
]

matrix_list = read_tiff_files_multiple(folder_path,file_energy)

plt.figure(figsize=(10,6))
plt.title(f"Exponential Fits for Column {col_to_test}")
plt.xlabel("Row Number")
plt.ylabel("Counts")
plt.grid(True)

Rows = []
As = []
Bs = []
Cs = []
for i, matrix_2d in enumerate(matrix_list):
    rows, col_values, popt = fit_exponential_to_column_multiple(matrix_2d, 0, col_to_tests)
    print(f"Energy: {labels_energy[i]}kV")
    print("Fit parameters:")
    print(f"a = {popt[0]:.4f}, b = {popt[1]:.6f}, c = {popt[2]:.4f}")
    Rows.append(rows)
    As.append(popt[0])
    Bs.append(popt[1])
    Cs.append(popt[2])

    plot_exponential_fit(rows, col_values, popt, label=f"{labels_energy[i]}")

df = pd.DataFrame({

    "energy": labels_energy,
    "rows": Rows,
    "a": As,
    "b": Bs,
    "c": Cs
}
)

plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.title(f"Exponential Fits for Column {col_to_test}")
plt.xlabel("Row Number")
plt.ylabel("Counts")
plt.grid(True)
for i, row in enumerate(df["rows"]):
    print(row)
    plt.plot(row,exponential(row,df["a"][i],df["b"][i], df["c"][i] ), label=f"{labels_energy[i]}")
plt.legend()
plt.show()

file_csv = folder_path + "exp_params.csv"
df.to_csv(file_csv)
print(df)

# %%
import tifffile as tiff
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# --------------------------
# Helper Functions
# --------------------------

def read_tiff_files_multiple(folder_path, files):
    """Read all TIFF files and return list of 2D numpy arrays."""
    tiff_files = []
    
    for file in files:
        f = folder_path + file
        print(" -", os.path.basename(f))
        tiff_files.append(f)
    matrix_list = [tiff.imread(f).astype(np.float32) for f in tiff_files]
    return matrix_list


def exponential(x, a, b, c):
    """Exponential model: a * exp(b*x) + c"""
    return a * np.exp(b * x) + c


def fit_exponential_to_column_multiple(matrix, frame_index, column_indexes):
    """Fit exponential to selected column in a single frame (matrix)."""
    col_values_multiple = []
    for column_index in column_indexes:
        col_values = matrix[:, int(column_index)]
        col_values_multiple.append(col_values)
    
    col_values = np.mean(np.stack(col_values_multiple, axis=0), axis=0)
    rows = np.arange(len(col_values))
    
    popt, _ = curve_fit(
        exponential, rows, col_values,
        p0=[col_values.max(), -0.001, col_values.min()],
        maxfev=5000
    )
    return rows, col_values, popt


def plot_exponential_fit(rows, col_values, popt, label):
    """Plot the exponential fit for a given column."""
    plt.plot(rows, col_values, 'o', alpha=0.5, label=f"{label}uA data")
    plt.plot(rows, exponential(rows, *popt), '-', label=f"{label}uA fit")


# --------------------------
# Main Script
# --------------------------

folder_path = "/Users/samiullahkhan/Downloads/tiff files/tiff3/"

# Columns for averaging
col_to_tests = list(range(434, 459))
print(col_to_tests)

# ✅ FILES BASED ON CURRENT (90 kV constant)
file_current = [
    "06_11_40uA_90kV_tiff16.tiff",
    "06_11_50uA_90kV_tiff16.tiff",
    "06_11_60uA_90kV_tiff16.tiff",
    "06_11_70uA_90kV_tiff16.tiff",
    "06_11_80uA_90kV_tiff16.tiff",
    "06_11_90uA_90kV_tiff16.tiff"
]

labels_current = [40, 50, 60, 70, 80, 90]  # in µA
voltage = 90  # kV constant

# Read TIFF matrices
matrix_list = read_tiff_files_multiple(folder_path, file_current)

# Plot exponential fits (multiple currents)
plt.figure(figsize=(10,6))
plt.title(f"Exponential Fits for Columns {col_to_tests[0]}–{col_to_tests[-1]} at {voltage}kV")
plt.xlabel("Row Number")
plt.ylabel("Counts")
plt.grid(True)

Rows = []
As = []
Bs = []
Cs = []

for i, matrix_2d in enumerate(matrix_list):
    rows, col_values, popt = fit_exponential_to_column_multiple(matrix_2d, 0, col_to_tests)

    print(f"Current: {labels_current[i]}uA")
    print("Fit parameters:")
    print(f"a = {popt[0]:.4f}, b = {popt[1]:.6f}, c = {popt[2]:.4f}")

    Rows.append(rows)
    As.append(popt[0])
    Bs.append(popt[1])
    Cs.append(popt[2])

    plot_exponential_fit(rows, col_values, popt, label=f"{labels_current[i]}")

plt.legend()
plt.show()

# Save parameters to CSV
df = pd.DataFrame({
    "current_uA": labels_current,
    "voltage_kV": [voltage]*len(labels_current),
    "rows": Rows,
    "a": As,
    "b": Bs,
    "c": Cs
})

file_csv = folder_path + "exp_params_current.csv"
df.to_csv(file_csv)
print(df)


# Second plot showing only fitted curves
plt.figure(figsize=(10,6))
plt.title(f"Fitted Exponentials for Different Currents (Voltage = {voltage}kV)")
plt.xlabel("Row Number")
plt.ylabel("Counts")
plt.grid(True)

for i, row in enumerate(df["rows"]):
    plt.plot(row, exponential(row, df["a"][i], df["b"][i], df["c"][i]),
             label=f"{labels_current[i]}uA")

plt.legend()
plt.show()

# %%
import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

# -------------------------------
# FOLDER PATHS (EDIT IF NEEDED)
# -------------------------------
original_folder = "/Users/samiullahkhan/Downloads/tiff files/tiff3/Original/"
modified_folder = "/Users/samiullahkhan/Downloads/tiff files/tiff3/Sara files"

# -------------------------------
# Load files
# -------------------------------
orig_files = sorted([f for f in os.listdir(original_folder) if f.endswith(".tiff")])
mod_files  = sorted([f for f in os.listdir(modified_folder) if f.endswith(".tiff")])

print("Original files:", orig_files)
print("Modified files:", mod_files)

# Check equal number of files
if len(orig_files) != len(mod_files):
    print("WARNING: Number of files is not equal!")
    print("Ensure the files match 1-to-1.")
    
# -------------------------------
# Process each pair
# -------------------------------
for i in range(min(len(orig_files), len(mod_files))):
    
    orig_path = os.path.join(original_folder, orig_files[i])
    mod_path  = os.path.join(modified_folder,  mod_files[i])
    
    print(f"\nProcessing pair:")
    print("Original:", orig_files[i])
    print("Modified:", mod_files[i])
    
    # Load both images
    orig_img = tiff.imread(orig_path).astype(np.float64)
    mod_img  = tiff.imread(mod_path).astype(np.float64)

    # Pixel-by-pixel difference
    diff = orig_img - mod_img

    # -------------------------------
    # PLOT: difference heatmap
    # -------------------------------
    plt.figure(figsize=(8,6))
    plt.imshow(diff, cmap='seismic', vmin=-np.max(abs(diff)), vmax=np.max(abs(diff)))
    plt.colorbar(label="Pixel Difference")
    plt.title(f"Difference Map: {orig_files[i]} - {mod_files[i]}")
    plt.tight_layout()
    plt.savefig(f"difference_map_{i}.png", dpi=200)
    plt.close()

    # -------------------------------
    # PLOT: column-wise mean difference
    # -------------------------------
    col_profile = np.mean(diff, axis=0)

    plt.figure(figsize=(10,4))
    plt.plot(col_profile)
    plt.title(f"Column-wise Difference Profile ({orig_files[i]})")
    plt.xlabel("Column Index")
    plt.ylabel("Mean Difference")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"column_profile_{i}.png", dpi=200)
    plt.close()

    print(f"Saved: difference_map_{i}.png")
    print(f"Saved: column_profile_{i}.png")

print("\n✅ All difference plots generated successfully!")
# %%