import os
import json
import matplotlib.pyplot as plt
import matplotlib
import sys

def load_metrics_from_folder(folder_path):
    json_path = os.path.join(folder_path, "per_image_results.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    
    json_path_full_result = os.path.join(folder_path, "results.json")
    with open(json_path_full_result, "r") as f:
        results = json.load(f)
        print(json_path_full_result, results)
    
    time = []
    psnr = []
    ssim = []
    flip = []
    vmaf = []

    for filename, metrics in data.items():
        t = int(filename.split(".")[0])
        time.append(t)
        psnr.append(metrics["PSNR"])
        ssim.append(metrics["SSIM"])
        flip.append(metrics["FLIP"])
        vmaf.append(metrics["VMAF"])

    # Sort by time
    sorted_indices = sorted(range(len(time)), key=lambda i: time[i])
    time = [time[i] for i in sorted_indices]
    psnr = [psnr[i] for i in sorted_indices]
    ssim = [ssim[i] for i in sorted_indices]
    flip = [flip[i] for i in sorted_indices]
    vmaf = [vmaf[i] for i in sorted_indices]

    return time, psnr, ssim, flip, vmaf

def plot_all_metrics(folders):
    
    colors = plt.get_cmap('Dark2', len(folders))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    metrics_names = ["PSNR", "SSIM", "FLIP", "VMAF"]

    for i, folder in enumerate(folders):
        time, psnr, ssim, flip, vmaf = load_metrics_from_folder(folder)
        color = colors(i)
        label = " - ".join(folder.rstrip('\\').split("\\")[-2:])

        axs[0].plot(time, psnr, label=label, color=color)
        axs[1].plot(time, ssim, label=label, color=color)
        axs[2].plot(time, flip, label=label, color=color)
        axs[3].plot(time, vmaf, label=label, color=color)

    for ax, name in zip(axs, metrics_names):
        ax.set_title(f"{name} over Time")
        ax.set_xlabel("Frame Number")
        ax.set_ylabel(name)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # You can pass folder paths as command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python plot_metrics.py <folder1> <folder2> ...")
    else:
        plot_all_metrics(sys.argv[1:])