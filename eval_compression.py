import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import json
import sys
import pandas as pd

FFMPEG = r"ffmpeg.exe"

SERVER_FRAME_RATE = int(sys.argv[1])
CLIENT_FRAME_RATE = 120

FPS_SUFFIX = "" if SERVER_FRAME_RATE == 10 else f"_{SERVER_FRAME_RATE}fps"

SCENES = ["sponza\\dynamic", "viking\\dynamic", "robotlab\\dynamic","viking\\static"]
INPUT_FOLDERS  = [f"..\\Recordings{FPS_SUFFIX}\\{s}" for s in SCENES]
OUTPUT_FOLDERS = [f"..\\Recordings{FPS_SUFFIX}_compressed\\{s}" for s in SCENES]
RENDER_FOLDERS = [f"..\\Renderings{FPS_SUFFIX}_curve\\{s}" for s in SCENES]

BIT_RATES = [10, 20, 40, 60, 80, 100]

COMPRESSED_FOLDER = "compressed"
UNCOMPRESSED_FOLDER = "uncompressed"
CLIENT_FOLDER = "client_ol"

metric_names = ["PSNR", "SSIM", "FLIP"]

for i, (input_f, output_f, render_f) in enumerate(zip(INPUT_FOLDERS, OUTPUT_FOLDERS, RENDER_FOLDERS)):
    compression_metrics = {name: [] for name in metric_names}
    gt_metrics = {name: [] for name in metric_names}
    bitrates_comp = []

    for bit_rate in BIT_RATES:
        output_f_b = output_f + "_" + str(bit_rate)
        render_f_b = render_f + "_" + str(bit_rate)
        subprocess.run(f"python ./compression.py {input_f} {output_f_b} -mode cbr -bitrate {bit_rate} -server_fps {SERVER_FRAME_RATE}", shell=True)
        subprocess.run(fr'.\scripts\run_benchmarks.bat {os.path.dirname(output_f_b)} {os.path.basename(output_f_b)} {render_f_b}', shell=True)

        os.makedirs(os.path.join(render_f_b, CLIENT_FOLDER), exist_ok=True)

        subprocess.run(f'{FFMPEG} -v quiet -stats -y -i "{os.path.join(output_f_b, CLIENT_FOLDER, f"{COMPRESSED_FOLDER}.mkv")}" -start_number 0 -fps_mode passthrough "{os.path.join(render_f_b, CLIENT_FOLDER, "%04d.png")}"', shell=True)
        subprocess.run(f'{FFMPEG} -r 120 -i {os.path.join(render_f_b, CLIENT_FOLDER, "%04d.png")} -r 120 -i {os.path.join(input_f, CLIENT_FOLDER, "color", "%04d.png")} -lavfi "[0:v][1:v]libvmaf=log_path={os.path.join(render_f_b, CLIENT_FOLDER, "vmaf.json").replace("\\","/")}:log_fmt=json" -f null - ', shell=True)
        subprocess.run(f'python ./scripts/eval.py {os.path.join(input_f, "client_ol", "color")} {os.path.join(render_f_b, CLIENT_FOLDER)}', shell=True)
        subprocess.run(f'del /q "{os.path.join(render_f_b, CLIENT_FOLDER)}\\*.png"', shell=True)

    subprocess.run(fr'.\scripts\run_benchmarks.bat {os.path.dirname(input_f)} {os.path.basename(input_f)} {render_f}', shell=True)