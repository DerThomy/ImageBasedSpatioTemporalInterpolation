#
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import sys

import matplotlib as mpl

# Enable LaTeX rendering and set font
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",         # or 'sans-serif' depending on your doc
    "font.serif": ["Times"],        # match with your LaTeX document font
    "axes.labelsize": 12,           # font sizes are customizable
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

def METHOD2MARKER(method):
    if "Timewarp (30" in method:
        return "s"
    if "Timewarp (60" in method:
        return "^"
    return "o"

def CAMS(leftright=False, mid=False):
    return ["main"] + (["left", "right"] if leftright else []) + (["mid"] if mid else [])

def BUFFERS(depth=False, motion=False, shadow=False, bidir=False):
    result = ["color"] + (["depth"] if depth else [])
    dirs = ["forward", "backward"] if bidir else ["forward"]
    if motion:
        result += [d + "FlowField" for d in dirs]
    if shadow:
        result += [d + b for b in ["ShadowInfo", "ShadowColorA", "ShadowColorB"] for d in dirs]
    return result


CURVE_PLOT = True
CREATE_TABLE = False

if CURVE_PLOT:
    # Magma
    # CVALS = [
    # '#00438a',
    # '#00438a',
    # '#00438a',
    # '#da4081',
    # '#894398',
    # '#ff6550',
    # '#ffa600',
    # ]
    CVALS = [
        '#265780',
        '#377eb8',
        '#60a6e0',
        '#c4171c',
        '#4daf4a',
        '#984ea3',
        '#ff7f00',
    ]
    custom_cap = mpl.colors.ListedColormap(CVALS, name='custom5')
    METHODS = [
        ("", "timewarp_ol", ("Timewarp (10 FPS)", CAMS(mid=True), BUFFERS())),
        # ("20fps", "timewarp_ol", ("Timewarp (20 FPS)", CAMS(mid=True), BUFFERS())),
        ("30fps", "timewarp_ol", ("Timewarp (30 FPS)", CAMS(mid=True), BUFFERS())),
        ("60fps", "timewarp_ol", ("Timewarp (60 FPS)", CAMS(mid=True), BUFFERS())),
        ("", "splatting_ol", ("Forward Splatting", CAMS(mid=True), BUFFERS(True, True, False, False))),
        ("", "client_ol", ("Video Streaming (120 FPS)", CAMS(), BUFFERS())),
        ("", "ours_extrapolate", ("Ours (extrapolate)", CAMS(leftright=True), BUFFERS(True, True, False, False))),
        ("", "ours_ol", ("Ours", CAMS(leftright=True), BUFFERS(True, True, True, True))),
    ]
elif CREATE_TABLE:
    METHODS = [
        ("", "ours_ol", ("Ours", CAMS(leftright=True), BUFFERS(True, True, True, True))),
        ("", "ours_extrapolate", ("Ours (extrapolate)", CAMS(leftright=True), BUFFERS(True, True, False, False))),
        ("", "splatting_ol", ("Forward Splatting", CAMS(mid=True), BUFFERS(True, True, False, False))),
        ("", "timewarp_ol", ("Timewarp (10 FPS)", CAMS(mid=True), BUFFERS())),
        ("", "backward_bidirectional_ol", ("Backward Bidir.", CAMS(), BUFFERS(True, True, False, True))),
    ]

data = []

METRICS = ["SSIM", "FLIP", "VMAF"]
def load_data(recording_dir_arg, render_dir_arg, subdir, bitrate=0, latency=0, stereo=False):
    for midfix, method, (method_name, cams, buffers) in [(midfix, k + suffix, v) for (midfix, k, v) in METHODS for suffix in (["_left", "_right"] if stereo else [""])]:
        
        recording_dir = recording_dir_arg if midfix == "" else "_".join(recording_dir_arg.split("_")[:1] + [midfix] + recording_dir_arg.split("_")[1:])
        render_dir = render_dir_arg if midfix == "" else "_".join(render_dir_arg.split("_")[:1] + [midfix] + render_dir_arg.split("_")[1:])

        if latency > 0:
            method = f"{method}_l{latency}"
            method_name = f"{method_name} $l={latency}$"
        method_dir = os.path.join(render_dir, subdir, method)
        result_filename = os.path.join(method_dir, "results.json")
        if not os.path.exists(method_dir) or not os.path.exists(result_filename):
            continue

        recording_subdir = os.path.join(recording_dir, subdir)
        compression_results_filename = os.path.join(recording_subdir, "compression_results.json")

        compressed_total_per_second = 0
        if os.path.exists(recording_subdir) and os.path.exists(compression_results_filename):
            compression_df = pd.read_json(compression_results_filename)
            filtered_df = compression_df[compression_df["cam"].isin(cams) & compression_df["buffer"].isin(buffers)]
            compressed_total_byte = filtered_df["compressed"].sum()
            compressed_total_per_second = compressed_total_byte / ((8 if "sponza" in subdir else 15) * 10**6 / 8)
                
        with open(result_filename, "r") as f:
            metrics = json.load(f)
            data.append([os.path.dirname(method_dir).split("\\")[-1].split("_")[0], os.path.dirname(method_dir).split("\\")[-2], method_name, stereo, bitrate, latency, compressed_total_per_second] + [metrics[m] for m in METRICS])

if (len(sys.argv) < 4):
    print("Usage: *.py RecordingDir RenderingDir subfolders")
    exit(0)

recording_dir = sys.argv[1]
render_dir = sys.argv[2]
for i in range(3, len(sys.argv)):
    load_data(recording_dir.replace("_compressed", ""), render_dir, sys.argv[i], stereo=False)

    if CURVE_PLOT:
        for bitrate in [10, 20, 40, 60, 80, 100]:
            load_data(recording_dir, render_dir, f"{sys.argv[i]}_{bitrate}", bitrate=bitrate)

    # for latency in [1,2,4,8]:
    #     load_data(recording_dir, render_dir, f"{sys.argv[i]}", latency=latency)


SCENE_TRANSLATION = {
    "viking": "Viking Village",
    "robotlab": "Robot Lab",
    "sponza": "Sponza",
}

df = pd.DataFrame(data, columns=["mode", "scene", "method", "stereo", "bitrate", "latency", "Mb/s"] + METRICS)
df['scene'] = df['scene'].replace(SCENE_TRANSLATION)

COLUMNS = METRICS# + ["Mb/s"]
print(df)


# Get all unique (mode, scene) combinations
compressed_entries_df = df[df["bitrate"] > 0]
grouped = compressed_entries_df.groupby(['mode', 'scene'])


if CURVE_PLOT:
    # Create plots for both SSIM and FLIP
    for metric in ['FLIP', 'SSIM', 'VMAF']:
        fig, axes = plt.subplots(2, 2, figsize=(2 * 3, 2 * 4), sharex=False, sharey=True)
        axes = axes.flatten()  # Flatten to 1D array for easy indexing

        metric_label = metric + (r"\textsuperscript{↓}" if metric in ["FLIP"] else r"\textsuperscript{↑}")

        all_handles = {}
        for idx, ((mode, scene), group) in enumerate(grouped):
            
            ax = axes[idx]

            # Sort by bitrate to ensure lines are in order
            group_sorted = group.sort_values(by='bitrate')
            ggroup_sorted = group_sorted.groupby('method')
            
            # Plot one line per method
            for method_idx, method in enumerate([m[2][0] for m in METHODS]):
                if method not in ggroup_sorted.indices:
                    continue
                method_group = ggroup_sorted.get_group(method)
                method_group_sorted = method_group.sort_values('Mb/s')
                [line] = ax.plot(method_group_sorted['Mb/s'], method_group_sorted[metric], marker=METHOD2MARKER(method), label=method, color=custom_cap(method_idx), zorder=6)
                
                uncompressed_metric = df[(df["bitrate"] == 0) & (df["method"] == method) & (df["scene"] == scene) & (df["mode"] == mode)][metric]
                if (len(uncompressed_metric) == 1):
                    ax.axhline(y=uncompressed_metric.sum(), color=line.get_color(), linestyle='--', label=f"{method} (Uncompressed)", zorder=3)
                    # if ("Timewarp" in method):
                    #     ax.plot(80, uncompressed_metric.sum(), color=line.get_color(), marker=METHOD2MARKER(method), zorder=3)
                
                # Store the first instance of each method for the legend
                if method not in all_handles:
                    all_handles[method] = line
            
            ax.grid(True, zorder=0)
            ax.set_title(f'{scene} - {mode}')
            ax.set_xlabel('Mb/s')
            
            if (idx % 2) == 0:
                ax.set_ylabel(metric)

        fig.suptitle(f"Compression Ablation: {metric_label} vs. Mb/s")
        
        # fig.tight_layout(pad=0.75)
        # fig.savefig(f"{metric}_curve_nolegend.pdf")
        
        fig.tight_layout(pad=0.75, rect=[0, 0.08, 1, 1])  # leave space at bottom
        fig.legend(
            handles=list(all_handles.values()),
            labels=list(all_handles.keys()),
            loc='lower center',
            ncol=3,
            frameon=True
        )
        # plt.show()
        fig.savefig(f"{metric}_curve_new.pdf")

if CREATE_TABLE:
    pivot_df = pd.pivot_table(df, values=COLUMNS, index="method", columns=["mode", "scene"], aggfunc='mean')
    pivot_df = pivot_df.reorder_levels([1, 2, 0], axis=1)
    pivot_df = pivot_df.sort_index(axis=1, level=[0, 1])
    print(pivot_df)

    MODE_ORDER = ["static", "dynamic"]
    SCENE_ORDER = ["Viking Village", "Sponza", "Robot Lab"]
    sorted_columns = sorted(pivot_df.columns, key=lambda x: (MODE_ORDER.index(x[0]), SCENE_ORDER.index(x[1]), COLUMNS.index(x[2])))
    pivot_df = pivot_df[sorted_columns]
    pivot_df = pivot_df.reindex([m[2][0] for m in METHODS])
    print(pivot_df)

    def highlight_top3(s, precision=2, desc=False, highlight=False):
        ranked = s.rank(ascending=not desc, method='min')
        def rank_to_color(rank):
            return str(int(50 - (rank - 1) * 20)) if highlight and rank <= 3 else "00"
        return [f"\\cellcolor{{tab_color!{rank_to_color(ranked.iloc[i])}}}{v:.{precision}f}" for i, v in enumerate(s)]


    styled_pivot_df = pivot_df.copy()
    for col in styled_pivot_df.columns:
        higher_is_better = col[2] not in ["FLIP", "LPIPS", "Mb/s"]
        precision = 2 if col[2] in ["PSNR", "Mb/s", "VMAF"] else 3
        highlight = False if col[2] == "Mb/s" else True
        styled_pivot_df[col] = highlight_top3(styled_pivot_df[col], precision=precision, desc=higher_is_better, highlight=highlight)

    latex_code = styled_pivot_df.to_latex(escape=False, multicolumn=True, multirow=True)
    print(latex_code)