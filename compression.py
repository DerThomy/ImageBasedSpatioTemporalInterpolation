import os
import shutil
import subprocess
import re
import argparse
import OpenEXR
import numpy as np
import pandas as pd
import os
import cv2
import time
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import zstandard as zstd

FFMPEG = r"ffmpeg.exe"

CAM_BITRATE_SPLIT = {"main" : 0.6, "left": 0.2, "right": 0.2, "mid": 0.2}
BUFFER_BITRATE_SPLIT = {"color": 0.5, "depth": 0.2, "backwardFlowField": 0.1, "forwardFlowField": 0.1, "backwardShadowColorA": 0.025, "backwardShadowColorB": 0.025, "forwardShadowColorA": 0.025, "forwardShadowColorB": 0.025}

zstd_cmp_lvl = 10
zstd_threads = 12

# framerate_server = 10 # see args.server_fps
framerate_client = 120

hevc_preset = "p7"

decode_cmd = f'{FFMPEG} -hide_banner -loglevel error -y -hwaccel cuda -i "{{}}" -start_number 0 -fps_mode passthrough -pix_fmt {{}} "{{}}"'

full_range_filter = f'-vf "scale=in_range=full:out_range=full" -bsf:v "hevc_metadata=video_full_range_flag=1"'

color_in_fmt = "yuv444p"
shadow_info_in_fmt = "gbrp"

flow_in_fmt = "gbrp12le"
depth_in_fmt = "gray12le"

color_out_fmt = 'rgba -vf "colorchannelmixer=aa=1"'
shadow_info_out_fmt = 'rgba -vf "colorchannelmixer=aa=1"'
depth_out_fmt = "gray16be"
flow_out_fmt = "rgb48be"

def encode_cmd_12bit(folder, lossless, quality, filter, pix_fmt, out, mode, server_fps):
    return f'{FFMPEG} -hide_banner -loglevel error -y -hwaccel cuda -r {server_fps} -re -i "{folder}/%04d.png" -c:v libx265 -tune zerolatency -preset slow -x265-params "log-level=quiet{":lossless=1" if lossless else ""}" {"-b:v " + str(quality) + "M" if mode == "cbr" else "-crf " + str(quality)} {filter} -pix_fmt {pix_fmt} -an "{out}"'

def encode_cmd_new(folder, lossless, quality, filter, pix_fmt, out, mode, server_fps):
    if lossless:
        tune = "lossless -rc constqp"
    else:
        if mode == "vbr":
            quality_part = f"-cq {quality}"
        else:
            quality_part = f"-b:v {quality}M"
        tune = f"ll -rc {mode} {quality_part}"
    return f'{FFMPEG} -hide_banner -loglevel error -y -re -r {server_fps} -i "{folder}/%04d.png" -c:v hevc_nvenc -preset {hevc_preset} -tune {tune} -an {filter} -pix_fmt {pix_fmt} "{out}"'

def load_exr(filepath):
    """ Load an EXR file and return its R, G, B channels as float16 arrays. """
    with OpenEXR.File(filepath) as exr:
        img_data = next(iter(exr.channels().values())).pixels
        img_data = np.where(np.abs(img_data) < 0.0001, 0.0, img_data)
        return img_data[:, :, :-1] if img_data.ndim > 2 else img_data

def save_exr(filepath, img_data):
    if img_data.ndim > 2:
        alpha_channel = np.ones_like(img_data[:, :, 0], dtype=np.float16)
        channels = { "RGBA": np.dstack((img_data, alpha_channel))}
    else:
        channels = { "Y": img_data }
    header = { "compression" : OpenEXR.ZIP_COMPRESSION,
           "type" : OpenEXR.scanlineimage }

    with OpenEXR.File(header, channels) as outfile:
        outfile.write(filepath)

def load_png_12bit(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED).astype(np.uint16)
    return img

def save_png_12bit(filepath, img_data):
    cv2.imwrite(filepath, img_data)

def normalize_exr(img_data):
    min_val, max_val = np.min(img_data), np.max(img_data)

    if abs(max_val - min_val) <= 1e-6:
        return np.zeros_like(img_data, dtype=np.uint16), 0, 0
    
    normalized = np.round(np.clip((img_data.astype(np.float32) - min_val) / (max_val - min_val), 0.0, 1.0) * np.float32(np.iinfo(np.uint16).max)).astype(np.uint16)
    
    return normalized, min_val, max_val

def denormalize_exr(img_data, min_val, max_val):
    if abs(max_val - min_val) <= 1e-6:
        return np.zeros_like(img_data, dtype=np.float16)

    denormalized = ((img_data.astype(np.float32) / np.float32(np.iinfo(np.uint16).max)) * np.float32(max_val - min_val) + np.float32(min_val)).astype(np.float16)
    return denormalized

def convert_RBG_to_16bit(input_path: str):
    img = cv2.imread(input_path, cv2.IMREAD_COLOR_RGB)

    b_channel = img[:, :, 2].astype(np.uint16)
    g_channel = img[:, :, 1].astype(np.uint16)
    
    combined = np.bitwise_or(b_channel, (np.left_shift(g_channel, 8)))
    
    cv2.imwrite(input_path, combined)

def convert_16bit_to_RGB(input_path: str):
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED).astype(np.uint16)
    
    g_channel = np.right_shift(img, 8).astype(np.uint8)
    b_channel = np.bitwise_and(img, 0xFF).astype(np.uint8)

    # Reconstruct an RGB image
    reconstructed = np.stack([np.zeros_like(g_channel), g_channel, b_channel], axis=-1)
    reconstructed = cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(input_path, reconstructed)

def is_single_channel_exr(image_file):
    """Check if the EXR image has only one channel (grayscale) on Windows."""
    cmd = f'{FFMPEG} -i "{image_file}" 2>&1 | findstr /i "Stream #0:0"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return "gray" in result.stdout

def apply_morphology(filepath: str, kernel_size: int = 3, iterations: int = 2):
    # Structuring element for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)  # Preserve alpha if present
    
    # Compute binary mask: 1 where at least one channel is non-zero
    mask = np.any(image[:, :, :3] > 0, axis=2, keepdims=True).astype(np.uint8) * 255

    # Apply morphology on the binary mask
    mask = cv2.dilate(cv2.erode(mask, kernel, iterations=iterations), kernel, iterations=iterations)
    
    # Apply the mask to the original image
    processed_image = image.copy()
    processed_image[:, :, :3] *= (mask[:, :, None] > 0)
    
    # Overwrite original image
    cv2.imwrite(filepath, processed_image)

    return mask

def apply_morphology_mask(filepath: str, mask):
    image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    # Apply the mask to the original image
    processed_image = image.copy()
    processed_image[:, :, :3] *= (mask[:, :, None] > 0)
    
    # Overwrite original image
    cv2.imwrite(filepath, processed_image)

    return mask

def delta_encode(folder, image_paths):
    encoded_frames = {}
    prev_frame = None
    
    for i, (img_path, _) in enumerate(image_paths):
        img_path = os.path.join(folder, img_path)
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if prev_frame is None:
            encoded_frames[img_path] = img  # Store first frame as-is
        else:
            delta = np.bitwise_xor(prev_frame, img)  # Compute delta
            encoded_frames[img_path] = delta
        prev_frame = img
    
    return encoded_frames

def rle_encode(data: bytes) -> bytes:
    """Fast RLE for byte data using NumPy."""
    arr = np.frombuffer(data, dtype=np.uint8)
    n = len(arr)

    if n == 0:
        return b""

    # Find run boundaries
    diffs = np.diff(arr)
    change_indices = np.nonzero(diffs != 0)[0] + 1
    run_starts = np.concatenate(([0], change_indices, [n]))
    encoded = bytearray()

    for i in range(len(run_starts) - 1):
        start = run_starts[i]
        end = run_starts[i + 1]
        value = arr[start]
        run_len = end - start

        while run_len > 0:
            count = min(run_len, 255)
            if count > 1:
                encoded.append(count)
                encoded.append(value)
            else:
                encoded.append(0)
                encoded.append(value)
            run_len -= count

    return bytes(encoded)

def rle_decode(data: bytes) -> bytes:
    """Decodes RLE [count, value] format."""
    out = bytearray()
    i = 0
    while i < len(data):
        count = data[i]
        value = data[i + 1]
        if count == 0:
            out.append(value)
        else:
            out.extend([value] * count)
        i += 2
    return bytes(out)

def compress_raw_images_lossless(input_dir, output_file, server_fps, level=10, delta_mode=False, rle_mode=False):
    """Compresses raw RGBA image data with optional XOR delta and/or RLE."""
    cctx = zstd.ZstdCompressor(level=zstd_cmp_lvl, threads=zstd_threads)
    prev_frame = None
    total_pixels = 0
    frame_count = 0
    raw_size = 0
    imgs = []
    sizes = []

    for i in sorted(os.listdir(input_dir)):
        if i.lower().endswith('.png'):
            img = Image.open(os.path.join(input_dir, i)).convert('RGB')
            imgs.append(np.asarray(img, dtype=np.uint8)[:,:,2])
            sizes.append(img.size)

    start_time = time.time()  # Start measuring time

    with open(output_file, 'wb') as out_stream:
        with cctx.stream_writer(out_stream) as compressor:
            for (pixel_array, (width, height), filename) in zip(imgs, sizes, sorted(os.listdir(input_dir))):
                if filename.lower().endswith('.png'):
                    #print(f"[RAW{' Δ' if delta_mode else ''}{' + RLE' if rle_mode else ''}] Compressing: {filename}")
                    flat_bytes = pixel_array.flatten()

                    if delta_mode:
                        if frame_count % server_fps == 0:
                            prev_frame = pixel_array.flatten().copy()
                        elif prev_frame is not None:
                            flat_bytes = np.bitwise_xor(flat_bytes, prev_frame)

                    if rle_mode:
                        flat_bytes = rle_encode(flat_bytes.tobytes())
                    else:
                        flat_bytes = flat_bytes.tobytes()

                    #compressed_data = compress_on_gpu(flat_bytes)
                    # Write metadata
                    filename_bytes = filename.encode('utf-8')
                    compressor.write(len(filename_bytes).to_bytes(2, 'little'))
                    compressor.write(filename_bytes)
                    compressor.write(width.to_bytes(4, 'little'))
                    compressor.write(height.to_bytes(4, 'little'))
                    compressor.write(len(flat_bytes).to_bytes(4, 'little'))
                    compressor.write(flat_bytes)

                    total_pixels += width * height
                    raw_size += len(flat_bytes)
                    frame_count += 1

        end_time = time.time()  # End measuring time

    # Calculate file size and Mb/s
    file_size = os.path.getsize(output_file)
    elapsed_time = end_time - start_time  # Time taken to compress

    if elapsed_time > 0:
        mb_per_sec = (file_size * 8) / elapsed_time / 1_000_000  # Mb/s
        bits_per_pixel = (file_size * 8) / total_pixels if total_pixels else 0
    else:
        mb_per_sec = 0

    return mb_per_sec

def decompress_raw_images_lossless(input_file, output_dir, server_fps, delta_mode=False, rle_mode=False):
    dctx = zstd.ZstdDecompressor()
    prev_frame = None

    os.makedirs(output_dir, exist_ok=True)
    idx = 0
    with open(input_file, 'rb') as compressed_file:
        with dctx.stream_reader(compressed_file) as reader:
            while True:
                header = reader.read(2)
                if not header:
                    break  # EOF

                # Read filename
                name_len = int.from_bytes(header, 'little')
                filename = reader.read(name_len).decode('utf-8')

                # Read metadata
                width = int.from_bytes(reader.read(4), 'little')
                height = int.from_bytes(reader.read(4), 'little')
                data_len = int.from_bytes(reader.read(4), 'little')
                raw_data = reader.read(data_len)

                if rle_mode:
                    raw_data = rle_decode(raw_data)

                flat = np.frombuffer(raw_data, dtype=np.uint8)

                if delta_mode:
                    if idx % server_fps == 0:
                        prev_frame = flat.copy()
                    else:
                        flat = np.bitwise_xor(flat, prev_frame)
                
                image_array = flat.reshape((height, width))
                rgba_array = np.zeros((height, width, 4), dtype=np.uint8)
                rgba_array[..., 2] = image_array
                rgba_array[..., 3] = 255
                img = Image.fromarray(rgba_array, mode="RGBA")
                img.save(os.path.join(output_dir, filename))

                idx += 1

def preprocess_images(folder, extension, offset):
    """Rename images in the folder to have contiguous numbering, applying the offset."""
    image_files = sorted([f for f in os.listdir(folder) if f.endswith(extension) and "mask" not in f])

    # Rename images to have contiguous numbering
    renamed_files = []
    min_values = []
    max_values = []

    for i, image_file in enumerate(image_files):
        new_name = f"{i:04d}.{extension}"  # New name in contiguous order (0000.png, 0001.png, ...)
        new_file_path = os.path.join(folder, new_name)
        os.rename(os.path.join(folder, image_file), new_file_path)
        renamed_files.append((new_name, image_file))  # Keep track of the original names

        if extension == "exr":
            # Convert EXR images to 12 bit png
            img_data = load_exr(new_file_path)
            normalized, min_val, max_val = normalize_exr(img_data)
            save_png_12bit(os.path.join(folder, f"{i:04d}.png"), normalized)
            min_values.append(min_val)
            max_values.append(max_val)
        elif extension == "png" and "ShadowInfo" in folder:
            morph_mask = apply_morphology(new_file_path, kernel_size=3, iterations=3 if "main" in folder else 2)
            for dirpath, dirnames, _ in os.walk(os.path.dirname(folder)):
                if folder.replace("Info", "Color") in dirpath:
                    cv2.imwrite(os.path.join(dirpath, f"{i:04d}_mask.png"), morph_mask)
        elif extension == "png" and "ShadowColor" in folder:
            mask_path = os.path.join(folder, f"{i:04d}_mask.png")
            morph_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            apply_morphology_mask(new_file_path, morph_mask)
            os.remove(mask_path)

    return renamed_files, min_values, max_values

def postprocess_images(folder, renamed_files, min_values, max_values, extension):
    image_files = sorted([f for f in os.listdir(folder) if f.endswith("png")])

    if extension == "exr":
        for i, (image_file, min_val, max_val) in enumerate(zip(image_files, min_values, max_values)):
            decoded_image = load_png_12bit(os.path.join(folder, image_file))
            denormalized = denormalize_exr(decoded_image, min_val, max_val)
            save_exr(os.path.join(folder, f"{i:04d}.exr"), denormalized)
            os.remove(os.path.join(folder, image_file))

    restore_original_names(folder, renamed_files)

def restore_original_names(folder, renamed_files):
    """Restore the original filenames from the renamed images."""
    for new_name, original_name in reversed(renamed_files):
        os.rename(os.path.join(folder, new_name), os.path.join(folder, original_name))

def ignore_client_color(directory, files):
    if "client" in directory:
        return {"color"}
    return set()

def copy_folder(src, dst):
    """Copy entire directory tree from src to dst."""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore_client_color)
    print(f"Copied {src} → {dst}")

def is_leaf_folder(folder):
    """Check if a folder is a leaf (has no subdirectories)."""
    return all(not os.path.isdir(os.path.join(folder, sub)) for sub in os.listdir(folder))

def find_image_sequences(folder):
    """Find PNG or EXR sequences matching %04d format."""
    files = os.listdir(folder)
    exr_files = sorted([f for f in files if re.match(r'^\d{4}\.exr$', f)])
    png_files = sorted([f for f in files if re.match(r'^\d{4}\.png$', f)])

    if exr_files:
        return "exr", exr_files
    elif png_files:
        return "png", png_files
    return None, []

def encode_color(folder, output_video, crf, mode, server_fps):
    return encode_cmd_new(folder, False, crf, full_range_filter, color_in_fmt, output_video, mode, server_fps)

def encode_flowfield(folder, output_video, crf, mode, server_fps):
    #return encode_cmd_new(folder, False, crf, full_range_filter, flow_in_fmt, output_video, mode)
    return encode_cmd_12bit(folder, False, crf, full_range_filter, flow_in_fmt, output_video, mode, server_fps)

def encode_depth(folder, output_video, crf, mode, server_fps):
    #return encode_cmd_new(folder, False, crf, "", depth_in_fmt, output_video, mode)
    return encode_cmd_12bit(folder, False, crf, full_range_filter, depth_in_fmt, output_video, mode, server_fps)

def convert_images_to_video(folder, image_type, quality, mode, server_fps):
    """Convert images in a folder to HEVC (H.265) mkv video."""
    output_video = os.path.join(folder, "compressed.mkv")
    split_path = os.path.normpath(folder).split(os.sep)

    if mode == "cbr":
        quality *= CAM_BITRATE_SPLIT[split_path[-2]]
        quality *= BUFFER_BITRATE_SPLIT[split_path[-1]]

    if image_type == "exr":
        if is_single_channel_exr(os.path.join(folder, "0000.exr")):
            cmd = encode_depth(folder, output_video, quality if mode == "cbr" else quality[1], mode, server_fps)
        else:
            cmd = encode_flowfield(folder, output_video, quality if mode == "cbr" else quality[1], mode, server_fps)
    else:  # PNG
        cmd = encode_color(folder, output_video, quality if mode == "cbr" else quality[1], mode, server_fps)
    
    print(f"Encoding {image_type.upper()} sequence in {folder} → {output_video}")
    subprocess.run(cmd, shell=True, check=True)

def convert_video_to_images(folder, image_type):
    """Convert mkv video back to images (overwrites originals)."""
    input_video = os.path.join(folder, "compressed.mkv")

    if not os.path.exists(input_video):
        print(f"Skipping {folder}, no compressed.mkv found.")
        return

    image_pattern = os.path.join(folder, "%04d.png")

    pix_fmt = color_out_fmt
    if image_type == "exr":
        pix_fmt = depth_out_fmt if is_single_channel_exr(os.path.join(folder, "0000.exr")) else flow_out_fmt
    elif "ShadowInfo" in folder:
        pix_fmt = shadow_info_out_fmt

    cmd = decode_cmd.format(input_video, pix_fmt, image_pattern)

    print(f"Decoding {input_video} → {image_type.upper()} images in {folder}")
    subprocess.run(cmd, shell=True, check=True)

def generate_latex_table(names: list, origs: np.ndarray, comps: np.ndarray):
    data = []
    for (name, orig, comp) in zip(names, origs, comps):
        name = name.replace("\\", "/").replace("_", "\\_")
        cam_name, buffer_name = name.split("/")
        data.append([cam_name, buffer_name, orig, comp])
    
    orig_df = pd.DataFrame(data, columns=["cam", "buffer", "original", "compressed"])

    MiB = 2**20
    df = orig_df.copy()
    df["ratio"] = df["original"] / df["compressed"]
    df["original"] /= MiB
    df["compressed"] /= MiB

    # Only include certain cams
    filtered_df = df[df["cam"].isin(["main", "left", "right", "mid"])].copy()

    # Exclude ShadowInfo buffers when calculating compression totals
    non_shadow_df = filtered_df[~filtered_df["buffer"].str.contains("ShadowInfo")]

    # Compressed totals per cam (excluding ShadowInfo)
    cam_group = non_shadow_df.groupby("cam")["compressed"].sum()
    total_compressed = cam_group.sum()

    # Compute portion of each buffer within its cam
    filtered_df["buffer_portion_in_cam"] = filtered_df.apply(
        lambda row: row["compressed"] / cam_group[row["cam"]]
        if row["cam"] in cam_group and not "ShadowInfo" in row["buffer"]
        else float('nan'),
        axis=1
    )

    # Compute cam portion of total compressed size
    filtered_df["cam_portion_compressed"] = filtered_df["cam"].map(
        lambda cam: cam_group[cam] / total_compressed if cam in cam_group else float('nan')
    )

    # Pivot into LaTeX-ready format
    pivot_df = pd.pivot_table(
        filtered_df,
        values=["original", "compressed", "ratio", "cam_portion_compressed", "buffer_portion_in_cam"],
        index=["cam", "buffer"],
        columns=[]
    )

    # Total row
    total_values = {
        "original": filtered_df["original"].sum(),
        "compressed": filtered_df["compressed"].sum(),
        "ratio": filtered_df["ratio"].mean(),
        "cam_portion_compressed": float('nan'),
        "buffer_portion_in_cam": float('nan')
    }
    total_row = pd.DataFrame([total_values])
    total_row.index = pd.MultiIndex.from_tuples([("total", "")], names=["cam", "buffer"])

    # Build final table
    pivot_df_with_total = pd.concat([pivot_df, total_row])
    latex_table = pivot_df_with_total.to_latex(float_format="%.2f")

    return latex_table, orig_df

def process_leaf_folder(dirpath, quality, mode, server_fps):
    image_type, images = find_image_sequences(dirpath)
    size_orig, size_comp = 0, 0
    processed = False

    if images:
        out = os.path.join(dirpath, "compressed.mkv")
        size_orig = sum(d.stat().st_size for d in os.scandir(dirpath) if d.is_file())
        renamed_files, min_values, max_values = preprocess_images(dirpath, image_type, offset=6)
        if "ShadowInfo" in dirpath:
            out = os.path.join(dirpath, "compressed.zstd")
            compress_raw_images_lossless(dirpath, out, server_fps)
            decompress_raw_images_lossless(out, dirpath, server_fps)
        else:
            convert_images_to_video(dirpath, image_type, quality, mode, server_fps)
            convert_video_to_images(dirpath, image_type)
        postprocess_images(dirpath, renamed_files, min_values, max_values, image_type)
        size_comp = os.stat(out).st_size
        processed = True
    else:
        print(f"Skipping {dirpath}, no valid images found.")
    return dirpath, size_orig, size_comp, processed

def process_folders(root, crfs, bitrate, mode, server_fps):
    # Gather all leaf directories from the root
    leaf_dirs = []
    shadow_info_dirs = []
    for dirpath, dirnames, _ in os.walk(root, topdown=False):
        if is_leaf_folder(dirpath) and not "client" in dirpath and ("main" in dirpath or "left" in dirpath or "right" in dirpath or "mid" in dirpath):
            if "ShadowInfo" in dirpath:
                shadow_info_dirs.append(dirpath)
            else:
                leaf_dirs.append(dirpath)

    dirnames, sizes_orig, sizes_comp = [], [], []

    
    with ProcessPoolExecutor(8) as executor:
        future_to_dir = {executor.submit(process_leaf_folder, d, crfs if mode == "vbr" else bitrate, mode, server_fps): d for d in shadow_info_dirs}
        for future in as_completed(future_to_dir):
            d = future_to_dir[future]
            try:
                dir, size_orig, size_comp, processed = future.result()
                if processed:
                    dirnames.append(os.sep.join(dir.strip(os.sep).split(os.sep)[-2:]))
                    sizes_orig.append(size_orig)
                    sizes_comp.append(size_comp)
                    print(f"Finished processing {dir}")
            except Exception as exc:
                print(f"Folder {d} generated an exception: {exc}")

    # Process leaf folders concurrently
    with ProcessPoolExecutor(8) as executor:
        future_to_dir = {executor.submit(process_leaf_folder, d, crfs if mode == "vbr" else bitrate, mode, server_fps): d for d in leaf_dirs}
        for future in as_completed(future_to_dir):
            d = future_to_dir[future]
            try:
                dir, size_orig, size_comp, processed = future.result()
                if processed:
                    dirnames.append(os.sep.join(dir.strip(os.sep).split(os.sep)[-2:]))
                    sizes_orig.append(size_orig)
                    sizes_comp.append(size_comp)
                    print(f"Finished processing {dir}")
            except Exception as exc:
                print(f"Folder {d} generated an exception: {exc}")

    latex_table_str, df = generate_latex_table(dirnames, np.array(sizes_orig), np.array(sizes_comp))
    with open(os.path.join(root, "compression_table.tex"), "w") as f:
        f.write(latex_table_str)

    df.to_json(os.path.join(root, "compression_results.json"), orient="records")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("args")
    parser.add_argument("source_path", help="Source folder to compress")
    parser.add_argument("target_path", help="Target folder to render to")
    parser.add_argument("-mode", choices=["vbr", "cbr"], default="cbr", help="Use constant quality (vbr) or constant bitrate")
    parser.add_argument("-bitrate", type=int, default=30, help="Bitrate to use when mode is in cbr")
    parser.add_argument("-color_crf", type=int, default=30, help="CRF for color")
    parser.add_argument("-depth_crf", type=int, default=12, help="CRF for depth")
    parser.add_argument("-flow_crf", type=int, default=12, help="CRF for flow-fields")
    parser.add_argument("-server_fps", type=int, default=10, help="Server Framerate FPS")
    parser.add_argument("-render_path", default=None, help="Path to render folder (optional, for upsampling frames)")
    args = parser.parse_args()

    source_folder = os.path.normpath(args.source_path)
    if not os.path.exists(source_folder):
        exit(-1)

    destination_folder = os.path.normpath(args.target_path)

    copy_folder(os.path.abspath(source_folder), os.path.abspath(destination_folder))
    process_folders(os.path.abspath(destination_folder), [args.color_crf, args.depth_crf, args.flow_crf], args.bitrate, args.mode, args.server_fps)

    if (args.render_path is None) or (not os.path.exists(args.render_path)):
        print("Skipping upsampling frames, no render path provided or does not exist.")
    else:
        print("Upsampling Frames")
        render_folder = os.path.normpath(args.render_path)
        subprocess.run(fr'.\scripts\run.bat {os.path.dirname(destination_folder)} {os.path.basename(destination_folder)} client_ol {os.path.dirname(render_folder)} {os.path.basename(render_folder)} "-a left right" -r -v', shell=True)