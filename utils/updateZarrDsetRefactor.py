#!/usr/bin/env python

import zarr
import shutil
import os
import sys
import numpy as np
import argparse
from PIL import Image
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from numcodecs import Blosc

# --- Constants ---
COMPRESSOR = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

def surgical_zarr_update(
    source_zip_path: Path,
    key_to_update: str,
    new_value: np.ndarray
):
    """
    Performs a high-performance, safe, "in-place" update of a Zarr ZipStore.

    It creates a new zip archive by copying unchanged data at the byte level
    and writing only the new/modified data, avoiding full decompression/recompression.
    This is significantly faster and uses less temporary storage than a full extraction.
    The operation is atomic, replacing the original file only on success.

    Args:
        source_zip_path: Path to the .zip Zarr archive.
        key_to_update: The internal path to the dataset (e.g., 'exchange/data').
        new_value: The NumPy array containing the new data.
    """
    new_zip_path = source_zip_path.with_suffix('.zip.new')
    
    # Use a temporary store to prepare the new Zarr dataset chunks
    with tempfile.TemporaryDirectory(suffix="_zarr_staging") as temp_dir:
        staging_store = zarr.DirectoryStore(temp_dir)
        # Create chunks that are reasonable for array datasets
        chunks = (1, *new_value.shape[1:]) if new_value.ndim > 1 else new_value.shape
        zarr.save_array(staging_store, new_value, path=key_to_update, chunks=chunks, compressor=COMPRESSOR)
        
        # Get the list of new files to write (e.g., 'key/.zarray', 'key/0.0')
        new_files_map = {Path(p).as_posix(): Path(temp_dir) / p for p in os.listdir(temp_dir)}

        print(f"Staged new data for key '{key_to_update}' with {len(new_files_map)} files.")

        try:
            with ZipFile(str(source_zip_path), 'r') as source_zip:
                with ZipFile(str(new_zip_path), 'w', compression=ZIP_DEFLATED) as dest_zip:
                    # 1. Copy all members from the old archive EXCEPT those being replaced
                    print("Copying unchanged data...")
                    for item in source_zip.infolist():
                        # The path to check against must not contain the key being updated
                        # e.g., if updating 'exchange/data', skip 'exchange/data/.zarray' etc.
                        if not item.filename.startswith(key_to_update + '/'):
                            dest_zip.writestr(item, source_zip.read(item.filename))
                    
                    # 2. Write the new/updated data from the staging area
                    print("Writing updated data...")
                    for archive_path, local_path in new_files_map.items():
                        dest_zip.write(local_path, archive_path)

        except Exception as e:
            print(f"\nAn error occurred during the update process: {e}")
            # Clean up the partial new file if it exists
            if new_zip_path.exists():
                os.remove(new_zip_path)
            raise # Re-raise the exception to be caught by the main handler

    # 3. Final atomic replacement
    backup_path = source_zip_path.with_suffix('.zip.bak')
    print("Finalizing update...")
    shutil.move(source_zip_path, backup_path)
    shutil.move(new_zip_path, source_zip_path)

    print("\nUpdate successful!")
    print(f"Original file backed up at: {backup_path}")

def main():
    parser = argparse.ArgumentParser(
        description='''Safely and efficiently update datasets within a Zarr.zip archive.
        This version performs a high-speed, "surgical" in-place update.''',
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # --- Sub-parser for updating a single scalar value ---
    p_scalar = subparsers.add_parser('update-scalar', help='Update or create a key with a single value.')
    p_scalar.add_argument('fn', type=str, help='Zarr.zip filename to update.')
    p_scalar.add_argument('key', type=str, help='Full path to the key inside the Zarr file (e.g., analysis/process/tx).')
    p_scalar.add_argument('value', type=str, help='The new value.')
    p_scalar.add_argument('--dtype', type=str, default='auto', choices=['int', 'float', 'str'], help='Force the data type.')

    # --- Sub-parser for updating an array ---
    p_array = subparsers.add_parser('update-array', help='Update or create a key with an array of values.')
    p_array.add_argument('fn', type=str, help='Zarr.zip filename to update.')
    p_array.add_argument('key', type=str, help='Full path to the key inside the Zarr file.')
    p_array.add_argument('values', nargs='*', help='A flat list of values for the array.')
    p_array.add_argument('--reshape', type=int, nargs='+', help='Shape to reshape the array into (e.g., --reshape 5 2 for a 5x2 array).')
    p_array.add_argument('--dtype', type=str, default='float', choices=['int', 'float'], help='Data type of the array.')

    # --- Sub-parser for updating from a file (e.g., mask) ---
    p_file = subparsers.add_parser('update-from-file', help='Update or create a key with data from an image file.')
    p_file.add_argument('fn', type=str, help='Zarr.zip filename to update.')
    p_file.add_argument('key', type=str, help='Full path to the key inside the Zarr file.')
    p_file.add_argument('source_file', type=str, help='Path to the image file (e.g., a TIFF mask).')

    args = parser.parse_args()
    filepath = Path(args.fn)
    
    if not filepath.exists():
        print(f"Error: Input file not found: {filepath}")
        sys.exit(1)

    # Prepare the new data array based on user input
    new_data = None
    try:
        if args.command == 'update-scalar':
            value_str = args.value
            dtype_str = args.dtype
            if dtype_str == 'auto':
                try: int(value_str); dtype_str = 'int'
                except ValueError:
                    try: float(value_str); dtype_str = 'float'
                    except ValueError: dtype_str = 'str'
            
            if dtype_str == 'int': new_data = np.array([int(value_str)], dtype=np.int32)
            elif dtype_str == 'float': new_data = np.array([float(value_str)], dtype=np.double)
            else: new_data = np.array([np.bytes_(value_str.encode('utf-8'))])

        elif args.command == 'update-array':
            dtype = np.int32 if args.dtype == 'int' else np.double
            flat_array = np.array(args.values, dtype=dtype)
            new_data = flat_array.reshape(args.reshape) if args.reshape else flat_array

        elif args.command == 'update-from-file':
            with Image.open(args.source_file) as img:
                new_data = np.array(img)[np.newaxis, ...].astype(np.uint16)
        
        if new_data is None:
            raise ValueError("Could not prepare new data for the update operation.")

        print(f"  - New data shape: {new_data.shape}, dtype: {new_data.dtype}")
        
        # Execute the update
        surgical_zarr_update(filepath, args.key, new_data)

    except (ValueError, FileNotFoundError, Exception) as e:
        print(f"\nOperation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()