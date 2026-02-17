#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = []
    for fname in os.listdir(folder):
        try:
            # Try to parse the iteration number from the filename
            saved_iters.append(int(fname.split("_")[-1]))
        except (ValueError, IndexError):
            # Skip files that don't match the expected pattern
            continue
    if not saved_iters:
        return 0
    return max(saved_iters)
