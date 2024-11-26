#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

# Starting point of grid
phiStart =0 #2**-7 # Must be a power of 2.

# Define zero numerically
zero = 1e-20

# Set default dpi for plotting figures
dpi = 300

# Get data/paper output directories
dir_data = os.environ.get('DATA')
if not dir_data: dir_data = "data/"
dir_paper = "paper/"
