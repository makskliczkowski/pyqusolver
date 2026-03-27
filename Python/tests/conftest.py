import os

# Set matplotlib backend to Agg to avoid segmentation faults in headless environments
import matplotlib
matplotlib.use('Agg')

# Disable JAX for tests to avoid NameError from missing JAX_AVAILABLE definition in ran_wrapper.py
os.environ["PY_JAX_DONT_USE"] = "1"
