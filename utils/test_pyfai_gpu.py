import os
import time
import numpy as np

# 1. Force PyOpenCL to use a specific device (Platform 0, Device 0)
# This completely replaces the need for the 'opencl_device' kwarg.
# If you have multiple GPUs, you can change this to "0:1", "0:2", etc.
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
os.environ["PYOPENCL_CTX"] = "0:0"

# Import pyFAI AFTER setting the environment variables
import pyopencl as cl
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

# Optional: Print the device to absolutely guarantee it's hitting the H100/A6000
# and not a fallback CPU OpenCL driver.
ctx = cl.create_some_context(interactive=False)
print(f"Active OpenCL Device: {ctx.devices[0].name}")

# 2. Initialize pyFAI integrator
# (You can also use ai = pyFAI.load("your_calib.poni"))
ai = AzimuthalIntegrator(dist=0.90045)

# Then explicitly set the detector properties:
# pixel_size_m = 150 micrometers = 1.5e-4 meters
ai.detector.pixel1 = 1.72e-4
ai.detector.pixel2 = 1.72e-4
ai.detector.shape = (1475, 1679)

# 3. Create dummy data (matching your uint32 testing)
frame = np.random.randint(2000, 40000, size=(1475, 1679), dtype=np.uint32)

# 4. Define the method tuple: (splitting, algorithm, implementation)
method = ("full", "csr", "opencl")

print(f"Warming up pyFAI OpenCL engine with method={method}...")
# The first call compiles the OpenCL C-code for your specific GPU architecture
# and transfers the CSR sparse matrix to device memory.
for _ in range(5):
    res = ai.integrate1d(
        frame,
        npt=1000,           # Set to your desired radial bins
        method=method,
        unit="r_mm"
    )

# 5. Benchmark
print("Starting benchmark...")
n_iterations = 100
start_time = time.perf_counter()

for _ in range(n_iterations):
    res = ai.integrate1d(
        frame,
        npt=1000,
        method=method,
        unit="r_mm"
    )

end_time = time.perf_counter()
total_time = end_time - start_time
fps = n_iterations / total_time
ms_per_frame = (total_time / n_iterations) * 1000

print("-" * 40)
print(f"pyFAI OpenCL (CSR) Throughput: {fps:.2f} fps")
print(f"pyFAI OpenCL (CSR) Latency:    {ms_per_frame:.2f} ms/frame")
print("-" * 40)