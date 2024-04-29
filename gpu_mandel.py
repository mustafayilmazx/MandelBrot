import numpy as np
from PIL import Image
from numba import jit, cuda

# Function to compute Mandelbrot set, optimized for GPU
@cuda.jit(device=True)
def mandelbrot_gpu(c, max_iter):
    z = 0.0j
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return 0

# Kernel function to set up and compute Mandelbrot set on the GPU
@cuda.jit
def mandelbrot_kernel(min_x, max_x, min_y, max_y, image, iters):
    height, width = image.shape
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    
    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    for x in range(startX, width, gridX):
        real = min_x + x * pixel_size_x
        for y in range(startY, height, gridY):
            imag = min_y + y * pixel_size_y
            color = mandelbrot_gpu(complex(real, imag), iters)
            image[y, x] = color

def create_mandelbrot_image_gpu(xmin, xmax, ymin, ymax, width, height, max_iter, filename):
    image = np.zeros((height, width), dtype=np.uint8)
    # Configure the blocks
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(width / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(height / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    # Start kernel
    mandelbrot_kernel[blockspergrid, threadsperblock](xmin, xmax, ymin, ymax, image, max_iter)
    img = Image.fromarray(image)
    img.save(filename)

# Parameters
xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
img_width, img_height = 16000, 12000
max_iter = 64000  # Maximum iterations increased to 1023
filename = "output_mandelbrot_gpu.png"

# Create and save the image using GPU
create_mandelbrot_image_gpu(xmin, xmax, ymin, ymax, img_width, img_height, max_iter, filename)
