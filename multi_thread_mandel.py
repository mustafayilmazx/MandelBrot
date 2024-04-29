
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

# Adjusted compute_row function to reflect the color mapping from the GPU code
def compute_row(y, xmin, xmax, width, max_iter):
    x_values = np.linspace(xmin, xmax, width)
    c = x_values + 1j*y
    output = np.zeros(c.shape)
    z = np.zeros(c.shape, np.complex64)
    for i in range(max_iter):
        mask = np.abs(z) <= 2
        z[mask] = z[mask]**2 + c[mask]
        output[mask] = i
    # Points inside the set will be black (0), outside will be based on iteration number
    output = np.where(np.abs(z) <= 2, 0, output)
    # Normalize the escape time to the range of 0-255 for color mapping
    normalized_output = np.round(output / max_iter * 255)
    return normalized_output.astype(np.uint8)

def create_mandelbrot_image(xmin, xmax, ymin, ymax, width, height, max_iter, filename):
    image = np.zeros((height, width), dtype=np.uint8)
    y_values = np.linspace(ymin, ymax, height)
    with ProcessPoolExecutor() as executor:
        results = executor.map(compute_row, y_values, [xmin]*height, [xmax]*height, [width]*height, [max_iter]*height)
    for i, row in enumerate(results):
        image[i] = row
    img = Image.fromarray(image, mode="L")
    img.save(filename)

if __name__ == '__main__':
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    img_width, img_height = 16000, 12000
    max_iter = 1023
    filename = "output_multithreaded.png"

    create_mandelbrot_image(xmin, xmax, ymin, ymax, img_width, img_height, max_iter, filename)
