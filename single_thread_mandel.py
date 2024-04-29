import numpy as np
from PIL import Image

def mandelbrot(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    if n == max_iter:
        return 255  # Mandelbrot setinin bir üyesi olarak kabul edilir
    else:
        return n % 256  # Hızlı kaçış yoluyla elde edilen değer

def mandelbrot_set(xmin, xmax, ymin, ymax, img_width, img_height, max_iter):
    r1 = np.linspace(xmin, xmax, img_width)
    r2 = np.linspace(ymin, ymax, img_height)
    return (255 - np.array([[mandelbrot(complex(r, i), max_iter) for r in r1] for i in r2])).astype(np.uint8)

def create_mandelbrot_image(xmin, xmax, ymin, ymax, img_width, img_height, max_iter, filename):
    # Mandelbrot setini hesapla
    image = mandelbrot_set(xmin, xmax, ymin, ymax, img_width, img_height, max_iter)
    # Görüntüyü kaydet
    img = Image.fromarray(image, mode="L")
    img.save(filename)

# Görüntü parametreleri
xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
img_width, img_height = 2000, 1500
max_iter = 255
filename = "output_single_thread.png"

# Görüntüyü oluştur ve kaydet
create_mandelbrot_image(xmin, xmax, ymin, ymax, img_width, img_height, max_iter, filename)
