# ... (importaciones igual que antes)

import pycuda.driver as cuda
import pycuda.tools
from pycuda.compiler import SourceModule  # ✅ IMPORTANTE
cuda.init()  # ✅ Solo una vez

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # ✅ Importa CORS aquí
import numpy as np
from PIL import Image
from numba import njit, prange
import time
from io import BytesIO

app = Flask(__name__)

CORS(app, origins=["http://localhost:4200"])

def create_emboss_kernel(kernel_size):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    half = kernel_size // 2
    for y in range(kernel_size):
        for x in range(kernel_size):
            if x < half and y < half:
                kernel[y, x] = -1
            elif (x > half and y > half) or (x == half and y == half):
                kernel[y, x] = 1
    return kernel


@njit(parallel=True)
def apply_convolution_cpu(image, kernel):
    height, width = image.shape
    k_size = kernel.shape[0]
    half = k_size // 2
    result = np.zeros((height, width), dtype=np.float64)
    for y in prange(half, height - half):
        for x in range(half, width - half):
            sum_val = 0.0
            for ky in range(-half, half + 1):
                for kx in range(-half, half + 1):
                    pixel = image[y + ky, x + kx]
                    weight = kernel[ky + half, kx + half]
                    sum_val += pixel * weight
            result[y, x] = sum_val
    return result


def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return ((image - min_val) / (max_val - min_val) * 255).clip(0, 255).astype(np.uint8)


@app.route("/apply_filter", methods=["POST"])
def apply_emboss():
    try:
        kernel_size = int(request.form.get("kernel_size", 9))
        use_cpu = request.form.get("use_cpu", "false").lower() == "true"

        file = request.files["image"]
        img = Image.open(file.stream).convert("L")
        image = np.array(img).astype(np.uint8)  # Asegurar tipo correcto

        kernel = create_emboss_kernel(kernel_size)

        if use_cpu:
            start = time.time()
            result = apply_convolution_cpu(image, kernel)
            end = time.time()
            elapsed = (end - start) * 1000
        else:
            device = cuda.Device(0)
            context = device.make_context()

            try:
                img_height, img_width = image.shape
                img_bytes = image.nbytes
                kernel_bytes = kernel.nbytes
                result_bytes = img_height * img_width * np.dtype(np.float64).itemsize

                d_image = cuda.mem_alloc(img_bytes)
                d_kernel = cuda.mem_alloc(kernel_bytes)
                d_result = cuda.mem_alloc(result_bytes)

                cuda.memcpy_htod(d_image, image)
                cuda.memcpy_htod(d_kernel, kernel)

                mod = SourceModule(f"""
                __global__ void applyConvolutionGPU(unsigned char* image, double* kernel, double* result, int width, int height, int ksize)
                {{
                    int x = blockIdx.x * blockDim.x + threadIdx.x;
                    int y = blockIdx.y * blockDim.y + threadIdx.y;
                    int half = ksize / 2;

                    if (x >= half && x < width - half && y >= half && y < height - half)
                    {{
                        double sum = 0.0;
                        for (int ky = -half; ky <= half; ky++)
                        {{
                            for (int kx = -half; kx <= half; kx++)
                            {{
                                int px = x + kx;
                                int py = y + ky;
                                if (px >= 0 && px < width && py >= 0 && py < height)
                                {{
                                    int img_idx = py * width + px;
                                    int ker_idx = (ky + half) * ksize + (kx + half);
                                    sum += image[img_idx] * kernel[ker_idx];
                                }}
                            }}
                        }}
                        result[y * width + x] = sum;
                    }}
                }}
                """)

                func = mod.get_function("applyConvolutionGPU")
                block = (16, 16, 1)
                grid = (
                    (img_width + block[0] - 1) // block[0],
                    (img_height + block[1] - 1) // block[1],
                )

                start = cuda.Event()
                end = cuda.Event()
                start.record()
                func(
                    d_image,
                    d_kernel,
                    d_result,
                    np.int32(img_width),
                    np.int32(img_height),
                    np.int32(kernel_size),
                    block=block,
                    grid=grid,
                )
                end.record()
                end.synchronize()
                elapsed = start.time_till(end)

                result = np.empty((img_height, img_width), dtype=np.float64)
                cuda.memcpy_dtoh(result, d_result)

            finally:
                context.pop()

        norm_img = normalize_image(result)
        out_img = Image.fromarray(norm_img)
        buffer = BytesIO()
        out_img.save(buffer, format="JPEG")
        buffer.seek(0)

        return send_file(buffer, mimetype="image/jpeg", download_name="emboss_result.jpg")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
