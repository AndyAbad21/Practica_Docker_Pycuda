import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from PIL import Image
from numba import njit, prange
from io import BytesIO
import time
import math

# Inicializar Flask
app = Flask(__name__)
CORS(app, origins=["http://localhost:4200"])
cuda.init()

# Constantes
CHANNELS = 3
PI = math.pi

cartoon_laplace_kernel_code = """
#define CHANNELS 3
__global__ void cartoonLaplaceKernel(unsigned char* input, unsigned char* output, float* lapMask,
                                     int width, int height, int maskSize, int quantStep, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int half = maskSize / 2;
    int base = (y * width + x) * CHANNELS;
    for (int c = 0; c < CHANNELS; ++c) {
        int idx = base + c;
        int origVal = input[idx];
        int quant = ((origVal / quantStep) * quantStep) + (quantStep / 2);
        quant = max(0, min(255, quant));
        float lap = 0.0f;
        for (int j = -half; j <= half; ++j) {
            for (int i = -half; i <= half; ++i) {
                int nx = min(max(x + i, 0), width - 1);
                int ny = min(max(y + j, 0), height - 1);
                lap += input[(ny * width + nx) * CHANNELS + c] * lapMask[(j + half) * maskSize + (i + half)];
            }
        }
        output[idx] = (fabsf(lap) > threshold) ? 0 : (unsigned char)quant;
    }
}
"""

@njit
def generate_log_filter(size, sigma):
    half = size // 2
    mask = np.empty((size, size), dtype=np.float32)
    factor = -1.0 / (PI * sigma ** 4)
    for j in range(-half, half + 1):
        for i in range(-half, half + 1):
            r2 = i * i + j * j
            expo = math.exp(-r2 / (2.0 * sigma * sigma))
            mask[j + half, i + half] = factor * (1.0 - (r2 / (2.0 * sigma * sigma))) * expo
    return mask

@njit(parallel=True)
def cartoon_laplace_cpu_numba(input_img, lap_mask, width, height, mask_size, quant_step, threshold):
    output = np.empty_like(input_img)
    half = mask_size // 2
    for y in prange(height):
        for x in range(width):
            base = y * width + x
            for c in range(CHANNELS):
                pix_val = input_img[base * CHANNELS + c]
                quant = ((pix_val // quant_step) * quant_step) + (quant_step // 2)
                quant = max(0, min(255, quant))
                lap = 0.0
                for j in range(-half, half + 1):
                    for i in range(-half, half + 1):
                        nx = min(max(x + i, 0), width - 1)
                        ny = min(max(y + j, 0), height - 1)
                        lap += input_img[(ny * width + nx) * CHANNELS + c] * lap_mask[j + half, i + half]
                output[base * CHANNELS + c] = 0 if abs(lap) > threshold else quant
    return output

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
def apply_filter():
    try:
        filter_type = request.form.get("filter_type", "emboss")
        use_cpu = request.form.get("use_cpu", "false").lower() == "true"
        file = request.files["image"]

        if filter_type == "cartoon_laplace":
            mask_size = int(request.form.get("mask_size", 9))
            sigma = float(request.form.get("sigma", 2.0))
            quant_step = int(request.form.get("quant_step", 64))
            threshold = float(request.form.get("threshold", 20.0))

            img = Image.open(file.stream).convert("RGB")
            np_img = np.array(img, dtype=np.uint8)
            flat = np_img.ravel()
            width, height = img.size
            lap_mask = generate_log_filter(mask_size, sigma)

            if use_cpu:
                result = cartoon_laplace_cpu_numba(flat, lap_mask, width, height, mask_size, quant_step, threshold)
            else:
                context = cuda.Device(0).make_context()
                try:
                    mod = SourceModule(cartoon_laplace_kernel_code)
                    cartoon_laplace_gpu = mod.get_function("cartoonLaplaceKernel")

                    result = np.empty_like(flat)
                    d_in = cuda.mem_alloc(flat.nbytes)
                    d_out = cuda.mem_alloc(result.nbytes)
                    d_mask = cuda.mem_alloc(lap_mask.nbytes)

                    cuda.memcpy_htod(d_in, flat)
                    cuda.memcpy_htod(d_mask, lap_mask.ravel())

                    block = (16, 16, 1)
                    grid = ((width + 15) // 16, (height + 15) // 16)

                    cartoon_laplace_gpu(
                        d_in, d_out, d_mask,
                        np.int32(width), np.int32(height),
                        np.int32(mask_size), np.int32(quant_step), np.float32(threshold),
                        block=block, grid=grid
                    )
                    cuda.memcpy_dtoh(result, d_out)
                    d_in.free(); d_out.free(); d_mask.free()
                finally:
                    context.pop()

            out_img = Image.fromarray(result.reshape((height, width, 3)).astype(np.uint8))

        else:
            kernel_size = int(request.form.get("kernel_size", 9))
            img = Image.open(file.stream).convert("L")
            image = np.array(img).astype(np.uint8)
            kernel = create_emboss_kernel(kernel_size)

            if use_cpu:
                result = apply_convolution_cpu(image, kernel)
            else:
                context = cuda.Device(0).make_context()
                try:
                    img_height, img_width = image.shape
                    result = np.empty((img_height, img_width), dtype=np.float64)
                    d_image = cuda.mem_alloc(image.nbytes)
                    d_kernel = cuda.mem_alloc(kernel.nbytes)
                    d_result = cuda.mem_alloc(result.nbytes)

                    cuda.memcpy_htod(d_image, image)
                    cuda.memcpy_htod(d_kernel, kernel)

                    mod = SourceModule("""
                    __global__ void applyConvolutionGPU(unsigned char* image, double* kernel, double* result, int width, int height, int ksize) {
                        int x = blockIdx.x * blockDim.x + threadIdx.x;
                        int y = blockIdx.y * blockDim.y + threadIdx.y;
                        int half = ksize / 2;
                        if (x >= half && x < width - half && y >= half && y < height - half) {
                            double sum = 0.0;
                            for (int ky = -half; ky <= half; ky++) {
                                for (int kx = -half; kx <= half; kx++) {
                                    int px = x + kx;
                                    int py = y + ky;
                                    int img_idx = py * width + px;
                                    int ker_idx = (ky + half) * ksize + (kx + half);
                                    sum += image[img_idx] * kernel[ker_idx];
                                }
                            }
                            result[y * width + x] = sum;
                        }
                    }
                    """)
                    func = mod.get_function("applyConvolutionGPU")
                    block = (16, 16, 1)
                    grid = ((img_width + 15) // 16, (img_height + 15) // 16)
                    func(
                        d_image, d_kernel, d_result,
                        np.int32(img_width), np.int32(img_height), np.int32(kernel_size),
                        block=block, grid=grid
                    )
                    cuda.memcpy_dtoh(result, d_result)
                    d_image.free(); d_kernel.free(); d_result.free()
                finally:
                    context.pop()

            norm_img = normalize_image(result)
            out_img = Image.fromarray(norm_img)

        buffer = BytesIO()
        out_img.save(buffer, format="JPEG")
        buffer.seek(0)
        return send_file(buffer, mimetype="image/jpeg", download_name="result.jpg")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
