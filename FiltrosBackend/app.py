import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from PIL import Image
from numba import njit, prange
from io import BytesIO
import time
import math
import traceback  # asegúrate de tenerlo al inicio
import psutil
import os


app = Flask(__name__)
CORS(app, origins=["http://localhost:4200"])
cuda.init()

CHANNELS = 3
PI = math.pi

# ================= CUDA KERNELS =================
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

edge_detect_kernel_code = """
__global__ void edgeDetectKernel(unsigned char *input, unsigned char *output, int width, int height, int channels, float *kernel, int ksize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k2 = ksize / 2;
    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int ky = -k2; ky <= k2; ++ky) {
            for (int kx = -k2; kx <= k2; ++kx) {
                int xx = min(max(x + kx, 0), width - 1);
                int yy = min(max(y + ky, 0), height - 1);
                int img_idx = (yy * width + xx) * channels + c;
                int k_idx = (ky + k2) * ksize + (kx + k2);
                sum += input[img_idx] * kernel[k_idx];
            }
        }
        int out_idx = (y * width + x) * channels + c;
        output[out_idx] = min(max(int(sum), 0), 255);
    }
}
"""


# ================ KERNEL GENERATORS ================
@njit
def generate_log_filter(size, sigma):
    half = size // 2
    mask = np.empty((size, size), dtype=np.float32)
    factor = -1.0 / (PI * sigma**4)
    for j in range(-half, half + 1):
        for i in range(-half, half + 1):
            r2 = i * i + j * j
            expo = math.exp(-r2 / (2.0 * sigma * sigma))
            mask[j + half, i + half] = (
                factor * (1.0 - (r2 / (2.0 * sigma * sigma))) * expo
            )
    return mask


@njit
def generate_edge_kernel(size):
    kernel = np.ones((size, size), dtype=np.float32)
    center = size // 2
    total = size * size
    kernel[center, center] = -1.0 * (total - 1)
    return kernel


@njit
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


# ================ CPU FILTERS ================
@njit(parallel=True)
def cartoon_laplace_cpu_numba(
    input_img, lap_mask, width, height, mask_size, quant_step, threshold
):
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
                        lap += (
                            input_img[(ny * width + nx) * CHANNELS + c]
                            * lap_mask[j + half, i + half]
                        )
                output[base * CHANNELS + c] = 0 if abs(lap) > threshold else quant
    return output


@njit(parallel=True)
def edge_detect_cpu(image, kernel):
    height, width, channels = image.shape
    k_size = kernel.shape[0]
    k_half = k_size // 2
    output = np.empty_like(image)
    for y in prange(height):
        for x in range(width):
            for c in range(channels):
                sum = 0.0
                for ky in range(-k_half, k_half + 1):
                    for kx in range(-k_half, k_half + 1):
                        px = min(max(x + kx, 0), width - 1)
                        py = min(max(y + ky, 0), height - 1)
                        sum += image[py, px, c] * kernel[ky + k_half, kx + k_half]
                output[y, x, c] = min(max(int(sum), 0), 255)
    return output


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


# ================= FLASK ROUTE =================
# Asegúrate de tener un directorio de "downloads" para guardar las imágenes procesadas
UPLOAD_FOLDER = "downloads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
context = None
gpu_info = None
gpu_memory = None

@app.route("/apply_filter", methods=["POST"])
def apply_filter():
    context = None  # Asegúrate de que la variable de contexto esté definida desde el principio
    try:
        # Definimos gpu_info y gpu_memory
        gpu_info = None
        gpu_memory = None

        use_cpu = request.form.get("use_cpu", "true").lower() == "true"

        # Solo si se usa GPU:
        if not use_cpu:
            context = cuda.Device(0).make_context()  # Asegúrate de crear un contexto de GPU válido

        start_time = time.time()  # Comienza a medir el tiempo total
        process = psutil.Process()  # Obtenemos el proceso actual para medir la memoria

        filter_type = request.form.get("filter_type")
        file = request.files["image"]

        # Medir tamaño de la imagen original
        original_image = Image.open(file.stream)
        original_size = os.stat(file.stream.name).st_size  # Tamaño en bytes
        width, height = original_image.size  # Obtener las dimensiones de la imagen

        # Inicializar la variable result
        result = None

        # Fase de aplicación del filtro
        filter_start_time = time.time()  # Comienza a medir el tiempo para aplicar el filtro

        if filter_type == "cartoon_laplace":
            mask_size = int(request.form.get("mask_size", 9))
            sigma = float(request.form.get("sigma", 2.0))
            quant_step = int(request.form.get("quant_step", 64))
            threshold = float(request.form.get("threshold", 20.0))
            np_img = np.array(original_image.convert("RGB"), dtype=np.uint8)
            flat = np_img.ravel()
            lap_mask = generate_log_filter(mask_size, sigma)

            if use_cpu:
                result = cartoon_laplace_cpu_numba(
                    flat, lap_mask, width, height, mask_size, quant_step, threshold
                )
                result = result.reshape((height, width, 3))
            else:
                mod = SourceModule(cartoon_laplace_kernel_code)
                kernel_func = mod.get_function("cartoonLaplaceKernel")
                result = np.empty_like(flat)
                d_in = cuda.mem_alloc(flat.nbytes)
                d_out = cuda.mem_alloc(result.nbytes)
                d_mask = cuda.mem_alloc(lap_mask.nbytes)
                cuda.memcpy_htod(d_in, flat)
                cuda.memcpy_htod(d_mask, lap_mask.ravel())
                block = (16, 16, 1)
                grid = ((width + 15) // 16, (height + 15) // 16)
                kernel_func(
                    d_in,
                    d_out,
                    d_mask,
                    np.int32(width),
                    np.int32(height),
                    np.int32(mask_size),
                    np.int32(quant_step),
                    np.float32(threshold),
                    block=block,
                    grid=grid,
                )
                cuda.memcpy_dtoh(result, d_out)
                result = result.reshape((height, width, 3))

                # Obtener detalles de la GPU dentro del contexto
                gpu_memory = cuda.mem_get_info()  # Memoria disponible y total de la GPU
                gpu_info = cuda.Device(0).name()  # Información de la GPU

        elif filter_type == "emboss":
            kernel_size = int(request.form.get("kernel_size", 9))
            img = Image.open(file.stream).convert("L")
            image = np.array(img).astype(np.uint8)
            kernel = create_emboss_kernel(kernel_size)

            if use_cpu:
                result = apply_convolution_cpu(image, kernel)
            else:
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
                    d_image,
                    d_kernel,
                    d_result,
                    np.int32(img_width),
                    np.int32(img_height),
                    np.int32(kernel_size),
                    block=block,
                    grid=grid,
                )
                cuda.memcpy_dtoh(result, d_result)

            result = normalize_image(result)

        elif filter_type == "edge_detect":
            kernel_size = int(request.form.get("kernel_size", 9))
            img = Image.open(file.stream).convert("RGB")
            np_img = np.array(img, dtype=np.uint8)
            h, w, c = np_img.shape
            kernel = generate_edge_kernel(kernel_size)

            if use_cpu:
                result = edge_detect_cpu(np_img, kernel)
            else:
                flat_img = np_img.ravel()
                result_flat = np.empty_like(flat_img)
                kernel_flat = kernel.ravel().astype(np.float32)
                mod = SourceModule(edge_detect_kernel_code)
                kernel_func = mod.get_function("edgeDetectKernel")
                d_input = cuda.mem_alloc(flat_img.nbytes)
                d_output = cuda.mem_alloc(result_flat.nbytes)
                d_kernel = cuda.mem_alloc(kernel_flat.nbytes)
                cuda.memcpy_htod(d_input, flat_img)
                cuda.memcpy_htod(d_kernel, kernel_flat)
                block = (16, 16, 1)
                grid = ((w + 15) // 16, (h + 15) // 16)
                kernel_func(
                    d_input,
                    d_output,
                    np.int32(w),
                    np.int32(h),
                    np.int32(c),
                    d_kernel,
                    np.int32(kernel_size),
                    block=block,
                    grid=grid,
                )
                cuda.memcpy_dtoh(result_flat, d_output)
                result = result_flat.reshape((h, w, c)).astype(np.uint8)

                # Obtener detalles de la GPU dentro del contexto
                gpu_memory = cuda.mem_get_info()  # Memoria disponible y total de la GPU
                gpu_info = cuda.Device(0).name()  # Información de la GPU

        else:
            return jsonify({"error": "Filtro no implementado."}), 400

        # Si la variable result está definida, guarda la imagen procesada y envía los resultados
        if result is not None:
            # Asegurémonos de que la imagen esté en el modo correcto para guardarla como JPEG
            out_img = Image.fromarray(result)

            if out_img.mode != "RGB":
                out_img = out_img.convert("RGB")  # Convertir a RGB si no está en ese modo

            out_filename = os.path.join(UPLOAD_FOLDER, "processed_image.jpg")
            out_img.save(out_filename)
            processed_size = os.stat(out_filename).st_size  # Tamaño en bytes

            # Calcular el número de píxeles procesados
            total_pixels = width * height  # Total de píxeles procesados

            # Calcular uso de memoria y tiempos
            end_time = time.time()
            execution_time = end_time - start_time  # Tiempo total de ejecución
            memory_usage = process.memory_info().rss / (1024 * 1024)  # Memoria en MB

            # Obtener detalles de la CPU
            cpu_percent = psutil.cpu_percent(interval=1)  # Porcentaje de uso de la CPU

            # Detalles adicionales a enviar
            return jsonify(
                {
                    "execution_time": execution_time,
                    "memory_usage": memory_usage,
                    "original_image_size": original_size,
                    "processed_image_size": processed_size,
                    "image_dimensions": (width, height),  # Dimensiones de la imagen
                    "total_pixels": total_pixels,  # Píxeles procesados
                    "gpu_info": gpu_info,  # Información de la GPU
                    "gpu_memory": gpu_memory,  # Memoria utilizada de la GPU
                    "cpu_usage": cpu_percent,  # Uso de la CPU
                    "image_url": f"/downloads/{os.path.basename(out_filename)}",  # URL de la imagen procesada
                }
            )

        else:
            return jsonify({"error": "No se pudo procesar la imagen."}), 500

    except Exception as e:
        print("ERROR DETECTADO EN /apply_filter:")
        print(traceback.format_exc())  # Imprime el stack completo
        return jsonify({"error": str(e)}), 500

    finally:
        if context:
            context.pop()  # Siempre liberar el contexto de GPU


# Ruta para servir la imagen procesada
@app.route("/downloads/<filename>")
def download_file(filename):
    try:
        return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
