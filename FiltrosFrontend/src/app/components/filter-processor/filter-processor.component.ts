import { Component } from '@angular/core';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  standalone: true,
  selector: 'app-filter-processor',
  imports: [CommonModule, FormsModule, HttpClientModule],
  templateUrl: './filter-processor.component.html',
  styleUrls: ['./filter-processor.component.scss']
})

export class FilterProcessorComponent {
  // Variables para la imagen y el filtro
  previewImage: string | null = null;
  selectedFile: File | null = null;
  selectedFilter = 'emboss';
  kernelSize = 9;
  blockX = 16;
  blockY = 16;
  gridX = 0;
  gridY = 0;
  useCPU = false;
  resultImage: string | null = null;
  kernelWarning = false;
  showFiltered = false;

  // Estado de carga
  isLoading: boolean = false;

  // Parámetros del filtro
  sigma: number = 2.0;
  quantStep: number = 64;
  threshold: number = 20.0;

  // Nuevas variables para almacenar la información adicional
  executionTime: number | null = null;
  filterExecutionTime: number | null = null;
  memoryUsage: number | null = null;
  originalImageSize: number | null = null;
  processedImageSize: number | null = null;
  totalPixels: number | null = null;
  gpuInfo: string | null = null;
  gpuMemory: [number, number] | null = null;
  cpuUsage: number | null = null;
  imageDimensions: [number, number] | null = null;

  filters = [
    { value: 'emboss', label: 'Filtro Emboss' },
    { value: 'edges', label: 'Detección de Bordes' },
    { value: 'cartoon', label: 'Filtro Cartoon' }
  ];

  constructor(private http: HttpClient) {
    // Cargar imagen procesada y datos iniciales (por defecto)
    // this.fetchDefaultData();
  }

  updateGridValues(){
    const width = this.imageDimensions?.[0];
    const height = this.imageDimensions?.[1];

    if (width && height) {
      this.gridX = Math.ceil((width + this.blockX - 1) / this.blockX);
      this.gridY = Math.ceil((height + this.blockY - 1) / this.blockY);
    }
  }

  // Función para seleccionar la imagen
  onFileSelected(event: any) {
    this.selectedFile = event.target.files[0];

    if (this.selectedFile) {
      const reader = new FileReader();
      reader.onload = (e: any) => {
        const image = new Image();
        image.onload = () => {
          const width = image.width;
          const height = image.height;

          this.imageDimensions = [width,height];

          this.gridX = Math.ceil((width+this.blockX-1)/this.blockX);
          this.gridY = Math.ceil((height+this.blockY-1)/this.blockY);
        
          this.previewImage = e.target.result;
        };
        image.src = e.target.result;
      };
      reader.readAsDataURL(this.selectedFile);
    } else {
      this.previewImage = null;
    }
  }

  // Función para alternar la vista entre la imagen original y la filtrada
  toggleImage() {
    this.showFiltered = !this.showFiltered;
  }

  // Función para validar el tamaño del kernel
  validateKernelSize() {
    const n = Number(this.kernelSize);
    this.kernelWarning = n < 3 || n % 2 === 0;
  }

  // Función que maneja el envío del formulario y realiza la solicitud al backend
  onSubmit(event: Event) {
    event.preventDefault();

    // Validación
    if (this.kernelSize < 3 || this.kernelSize % 2 === 0) {
      this.kernelWarning = true;
      return;
    }

    if (!this.selectedFile) {
      alert("Debes seleccionar una imagen.");
      return;
    }

    // Limpiar el estado de la imagen procesada antes de enviar la solicitud
    this.resultImage = null;  // Limpiar la imagen procesada
    this.showFiltered = false;  // Restablecer la alternancia a "Original"

    // Activar loader
    this.isLoading = true;

    const formData = new FormData();
    formData.append('image', this.selectedFile);

    const backendFilterType =
      this.selectedFilter === 'cartoon'
        ? 'cartoon_laplace'
        : this.selectedFilter === 'edges'
          ? 'edge_detect'
          : this.selectedFilter;

    formData.append('filter_type', backendFilterType);
    formData.append('use_cpu', this.useCPU.toString());

    // ⬇️ Agregar los nuevos parámetros de hilos y bloques
    formData.append('block_x', this.blockX.toString());
    formData.append('block_y', this.blockY.toString());
    formData.append('grid_x', this.gridX.toString());
    formData.append('grid_y', this.gridY.toString());

    if (this.selectedFilter === 'emboss' || this.selectedFilter === 'edges') {
      formData.append('kernel_size', this.kernelSize.toString());
    }

    if (this.selectedFilter === 'cartoon') {
      formData.append('mask_size', this.kernelSize.toString());
      formData.append('sigma', this.sigma.toString());
      formData.append('quant_step', this.quantStep.toString());
      formData.append('threshold', this.threshold.toString());
    }

    this.http.post<any>('http://localhost:5000/apply_filter', formData).subscribe({
      next: (res) => {
        this.resultImage = 'http://localhost:5000' + res.image_url + '?timestamp=' + new Date().getTime();
        this.executionTime = res.execution_time;
        this.filterExecutionTime = res.filter_execution_time;
        this.memoryUsage = res.memory_usage;
        this.originalImageSize = res.original_image_size;
        this.processedImageSize = res.processed_image_size;
        this.gpuInfo = res.gpu_info;
        this.gpuMemory = res.gpu_memory;
        this.cpuUsage = res.cpu_usage;
        this.imageDimensions = res.image_dimensions;
        this.totalPixels = res.total_pixels;

        // ✅ Mostrar imagen filtrada automáticamente
        this.showFiltered = true;
      },
      error: (err) => {
        console.error('Error al aplicar filtro:', err);
      },
      complete: () => {
        // Desactivar loader
        this.isLoading = false;
      }
    });
  }
}