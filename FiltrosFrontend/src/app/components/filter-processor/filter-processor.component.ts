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
  useCPU = false;
  resultImage: string | null = null;
  kernelWarning = false;
  showFiltered = false;

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
  gpuMemory: any | null = null;
  cpuUsage: number | null = null;
  imageDimensions: [number, number] | null = null;

  filters = [
    { value: 'emboss', label: 'Filtro Emboss' },
    { value: 'edges', label: 'Detección de Bordes' },
    { value: 'cartoon', label: 'Filtro Cartoon' }
  ];

  constructor(private http: HttpClient) { }

  // Función para seleccionar la imagen
  onFileSelected(event: any) {
    this.selectedFile = event.target.files[0];

    if (this.selectedFile) {
      const reader = new FileReader();
      reader.onload = (e: any) => {
        this.previewImage = e.target.result;
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

    // Validar el tamaño del kernel
    if (this.kernelSize < 3 || this.kernelSize % 2 === 0) {
      this.kernelWarning = true;
      return;
    }

    // Validar si se ha seleccionado una imagen
    if (!this.selectedFile) {
      alert("Debes seleccionar una imagen.");
      return;
    }

    // Crear un FormData para enviar la imagen y los parámetros del filtro al backend
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

    if (this.selectedFilter === 'emboss' || this.selectedFilter === 'edges') {
      formData.append('kernel_size', this.kernelSize.toString());
    }

    if (this.selectedFilter === 'cartoon') {
      formData.append('mask_size', this.kernelSize.toString());
      formData.append('sigma', this.sigma.toString());
      formData.append('quant_step', this.quantStep.toString());
      formData.append('threshold', this.threshold.toString());
    }

    // Realizar la solicitud POST al backend para procesar la imagen
    this.http.post<any>(`http://localhost:5000/apply_filter`, formData, { responseType: 'json' })
      .subscribe(response => {
        // Usamos la URL de la imagen procesada devuelta en la respuesta
        const timestamp = new Date().getTime();
        this.resultImage = `http://localhost:5000${response.image_url}?t=${timestamp}`;  // Nueva URL con timestamp

        // Mostrar la información adicional
        this.executionTime = response.execution_time;
        this.filterExecutionTime = response.filter_execution_time;
        this.memoryUsage = response.memory_usage;
        this.originalImageSize = response.original_image_size;
        this.processedImageSize = response.processed_image_size;
        this.totalPixels = response.total_pixels;
        this.gpuInfo = response.gpu_info;
        this.gpuMemory = response.gpu_memory;
        this.cpuUsage = response.cpu_usage;
        this.imageDimensions = response.image_dimensions;

        // Mostrar la imagen filtrada
        this.showFiltered = true;
      }, error => {
        alert('Error al procesar la imagen: ' + (error.error?.error || error.message));
      });
  }
}
