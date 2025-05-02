import { Component } from '@angular/core';
import { FilterProcessorComponent } from './components/filter-processor/filter-processor.component';

@Component({
  standalone: true,
  selector: 'app-root',
  imports: [FilterProcessorComponent],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'FiltrosFrontend';
}
