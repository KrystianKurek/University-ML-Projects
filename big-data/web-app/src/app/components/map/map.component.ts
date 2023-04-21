import { AfterViewInit, Component, ViewChild } from '@angular/core';
import * as L from 'leaflet';
import { Subscription, timer } from 'rxjs';
import { map } from 'rxjs/operators';
import { CassandraService } from 'src/app/services/cassandra.service';
import { PollutionLevelsService } from 'src/app/services/pollution-levels.service';

const iconRetinaUrl = 'assets/data/images/green_marker.png';
const iconUrl = 'assets/data/images/green_marker.png';
const shadowUrl = 'assets/marker-shadow.png';

var LeafIcon = L.Icon.extend({
    options: {
        shadowUrl: 'assets/marker-shadow.png',
        iconSize: [35, 35],
        iconAnchor: [12, 41],
        popupAnchor: [1, -34],
        tooltipAnchor: [16, -28],
        shadowSize: [41, 41]
    }
});

var greenIcon = new LeafIcon();
greenIcon.iconUrl = 'assets/data/images/yellow_marker.png';
var yellowIcon = new LeafIcon();
yellowIcon.iconUrl = 'assets/data/images/yellow_marker.png'; 
var redIcon = new LeafIcon();
redIcon.iconUrl = 'assets/data/images/red_marker.png';


const iconDefault = L.icon({
  iconRetinaUrl,
  iconUrl,
  shadowUrl,
  iconSize: [35, 35],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  tooltipAnchor: [16, -28],
  shadowSize: [41, 41]
});

L.Marker.prototype.options.icon = iconDefault;

@Component({
    selector: 'app-map',
    templateUrl: './map.component.html',
    styleUrls: ['./map.component.scss']
})
export class MapComponent implements AfterViewInit {
    @ViewChild('map', {static: false})
    private map;
    public timerSubscription: Subscription; 

    private initMap(): void {
        this.map = L.map('map', {
          center: [ 52.2, 19 ],
          zoom: 6
        });
    
        const tiles = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          maxZoom: 18,
          minZoom: 3,
          attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        });
    
        tiles.addTo(this.map);
      }

    constructor(private pollutionLevelsService: PollutionLevelsService,
        private cassandraService: CassandraService) {}

    ngAfterViewInit() {
        this.initMap();
        this.timerSubscription = timer(0, 20000).pipe( 
            map(() => { 
              this.getPredictionsData(); // load data contains the http request 
            }) 
          ).subscribe(); 
    }

    public async getPredictionsData(): Promise<void> {
        //await this.cassandraService.insert();
        this.cassandraService.getPredictions(this.map);
    }
}