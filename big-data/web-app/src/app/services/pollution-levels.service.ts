import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import * as L from 'leaflet';
import { PollutionForecast } from '../models/pollution-forecast.dto';
import { PopupService } from './popup.service';


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
var yellowIcon = new LeafIcon();
var redIcon = new LeafIcon();
greenIcon.options.iconUrl = 'assets/data/images/green_marker.png';
yellowIcon.options.iconUrl = 'assets/data/images/yellow_marker.png';
redIcon.options.iconUrl = 'assets/data/images/red_marker.png';

@Injectable({
  providedIn: 'root'
})
export class PollutionLevelsService {
    private usCapitals: string = '/assets/data/usa-capitals.geojson';
    private pollutionLevels: string = '/assets/data/pollution-levels.mock.json';

    constructor(private http: HttpClient,
        private popupService: PopupService) {}
  
    public makeCapitalMarkers(map: L.Map): void {
      this.http.get(this.usCapitals).subscribe((res: any) => {
        for (const c of res.features) {
          const lon = c.geometry.coordinates[0];
          const lat = c.geometry.coordinates[1];
          const marker = L.marker([lat, lon], {icon: this.selectMarker(c)});
          
          marker.bindTooltip(this.popupService.makeCapitalPopup(c.properties));

          marker.addTo(map);
        }
      });
    }

    public getPollutionLevels(map: L.Map): void {
        this.http.get(this.pollutionLevels).subscribe((forecasts: PollutionForecast[]) => {
            for (const f of forecasts) {
                const lon = f.longtitude;
                const lat = f.lattitude;
                const marker = L.marker([lat, lon], {icon: this.selectMarker(f)});
                
                marker.bindTooltip(this.popupService.makePollutionPopup(f));

                marker.addTo(map);
            }
        });
    }

    private selectMarker(c: any): L.Icon<L.IconOptions> {
        if (c.properties.name[0]=='M') {
            return greenIcon;
        } else if (c.properties.name[0]=='C') {
            return yellowIcon;
        } else {
            return redIcon;
        }
    }


}
