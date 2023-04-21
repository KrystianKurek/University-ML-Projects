import { Injectable } from '@angular/core';
import { TranslateService } from '@ngx-translate/core';
import { StationDataModel } from '../models/station-data.model';

@Injectable({
  providedIn: 'root'
})
export class PopupService {
    constructor(private translate: TranslateService) { }

    public makeCapitalPopup(data: any
        ): string {
        // return `` +
        // `<table>` +
        //     `<tr>`+
        //         `<th colspan="2" style="font-weight: 700; padding:5px;">` + "Legionowo" + "<br>" + "22-12-2022 17:05:78" + `</th>` +
        //     `</tr>`+
        //     `<tr>` +
        //         `<td style="width: 300px"> Capital: ` + `</td><td style="width: 300px">` + data.name + `</td>` + 
        //     `</tr>` +
        //     `<tr>` +
        //         `<td> State: ` + `</td><td>` + data.state + `</td>` + 
        //     `</tr>` +
        //     `<tr>` +
        //         `<td> Population: ` + `</td><td>` + data.population + `</td>` + 
        //     `</tr>` +
        // `</table>`;
        return `` +
        `<table>` +
            `<tr>`+
                `<th colspan="2" style="font-weight: 700; padding:5px;">` + 'Kansas City' + "<br>" +"22-12-2022 17:05:78" + `</th>` +
            `</tr>`+
            `<tr>` +
                `<td> PM10: ` + `</td><td>` + '110%' + `</td>` + 
            `</tr>` +
            `<tr>` +
                `<td> PM2.5: ` + `</td><td>` + '140%' + `</td>` + 
            `</tr>` +
            `<tr>` +
                `<td> NO2: ` + `</td><td>` + '50%' + `</td>` + 
            `</tr>` +
            `<tr>` +
                `<td> O3: ` + `</td><td>` + '120%' + `</td>` + 
            `</tr>` +
        `</table>`;
    }

    public makePollutionPopup(f: StationDataModel): string {
        return `` +
        `<table>` +
            `<tr>`+
                `<th colspan="2" style="font-weight: 700; padding:5px;">` + f.station_name + "<br>" + f.timestamp + `</th>` +
            `</tr>`+
            `<tr>` +
                `<td> PM10: ` + `</td><td>` + `${this.checkIfUndefined(f.predictions.pm10)}` + `</td>` + 
            `</tr>` +
            `<tr>` +
                `<td> PM2.5: ` + `</td><td>` + `${this.checkIfUndefined(f.predictions.pm25)}` + `</td>` + 
            `</tr>` +
            `<tr>` +
                `<td> NO2: ` + `</td><td>` + `${this.checkIfUndefined(f.predictions.no2)}` + `</td>` + 
            `</tr>` +
            `<tr>` +
                `<td> O3: ` + `</td><td>` + `${this.checkIfUndefined(f.predictions.o3)}` + `</td>` + 
            `</tr>` +
        `</table>`;
    }

    private checkIfUndefined(x: any) {
        return x != null ? x : this.translate.instant("notProvided");
    }
}
