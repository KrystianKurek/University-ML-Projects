import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Injectable } from '@angular/core';
import * as L from 'leaflet';
import { BatchViewModel } from '../models/batch-view.model';
import { PollutionForecast } from '../models/pollution-forecast.dto';
import { PredictionLevels } from '../models/prediction-levels';
import { StationDataModel } from '../models/station-data.model';
import { PopupService } from './popup.service';

var LeafIcon = L.Icon.extend({
    options: {
        shadowUrl: 'assets/marker-shadow.png',
        iconSize: [37, 37],
        iconAnchor: [12, 41],
        popupAnchor: [10, -14],
        tooltipAnchor: [16, -28],
        shadowSize: [33, 33]
    }
});

var greenIcon = new LeafIcon();
var yellowIcon = new LeafIcon();
var orangeIcon = new LeafIcon();
var redIcon = new LeafIcon();
var brownIcon = new LeafIcon();
greenIcon.options.iconUrl = 'assets/data/images/green_marker.png';
yellowIcon.options.iconUrl = 'assets/data/images/yellow_marker.png';
orangeIcon.options.iconUrl = 'assets/data/images/orange_marker.png';
redIcon.options.iconUrl = 'assets/data/images/red_marker.png';
brownIcon.options.iconUrl = 'assets/data/images/brown_marker.png';
@Injectable({
  providedIn: 'root'
})
export class CassandraService {
    //protected static readonly ServiceUrl: string = 'http://172.18.0.1:5001/';

	protected static readonly ServiceUrl: string = 'http://localhost:5001/';
    private mockPredictions: string[] = ['/assets/data/pollution-levels.mock.json', 
        '/assets/data/pollution-levels-2.mock.json',
        '/assets/data/pollution-levels-3.mock.json'];

    private mockBatchViews: string = '/assets/data/batch.json';

    constructor(private http: HttpClient,
        private popupService: PopupService) {

    }
    
    public async insert() {
        console.log('Insert!');

        return await this.httpPost<any, any>(`insert`, null);
    }

    public async getPredictions(map: L.Map) {

        let predictions = this.httpGet<any>(`predictions`, null).then((forecasts: any[]) => {

            forecasts = forecasts.map(x => new PollutionForecast(x[0], x[1], x[2], x[3], x[4], x[5]));

            var groupBy = function(xs, key) {
                return xs.reduce(function(rv, x) {
                  (rv[x[key]] = rv[x[key]] || []).push(x);
                  return rv;
                }, {});
              };
              
            let groupedForecasts = groupBy(forecasts, 'station_name');

            let listOfForecasts = [];

            for (let key in groupedForecasts) {
                let stationData: PollutionForecast[] = groupedForecasts[key];

                let stationModel: StationDataModel = {
                    station_name: stationData[0].station_name,
                    longtitude: stationData[0].longtitude,
                    lattitude: stationData[0].lattitude,
                    timestamp: stationData[0].timestamp,
                    predictions: new PredictionLevels()
                }

                for (let particleData in stationData) {
                    
                    if (stationData[particleData].particle == "NO2") {
                        stationModel.predictions.no2 = stationData[particleData].prediction;
                    } else if (stationData[particleData].particle == "O3") {
                        stationModel.predictions.o3 = stationData[particleData].prediction;
                    } else if (stationData[particleData].particle == "PM10") {
                        stationModel.predictions.pm10 = stationData[particleData].prediction;
                    } else if (stationData[particleData].particle == "PM2.5") {
                        stationModel.predictions.pm25 = stationData[particleData].prediction;
                    }
                }

                listOfForecasts.push(stationModel);
            }
            
            for (const f of listOfForecasts) {
                const lon = f.longtitude;
                const lat = f.lattitude;
                const marker = L.marker([lat, lon], {icon: this.selectMarker(f)});
                
                marker.bindTooltip(this.popupService.makePollutionPopup(f));

                marker.addTo(map);
            }
        });
    }

    public async getMREs(): Promise<BatchViewModel[]> {
        let mres: BatchViewModel[];
        await this.httpGet<any>(`batch_views`, null)
            .then((response: any[]) => {
                mres = response.map(x => new BatchViewModel(x[0], x[1], x[2], x[3]))
            })
            .catch();
        return mres;
    }
    

    private selectMarker(stationData: StationDataModel): L.Icon<L.IconOptions> {
        let measures = [stationData.predictions.no2, 
            stationData.predictions.o3, 
            stationData.predictions.pm10, 
            stationData.predictions.pm25]
                .filter(x => x != null || x != undefined)
        
                let maxLevel = measures.sort((val1,val2)=> { return (val1 < val2 ) ? 1 : -1 })[0]

        if (maxLevel < 15) {
            return greenIcon;
        } else if (maxLevel < 30) {
            return yellowIcon;
        } else if (maxLevel < 45) {
            return orangeIcon;
        } else if (maxLevel < 60) {
            return redIcon;
        } else {
            return brownIcon;
        }
    }

    protected async httpGet<TResponse>(
		endpoint: string,
		params?: any
	): Promise<TResponse> {
		const httpParams = this.getRequestParams(params);
		const composedUrl = `${this.serviceUrl}${endpoint}`;
		const response = await this.http
			.get<TResponse>(`${this.serviceUrl}${endpoint}`, {
			headers: this.getRequestHeaders(),
			params: httpParams,
			})
			.toPromise<TResponse>();

		return response;
	}

	protected async httpPost<TResponse, TBody>(
		endpoint: string,
		object: TBody,
		params?: any
	): Promise<TResponse> {
		const httpParams = this.getRequestParams(params);

		const response = await this.http
			.post<TResponse>(`${this.serviceUrl}${endpoint}`, object, {
                headers: this.getRequestHeaders(),
                params: httpParams,
			})
			.toPromise<TResponse>();

		return response;
	}

    protected get serviceUrl(): string {
		return CassandraService.ServiceUrl;
	}

    protected getRequestParams(params: any): any {
        if (params == null) { return null }

		let httpParams = new HttpParams();
		Object.keys(params).forEach((key) => {
			if (params[key] != null) {
			    httpParams = httpParams.append(key, params[key]);
			}
		});
		return httpParams;
	}

	protected getRequestHeaders(): HttpHeaders {
		const headers = new HttpHeaders({
			'Content-Type': 'application/json; charset=utf-8',
			'Cache-Control': 'no-cache, max-age=0',
			Pragma: 'no-cache',
		});
		return headers;
	}

    public getRandomInt(max) {
        return Math.floor(Math.random() * max);
    }
    
}
