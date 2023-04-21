
export class PollutionForecast {
    constructor(longtitude: number, latitude: number, particle:string, prediction: number, station_name: string, timestamp: Date) {
        this.longtitude = longtitude;
        this.lattitude = latitude;
        this.station_name = station_name;
        this.timestamp = timestamp;
        this.particle = particle;
        this.prediction = prediction;
    }
    public longtitude: number;
    public lattitude: number;
    public station_name: string;
    public timestamp: Date;
    public particle: string;
    public prediction: number;
}