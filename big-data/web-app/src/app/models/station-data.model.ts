import { PredictionLevels } from "./prediction-levels";

export class StationDataModel {
    public longtitude: number;
    public lattitude: number;
    public station_name: string;
    public timestamp: Date;
    public predictions?: PredictionLevels;
}