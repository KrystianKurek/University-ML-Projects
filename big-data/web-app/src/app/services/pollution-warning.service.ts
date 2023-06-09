import { Injectable } from '@angular/core';
import { CommonHelper, Parameter, PollutionCategory } from '../helpers/common-helper';
import { Column } from '../models/column';
import { PollutionGridElement } from '../models/pollution-grid-element';
import { Warning } from '../models/warning';

@Injectable({
  providedIn: 'root'
})
export class PollutionWarningService {
  addMeasurements(latestPollutionGridDataSource: PollutionGridElement[]) {
      throw new Error('Method not implemented.');
  }

  private measurements: PollutionGridElement[] = [];
  private distinctMeasurements: PollutionGridElement[] = [];
  private columns: Column[] = [];

  private coWarning: Warning;
  private no2Warning: Warning;
  private o3Warning: Warning;
  private pm10Warning: Warning;
  private pm25Warning: Warning;
  private so2Warning: Warning;
  private bcWarning: Warning;

  private warnings: Warning[] = [];

  constructor() {
  }

  private initialize(): void {
  }

  public reset(): void {
    this.measurements = [];
    this.distinctMeasurements = [];
    this.columns = [];

    this.coWarning = null;
    this.no2Warning = null;
    this.o3Warning = null;
    this.pm10Warning = null;
    this.pm25Warning = null;
    this.so2Warning = null;
    this.bcWarning = null;

    this.warnings = [];
  }

  public addColumns(columns: Column[]): void {
    this.columns = columns;
  }

  public getWarnings(): Warning[] {
    return this.warnings;
  }

  public prepareWarnings(): void {
    this.setDistinctMeasurements();

    if (this.columns.includes(Column.pm10)) {
      this.pm10Warning = new Warning(
        Parameter.pm10,
        this.checkAirPollutionCategory(Parameter.pm10, this.calculateAverage(24, Parameter.pm10))
      );
      this.warnings.push(this.pm10Warning);
    }

    if (this.columns.includes(Column.pm25)) {
      this.pm25Warning = new Warning(
        Parameter.pm25,
        this.checkAirPollutionCategory(Parameter.pm25, this.calculateAverage(24, Parameter.pm25))
      );
      this.warnings.push(this.pm25Warning);
    }

    if (this.columns.includes(Column.o3)) {
      this.o3Warning = new Warning(
        Parameter.o3,
        this.checkAirPollutionCategory(Parameter.o3, this.calculateAverage(8, Parameter.o3))
      );
      this.warnings.push(this.o3Warning);
    }

    if (this.columns.includes(Column.no2)) {
      this.no2Warning = new Warning(
        Parameter.no2,
        this.checkAirPollutionCategory(Parameter.no2, this.calculateAverage(1, Parameter.no2))
      );
      this.warnings.push(this.no2Warning);
    }
  }

  public setDistinctMeasurements(): void {
    this.measurements.sort((first: PollutionGridElement , second: PollutionGridElement) => {
      return second.date.getTime() - first.date.getTime();
    });

    const distinctDates: Date[] = CommonHelper.getDistinctDates(this.measurements.map(measurement => measurement.date));

    for (const date of distinctDates) {
      for (const measurement of this.measurements) {
        if (date.getTime() === measurement.date.getTime()) {
          this.distinctMeasurements.push(measurement);
          break;
        }
      }
    }
  }

  private calculateAverage(hours: number, parameter: string): number {
    let sum: number = 0;
    let hoursActive: number = hours;

    for (let i = 0; i < hours; i++) {
      if (this.distinctMeasurements[i] != null && this.distinctMeasurements[i][parameter] != null) {
        sum += this.distinctMeasurements[i][parameter].getValue();
      } else {
        hoursActive--;
      }
    }

    return sum / hoursActive;
  }



  private checkAirPollutionCategory(parameter: Parameter, pollution: number): PollutionCategory {
    if (parameter === Parameter.pm10) {
      if (pollution > 0 && pollution <= 55) {
        return PollutionCategory.Good;
      } else if (pollution > 55 && pollution <= 155) {
        return PollutionCategory.Moderate;
      } else if (pollution > 155 && pollution <= 255) {
        return PollutionCategory.UnhealthySensitive;
      } else if (pollution > 255 && pollution <= 355) {
        return PollutionCategory.Unhealthy;
      } else if (pollution > 355 && pollution <= 425) {
        return PollutionCategory.VeryUnhealthy;
      } else if (pollution > 425 && pollution <= 505) {
        return PollutionCategory.Hazardous;
      } else if (pollution > 505 && pollution < 604) {
        return PollutionCategory.VeryHazardous;
      } else {
        return PollutionCategory.OutOfRange;
      }
    }

    if (parameter === Parameter.pm25) {
      if (pollution > 0 && pollution <= 12.1) {
        return PollutionCategory.Good;
      } else if (pollution > 12.1 && pollution <= 35.5) {
        return PollutionCategory.Moderate;
      } else if (pollution > 35.5 && pollution <= 55.5) {
        return PollutionCategory.UnhealthySensitive;
      } else if (pollution > 55.5 && pollution <= 150.5) {
        return PollutionCategory.Unhealthy;
      } else if (pollution > 150.5 && pollution <= 250.5) {
        return PollutionCategory.VeryUnhealthy;
      } else if (pollution > 250.5 && pollution <= 350.5) {
        return PollutionCategory.Hazardous;
      } else if (pollution > 350.5 && pollution < 500.4) {
        return PollutionCategory.VeryHazardous;
      } else {
        return PollutionCategory.OutOfRange;
      }
    }

    if (parameter === Parameter.o3) {
      if (pollution > 0 && pollution <= 108.53) {
        return PollutionCategory.Good;
      } else if (pollution > 108.35 && pollution <= 139.87) {
        return PollutionCategory.Moderate;
      } else if (pollution > 139.87 && pollution <= 169.42) {
        return PollutionCategory.UnhealthySensitive;
      } else if (pollution > 169.42 && pollution < 208.82) {
        return PollutionCategory.Unhealthy;
      } else if (pollution > 208.82) {
        return PollutionCategory.VeryUnhealthy;
      } else {
        return PollutionCategory.OutOfRange;
      }
    }

    if (parameter === Parameter.no2) {
      if (pollution > 0 && pollution <= 101.52) {
        return PollutionCategory.Good;
      } else if (pollution > 101.52 && pollution <= 189.88) {
        return PollutionCategory.Moderate;
      } else if (pollution > 189.88 && pollution <= 678.68) {
        return PollutionCategory.UnhealthySensitive;
      } else if (pollution > 678.68 && pollution <= 1222) {
        return PollutionCategory.Unhealthy;
      } else if (pollution > 1222 && pollution <= 2350) {
        return PollutionCategory.VeryUnhealthy;
      } else if (pollution > 2350 && pollution <= 3102) {
        return PollutionCategory.Hazardous;
      } else if (pollution > 3102 && pollution < 3852.12) {
        return PollutionCategory.VeryHazardous;
      } else {
        return PollutionCategory.OutOfRange;
      }
    }
}
}

