import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { TranslateService } from '@ngx-translate/core';
import { Observable } from 'rxjs';
import { Parameter, WikiResponse } from '../helpers/common-helper';


@Injectable({
  providedIn: 'root'
})
export class WikiService {
  private apiUrl: string;
  
  constructor(private httpClient: HttpClient, private translate: TranslateService) {}

  private setApiUrl(): void {
    if (this.translate.currentLang === 'en') {
      this.apiUrl = 'https://en.wikipedia.org/api/rest_v1/page/summary/';
    } else if (this.translate.currentLang === 'pl') {
      this.apiUrl = 'https://pl.wikipedia.org/api/rest_v1/page/summary/';
    }
  }

  public getWiki(parameter: Parameter): Observable<WikiResponse> {
    this.setApiUrl();
    return this.httpClient.get<WikiResponse>(`${this.apiUrl}${this.getParameterPostfix(parameter)}`);
  }

  private getParameterPostfix(parameter: Parameter) {
    let parameterPostfix: string;

    if (this.translate.currentLang === 'en') {
      if (parameter === Parameter.no2) {
        parameterPostfix = 'Nitrogen_dioxide';
      }

      if (parameter === Parameter.o3) {
        parameterPostfix = 'Ozone';
      }

      if (parameter === Parameter.pm10) {
        parameterPostfix = 'Particulates';
      }

      if (parameter === Parameter.pm25) {
        parameterPostfix = 'Particulates';
      }
    }

    if (this.translate.currentLang === 'pl') {

      if (parameter === Parameter.no2) {
        parameterPostfix = 'Dwutlenek_azotu';
      }

      if (parameter === Parameter.o3) {
        parameterPostfix = 'Ozon';
      }

      if (parameter === Parameter.pm10) {
        parameterPostfix = 'PM10';
      }

      if (parameter === Parameter.pm25) {
        parameterPostfix = 'PM2,5';
      }
    }
    return parameterPostfix;
  }
}
