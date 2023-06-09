import { Component, Inject } from '@angular/core';
import { MatDialogRef, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { TranslateService } from '@ngx-translate/core';
import { CommonHelper, Parameter } from 'src/app/helpers/common-helper';
import { Wiki } from 'src/app/models/wiki';

export interface PopupData {
  wiki: Wiki;
  parameter: string;
}

@Component({
  selector: 'app-popup',
  templateUrl: './popup.component.html',
  styleUrls: ['./popup.component.scss']
})

export class PopupComponent {

  constructor(
    public popupRef: MatDialogRef<PopupComponent>,
    private translate: TranslateService,
    @Inject(MAT_DIALOG_DATA) public data: PopupData
  ) {}

  public static getWidth(isHandset: boolean): string {
    return isHandset ? '90vw' : '50vw';
  }

  public static getHeight(isHandset: boolean): string {
    return isHandset ? '95vh' : '50vh';
  }

  public getTitle(parameter): string {
    if (parameter === Parameter.pm10) {
      return this.translate.instant('pm10');
    } else if (parameter === Parameter.pm25) {
      return this.translate.instant('pm25');
    } else if (parameter === Parameter.o3) {
      return this.translate.instant('o3');
    } else if (parameter === Parameter.bc) {
      return this.translate.instant('bc');;
    } else if (parameter === Parameter.co) {
      return this.translate.instant('co');
    } else if (parameter === Parameter.no2) {
      return this.translate.instant('no2');
    } else if (parameter === Parameter.so2) {
      return this.translate.instant('so2');
    }
  }

  public createWikiParagraph() {
    document.getElementById('created-wiki-text').innerHTML = CommonHelper.getFormattedHTMLPopup(this.data.wiki.extract);
  }

  public createTitle() {
    document.getElementById('created-title').innerHTML = this.getTitle(this.data.parameter);
  }
}
