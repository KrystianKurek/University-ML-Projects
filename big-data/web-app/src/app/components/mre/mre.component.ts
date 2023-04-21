import { Component, OnInit } from '@angular/core';
import { BatchViewModel } from 'src/app/models/batch-view.model';
import { CassandraService } from 'src/app/services/cassandra.service';

@Component({
    selector: 'app-mre',
    templateUrl: './mre.component.html',
    styleUrls: ['./mre.component.scss']
})
export class MreComponent implements OnInit {
    public mres: BatchViewModel[] = [];

    constructor(private cassandraService: CassandraService) {}

    public async ngOnInit() {
        await this.getMREsData();
    }

    public async getMREsData(): Promise<void> {
        this.mres = await this.cassandraService.getMREs();
    }

    public roundMRE(mre: number): string {
        return (100 * mre).toFixed(2) + "%";
    }

    public transformInterval(interval: number): string {
        if (interval === 0) {
            return '<25'
        } else if (interval === 1) {
            return '25-50'
        } else if (interval === 2) {
            return '>50'
        }
    }

    
}