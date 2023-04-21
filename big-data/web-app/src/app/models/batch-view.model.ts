
export class BatchViewModel {
    constructor(timestamp: Date, particle: string, interval: number, mre: number) {
        this.timestamp = timestamp;
        this.particle = particle;
        this.interval = interval;
        this.MRE = mre;
    }
    public timestamp: Date;
    public particle: string;
    public interval: number;
    public MRE: number;
}