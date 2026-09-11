# Table 1: Benchmark Dataset Characteristics and Chronological Partitions

| Dataset | Geographic Scope / Target | Total Obs. (Hours) | Temporal Span | Resolution | Train Split (70%) | Val Split (15%) | Test Split (15%) | Non-Overlapping Test Days ($) | Target Unit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Modern PJM** | Regional Transmission Org. (RTO) | 8,784 | Jan 2023 – Dec 2023 | 1 Hour | 6,148 | 1,318 | 1,318 | 53 | Megawatts (MW) |
| **GEFCom2014** | US Utility (Zone 1) | 35,064 | Jan 2007 – Dec 2010 | 1 Hour | 24,544 | 5,260 | 5,260 | 456 | Kilowatts (kW) |
| **UCI Electricity** | Aggregate Load (Cohort 320 clients) | 14,016 | Jan 2012 – Aug 2013 | 1 Hour | 9,811 | 2,102 | 2,103 | 163 | Megawatts (MW) |

*Note: All datasets use an identical 168-hour (7-day) historical lookback to forecast a 24-hour ahead horizon (=24$). Standard scaling is fit exclusively on the chronological training split. Boundary lookback values for validation and test sets are drawn from the immediate historical partition without lookahead contamination.*
