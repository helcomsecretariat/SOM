The results are saved to the excel file specified in `config.toml`. The plots contain visualizations of these results for an easier analysis. An additional file, `results_raw.xlsx`, is saved with the measure, activity, pressure, state and area IDs instead of names. 

#### `sheet:PressureMean`

| Column | Type | Description |
| --- | --- | --- |
| ID | Text | Pressure ID |
| area_name | Number | One column per unique area, pressure level (mean) in area relative to before applying measures (fraction) |

#### `sheet:PressureError`

| Column | Type | Description |
| --- | --- | --- |
| ID | Text | Pressure ID |
| area_name | Number | One column per unique area, standard error to value in `sheet:PressureMean` for area |

#### `sheet:PressureDist`

| Column | Type | Description |
| --- | --- | --- |
| ID | Text | Pressure ID |
| area_name | Number | One column per unique area, all individual simulation round pressure levels for area, space separated |

#### `sheet:TPLMean`

| Column | Type | Description |
| --- | --- | --- |
| ID | Text | State ID |
| area_name | Number | One column per unique area, Total Pressure Load on state level (mean) in area relative to before applying measures (fraction) |

#### `sheet:TPLError`

| Column | Type | Description |
| --- | --- | --- |
| ID | Text | State ID |
| area_name | Number | One column per unique area, standard error to value in `sheet:TPLMean` for area |

#### `sheet:TPLDist`

| Column | Type | Description |
| --- | --- | --- |
| ID | Text | State ID |
| area_name | Number | One column per unique area, all individual simulation round Total Pressure Load levels for area, space separated |

#### `sheet:TPLRedMean`

| Column | Type | Description |
| --- | --- | --- |
| ID | Text | State ID |
| area_name | Number | One column per unique area, reduction on Total Pressure Load on state (mean) in area relative to before applying measures (fraction), equals 1 - value in `sheet:TPLMean` |

#### `sheet:TPLRedError`

| Column | Type | Description |
| --- | --- | --- |
| ID | Text | State ID |
| area_name | Number | One column per unique area, standard error to value in `sheet:TPLRedMean` for area |

#### `sheet:TPLRedDist`

| Column | Type | Description |
| --- | --- | --- |
| ID | Text | State ID |
| area_name | Number | One column per unique area, all individual simulation round Total Pressure Load reductions for area, space separated |

#### `sheet:Thresholds###Mean`

One sheet per target threshold ###.

| Column | Type | Description |
| --- | --- | --- |
| ID | Text | State ID |
| area_name | Number | One column per unique area, target reduction on Total Pressure Load on state (mean) in area relative to before applying measures (fraction) |

#### `sheet:Thresholds###Error`

One sheet per target threshold ###.

| Column | Type | Description |
| --- | --- | --- |
| ID | Text | State ID |
| area_name | Number | One column per unique area, standard error to value in `sheet:ThresholdsMean` for area |

#### `sheet:Thresholds###Dist`

One sheet per target threshold ###.

| Column | Type | Description |
| --- | --- | --- |
| ID | Text | State ID |
| area_name | Number | One column per unique area, all individual simulation round target reductions for area, space separated |

#### `sheet:MeasureEffectsMean`

| Column | Type | Description |
| --- | --- | --- |
| measure | Text | Measure ID |
| activity | Text | Activity ID |
| pressure | Text | Pressure ID |
| state | Text | State ID |
| reduction | Number | Average measure reduction effect (fraction) |

#### `sheet:MeasureEffectsError`

| Column | Type | Description |
| --- | --- | --- |
| measure | Text | Measure ID |
| activity | Text | Activity ID |
| pressure | Text | Pressure ID |
| state | Text | State ID |
| reduction | Number | Measure reduction effect standard error (fraction) |

#### `sheet:MeasureEffectsDist`

| Column | Type | Description |
| --- | --- | --- |
| measure | Text | Measure ID |
| activity | Text | Activity ID |
| pressure | Text | Pressure ID |
| state | Text | State ID |
| reduction | Number | Measure reduction effects from individual simulation rounds, space separated |

#### `sheet:ActivityContributionsMean`

| Column | Type | Description |
| --- | --- | --- |
| activity | Text | Activity ID |
| pressure | Text | Pressure ID |
| area_id | Text | Area ID |
| contribution | Number | Average activity contribution to pressure (fraction) |

#### `sheet:ActivityContributionsError`

| Column | Type | Description |
| --- | --- | --- |
| activity | Text | Activity ID |
| pressure | Text | Pressure ID |
| area_id | Text | Area ID |
| contribution | Number | Activity contribution to pressure standard error (fraction) |

#### `sheet:ActivityContributionsDist`

| Column | Type | Description |
| --- | --- | --- |
| activity | Text | Activity ID |
| pressure | Text | Pressure ID |
| area_id | Text | Area ID |
| contribution | Number | Activity contributions to pressure from individual simulation rounds, space separated |

#### `sheet:PressureContributionsMean`

| Column | Type | Description |
| --- | --- | --- |
| state | Text | State ID |
| pressure | Text | Pressure ID |
| area_id | Text | Area ID |
| contribution | Number | Average pressure contribution to total pressure load on state (fraction) |

#### `sheet:PressureContributionsError`

| Column | Type | Description |
| --- | --- | --- |
| state | Text | State ID |
| pressure | Text | Pressure ID |
| area_id | Text | Area ID |
| contribution | Number | Pressure contribution to total pressure load on state standard error (fraction) |

#### `sheet:PressureContributionsDist`

| Column | Type | Description |
| --- | --- | --- |
| state | Text | State ID |
| pressure | Text | Pressure ID |
| area_id | Text | Area ID |
| contribution | Number | Pressure contributions to total pressure load on state from individual simulation rounds, space separated |
