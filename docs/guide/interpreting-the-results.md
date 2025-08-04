The results are saved to the excel file specified in `config.toml`. The plots contain visualizations of these results for an easier analysis. 

#### `sheet:PressureMean`

- `column:ID`: Pressure ID
- `columns:area_name...`: Pressure level (mean) in area relative to before applying measures (fraction)

#### `sheet:PressureError`

- `column:ID`: Pressure ID
- `columns:area_name...`: Standard error to value in `sheet:PressureMean` for area

#### `sheet:TPLMean`

- `column:ID`:State ID
- `columns:area_name...`: Total Pressure Load on state level (mean) in area relative to before applying measures (fraction)

#### `sheet:TPLError`

- `column:ID`: State ID
- `columns:area_name...`: Standard error to value in `sheet:TPLMean` for area

#### `sheet:TPLRedMean`

- `column:ID`:State ID
- `columns:area_name...`: Reduction on Total Pressure Load on state (mean) in area relative to before applying measures (fraction), equals 1 - value in `sheet:TPLMean`

#### `sheet:TPLRedError`

- `column:ID`: State ID
- `columns:area_name...`: Standard error to value in `sheet:TPLRedMean` for area

#### `sheet:ThresholdsMean`

- `column:ID`:State ID
- `columns:area_name...`: Target reduction on Total Pressure Load on state (mean) in area relative to before applying measures (fraction)

#### `sheet:ThresholdsError`

- `column:ID`: State ID
- `columns:area_name...`: Standard error to value in `sheet:ThresholdsMean` for area

#### `sheet:MeasureEffectsMean`

- `column:measure`: Measure ID
- `column:activity`: Activity ID
- `column:pressure`: Pressure ID
- `column:state`: State ID
- `columns:reduction`: Average measure reduction effect (fraction)

#### `sheet:MeasureEffectsError`

- `column:measure`: Measure ID
- `column:activity`: Activity ID
- `column:pressure`: Pressure ID
- `column:state`: State ID
- `columns:reduction`: Measure reduction effect standard error (fraction)

#### `sheet:ActivityContributionsMean`

- `column:activity`: Activity ID
- `column:pressure`: Pressure ID
- `column:area_id`: Area ID
- `columns:contribution`: Average activity contribution to pressure (fraction)

#### `sheet:ActivityContributionsError`

- `column:activity`: Activity ID
- `column:pressure`: Pressure ID
- `column:area_id`: Area ID
- `columns:contribution`: Activity contribution to pressure standard error (fraction)

#### `sheet:PressureContributionsMean`

- `column:state`: State ID
- `column:pressure`: Pressure ID
- `column:area_id`: Area ID
- `columns:contribution`: Average pressure contribution to total pressure load on state (fraction)

#### `sheet:PressureContributionsError`

- `column:state`: State ID
- `column:pressure`: Pressure ID
- `column:area_id`: Area ID
- `columns:contribution`: Pressure contribution to total pressure load on state standard error (fraction)
