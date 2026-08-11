# Py1 design table

## Workbook structure

- Samples: 250 datapoints, no repeated range columns
- Parameter_Info: ranges stated once
- Metadata: sampling and constraint settings

## Editable settings

- N_SAMPLES = 250
- SEED = 42
- OVERGENERATION_FACTOR = 8
- MAX_ATTEMPTS = 20
- CORE_MASS_LIMIT_G = 200.0
- COIL_MASS_LIMIT_G = 50.0
- MIN_R1_MINUS_C1_MM = 0.0
- MIN_MANUFACTURED_DIMENSION_MM = 0.5
- COIL_BOTTOM_Z_MM = 2.0
- Axial fit: l1_mm >= h1_mm + 2
- Design-space fingerprint: 7c19f05da8d0f8f526ca172bae6f48d0a05d79b894317764f0e29e8c8797cfce
- MAX_CORE_TOTAL_HEIGHT_MM = None
- Excel engine used: openpyxl
