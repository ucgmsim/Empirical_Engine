# Empirical_Engine
Contains codes to calculate Empirical IMs

To calculate empirical IMs:

```
usage: calculate_empirical.py [-h] [--vs30_file VS30_FILE]
                              [--vs30_default VS30_DEFAULT]
                              [-r RUPTURE_DISTANCE] [-srf SRF_INFO]
                              [-s STATIONS [STATIONS ...]]
                              [-rm MAX_RUPTURE_DISTANCE] [-i IDENTIFIER]
                              [-c CONFIG] [-e] [-p PERIOD [PERIOD ...]]
                              [-m IM [IM ...]]
                              output

positional arguments:
  output                output directory

optional arguments:
  -h, --help            show this help message and exit
  --vs30_file VS30_FILE, -v VS30_FILE
                        vs30 file. Default value is 500 if station/file not
                        present
  --vs30_default VS30_DEFAULT
                        Sets the default value for the vs30
  -r RUPTURE_DISTANCE, --rupture_distance RUPTURE_DISTANCE
                        Path to the rupture distance csv file
  -srf SRF_INFO, --srf_info SRF_INFO
                        Path to srf-info file
  -s STATIONS [STATIONS ...], --stations STATIONS [STATIONS ...]
                        List of stations to calculate empiricals for
  -rm MAX_RUPTURE_DISTANCE, --max_rupture_distance MAX_RUPTURE_DISTANCE
                        Only calculate empiricals for stations that are within
                        X distance to rupture
  -i IDENTIFIER, --identifier IDENTIFIER
                        run-name for run
  -c CONFIG, --config CONFIG
                        configuration file to select which model is being used
  -e, --extended_period
                        Indicate the use of extended(100) pSA periods
  -p PERIOD [PERIOD ...], --period PERIOD [PERIOD ...]
                        pSA period(s) separated by a space. eg: 0.02 0.05 0.1.
  -m IM [IM ...], --im IM [IM ...]
                        Intensity measure(s) separated by a space(if more than
                        one). eg: PGV PGA CAV.
```

e.g.
```
python calculate_empirical.py -v non_uniform_whole_nz_with_real_stations-hh400_v18p6.vs30 -r rrups/2012p713691.csv -srf srf/Kelly_HYP01-01_S1244.info -i Kelly_HYP01-01_S1244 output -s CCCC GODS
```

To aggregate the empirical files calculated above:
```
usage: emp_aggregation.py [-h] [-o OUTPUT_DIR] [-i IDENTIFIER] [-r RUPTURE]
                          [-v VERSION]
                          im_files [im_files ...]

positional arguments:
  im_files

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        path to output folder that stores the aggregated
                        measures
  -i IDENTIFIER, --identifier IDENTIFIER
                        run-name for run
  -r RUPTURE, --rupture RUPTURE
                        Please specify the rupture name of the simulation.
                        eg.Albury
  -v VERSION, --version VERSION
                        The version of the simulation. eg.18p4
```
e.g.
```
python emp_aggregation
-o /home/jam335/Documents/Empirical-Engine/Data/output
-i Darfield_Benchmark -r Darfield
/home/jam335/Documents/Empirical-Engine/Data/output/Darfield_AS_16_Ds575.csv
/home/jam335/Documents/Empirical-Engine/Data/output/Darfield_AS_16_Ds595.csv
/home/jam335/Documents/Empirical-Engine/Data/output/Darfield_Br_13_PGA.csv
/home/jam335/Documents/Empirical-Engine/Data/output/Darfield_Br_13_PGV.csv
/home/jam335/Documents/Empirical-Engine/Data/output/Darfield_Br_13_pSA.csv
/home/jam335/Documents/Empirical-Engine/Data/output/Darfield_CB_12_AI.csv
/home/jam335/Documents/Empirical-Engine/Data/output/Darfield_CB_12_CAV.csv
```