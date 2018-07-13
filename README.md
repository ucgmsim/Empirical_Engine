# Empirical_Engine
Contains codes to calculate Empirical IMs

To calculate empirical IMs:

```
usage: calculate_empirical.py [-h] [--vs30_file VS30_FILE]
                              [--vs30_default VS30_DEFAULT]
                              [-r RUPTURE_DISTANCE] [-srf SRF_INFO]
                              [-s STATIONS [STATIONS ...]]
                              [-rm MAX_RUPTURE_DISTANCE] [-i IDENTIFIER]
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
```

e.g.
```
python calculate_empirical.py -v non_uniform_whole_nz_with_real_stations-hh400_v18p6.vs30 -r rrups/2012p713691.csv -srf srf/Kelly_HYP01-01_S1244.info -i Kelly_HYP01-01_S1244 output -s CCCC GODS
```