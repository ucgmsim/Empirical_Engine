Empirical Engine - Empirical IM calculation

(Based on https://wiki.canterbury.ac.nz/download/attachments/58458136/CodeVersioning_v18p2.pdf?version=1&modificationDate=1519269238437&api=v2 )

## Unreleased
### Fixed
### Changed
### Added

## [23.7.11] - 2023-07-11
### Changed
  - Removed most of the code, except for the OpenQuake vectorised wrapper
  - Old code is recoverable from the `backup_20230707` branch


## [21.1.1] - 2021-01-21
### Added
  - Added OpenQuake compatibility layer
  - Added OpenQuake models for ver component
  - Added component selection to command line arguments

## [19.9.3] - 2020-02-11 -
### Added
  - Added RotD100/50 (rotd100_50) and inelastic to elastic spectral displacement ratio (IESDR) empiricals
### Changed
  - Added 16 pSA periods to the default

## [19.9.2] - 2019-12-10 -
### Added
  - Permutation aggregation script to generate all combinations of intensity measure models for a given event or fault from the available intensity measure files
  - Abrahamson A18 model added and verification script created 
### Changed
  - Empirical calculation and aggregation scripts refactored to allow easier access to their functionality

## [19.9.1] - 2019-09-11
### Added
Additional Empirical Models. (and unit tests)

## [18.7.2] - 2018-08-16
### Added
Interpolation for Zhao beyond the period of 5.0s
Config file to select which model is used for each tect_type / im
Empirical IM_csv creation
IM_aggregation (aggregating an individual IM_csv into a larger file containing more than one IM_type)
### Changed
Verification changes (Z1p0 units and rock_site estimation for CB)

## [18.7.1] - 2018-07-13
### Added
Empirical calculations from site (rrup + vs30) and fault for Zhao_06, Bradley_13, Afshari_Stewart_16,
    Campbell_Bozorgina_2010, Campbell_Bozorgina_2012
Output IM into individual csv files
Estimation of parameters unspecified
