Empirical Engine - Empirical IM calculation

(Based on https://wiki.canterbury.ac.nz/download/attachments/58458136/CodeVersioning_v18p2.pdf?version=1&modificationDate=1519269238437&api=v2 )

##Unreleased
### Fixed
### Changed
### Added


##[18.7.2] - 2018-08-16
###Added
Interpolation for Zhao beyond the period of 5.0s
Config file to select which model is used for each tect_type / im
Empirical IM_csv creation
IM_aggregation (aggregating an individual IM_csv into a larger file containing more than one IM_type)
###Changed
Verification changes (Z1p0 units and rock_site estimation for CB)

##[18.7.1] - 2018-07-13
###Added
Empirical calculations from site (rrup + vs30) and fault for Zhao_06, Bradley_13, Afshari_Stewart_16,
    Campbell_Bozorgina_2010, Campbell_Bozorgina_2012
Output IM into individual csv files
Estimation of parameters unspecified