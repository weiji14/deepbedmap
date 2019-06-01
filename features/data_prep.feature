# language: en
@fixture.data_prep
Feature: Data preparation
  In order to have reproducible data inputs for everyone
  As a data scientist,
  We want to share cryptographically secured pieces of the datasets

  Scenario Outline: Download and check data
    Given this <url> link to a file hosted on the web
     When we download it to <filepath>
     Then the local file should have this <sha256> checksum

  Examples: Files to download and check
    | url | filepath | sha256 |
    | https://data.cresis.ku.edu/data/rds/2017_Antarctica_Basler/csv_good/Data_20171204_02.csv | highres/Data_20171204_02.csv | 53cef7a0d28ff92b30367514f27e888efbc32b1bda929981b371d2e00d4c671b |
    | http://ramadda.nerc-bas.ac.uk/repository/entry/get/Polar%20Data%20Centre/DOI/Rutford%20Ice%20Stream%20bed%20elevation%20DEM%20from%20radar%20data/bed_WGS84_grid.txt?entryid=synth%3A54757cbe-0b13-4385-8b31-4dfaa1dab55e%3AL2JlZF9XR1M4NF9ncmlkLnR4dA%3D%3D | highres/bed_WGS84_grid.txt | 7396e56cda5adb82cecb01f0b3e01294ed0aa6489a9629f3f7e8858ea6cb91cf |

  Scenario Outline: Grid datasets
    Given a collection of raw high resolution datasets <input_pattern>
     When we process the data through <pipeline_file>
      And interpolate the xyz data table to <output_file>
     Then a high resolution raster grid is returned

  Examples: ASCII text files to grid
    | input_pattern | pipeline_file | output_file |
    | bed_WGS84_grid.txt | bed_WGS84_grid.json | bed_WGS84_grid.nc |

  Scenario Outline: Tile datasets
    Given a big <dataset_type> raster grid <raster_grid>
      And a collection of square bounding boxes "model/train/tiles_3031.geojson"
     When we crop the big raster grid using those bounding boxes
     Then a stack of small raster tiles is returned

  Examples: Raster grids to tile
    | dataset_type | raster_grid |
    | highres | 2010tr.nc |
