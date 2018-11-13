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
    | https://data.cresis.ku.edu/data/rds/2017_Antarctica_Basler/csv_good/2017_Antarctica_Basler.csv | highres/2017_Antarctica_Basler.csv | 53cef7a0d28ff92b30367514f27e888efbc32b1bda929981b371d2e00d4c671b |
