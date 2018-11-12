# language: en
Feature: DeepBedMap
  In order to create a great map of Antarctica's bed
  As a scientist,
  We want a model that produces realistic images from many open datasets
  
  Scenario: Determine high resolution bed
    Given low and high resolution images related to Antarctica's bed
     When some view of Antarctica is selected
     Then a high resolution bed elevation map is returned
