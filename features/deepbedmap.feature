# language: en
@fixture.deepbedmap
Feature: DeepBedMap
  In order to create a great map of Antarctica's bed
  As a scientist,
  We want a model that produces realistic images from many open datasets

  @skip
  Scenario Outline: Determine high resolution bed
    Given some view of Antarctica <bounding_box>
     When we gather low and high resolution images related to that view
      And pass those images into our trained neural network model
     Then a four times upsampled super resolution bed elevation map is returned

  Examples: Bounding box views of Antarctica
    | bounding_box |
    | -1593714.328,-164173.7848,-1575464.328,-97923.7848 |
