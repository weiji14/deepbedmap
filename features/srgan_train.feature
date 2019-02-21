# language: en
@fixture.srgan_train
Feature: Train Super Resolution Model
  In order to have a well performing super resolution model
  As a machine learning engineer,
  We want to craft and teach the model to do well on a test area

  Background: Load the prepared data
    Given a prepared collection of tiled raster data

  Scenario Outline: Train Super Resolution Model with fixed hyperparameters
    Given some hyperparameter settings <num_residual_blocks> <residual_scaling> <learning_rate>
      And a compiled neural network model
     When the model is trained for a while
     Then we know how well the model performs on our test area

  Examples: Fixed hyperparameters
    | num_residual_blocks | residual_scaling | learning_rate |
    | 1 | 0.3 | 5e-4 |
