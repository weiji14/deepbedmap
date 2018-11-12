# Behaviour of DeepBedMap features

Code can be hard to understand, so the `.feature` files here summarizes what the code should be doing, i.e. the behaviour.
Each `.feature` file describes (in plain language) the reasoning behind the code, the audience it is catering for, and what is being offered.
In other words, the **Why**, **Who** and **What**.

This is a template for what the `.feature` file looks like inside:

```gherkin
    # language: en
    Feature: <title>
      In order to ... <Why>
      As a ... <Who>
      We want ... <What>

      Scenario: <example>
        Given ... <input>
         When ... <something happens>
         Then ... <output>
```
