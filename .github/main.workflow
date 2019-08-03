workflow "Build and Test DeepBedMap" {
  resolves = ["Run Tests"]
  on = "push"
}

action "Build DeepBedMap App" {
  uses = "actions/docker/cli@86ff551d26008267bb89ac11198ba7f1d807b699"
  args = "build --file Dockerfile --tag weiji14/deepbedmap --target app ."
}

action "Run Tests" {
  uses = "actions/docker/cli@86ff551d26008267bb89ac11198ba7f1d807b699"
  args = "run weiji14/deepbedmap python -m pytest --verbose --disable-warnings --nbval test_ipynb.ipynb"
  needs = ["Build DeepBedMap App"]
}
