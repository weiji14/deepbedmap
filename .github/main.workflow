workflow "Build Docker Container" {
  on = "push"
  resolves = ["Run Tests"]
}

action "Docker Build" {
  uses = "./"
  args = "echo 'Build Done'"
}

action "Run Tests" {
  uses = "./"
  needs = ["Docker Build"]
  args = "pipenv run python -m pytest --verbose --disable-warnings --nbval test_ipynb.ipynb"
}
