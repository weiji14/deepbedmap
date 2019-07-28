workflow "Build Docker Container" {
  on = "push"
  resolves = ["Docker Build"]
}

action "Docker Build" {
  uses = "actions/docker/cli@aea64bb1b97c42fa69b90523667fef56b90d7cff"
  args = "build -f Dockerfile -t weiji14/deepbedmap ."
}
