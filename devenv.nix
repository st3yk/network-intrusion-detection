#!nix
{ config, pkgs, ... }:
{
  # https://devenv.sh/packages/
  packages = [
    (pkgs.python311.withPackages (ps: [
      ps.numpy
      ps.pandas
      ps.dateutil
      ps.pytz
      ps.torch
    ]))
  ];

  # https://devenv.sh/basics/
  env.GREET = "dl-project-shell";
  scripts.hello.exec = "echo hello from $GREET";

  enterShell = ''
    hello
  '';

  pre-commit.hooks.black.enable = true;

  env.DATASET_DIR = "${config.env.DEVENV_ROOT}/datasets";
}
