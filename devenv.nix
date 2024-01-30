#!nix
{ config, pkgs, ... }:
{
  # https://devenv.sh/packages/
  packages = [
    pkgs.stdenv.cc.cc.lib
    pkgs.cudaPackages.cudatoolkit
    pkgs.linuxPackages.nvidia_x11
  ];

  languages.python = {
    enable = true;
    package = pkgs.python311;
    poetry.enable = true;
    poetry.install.enable = true;
  };

  enterShell = ''
    export CUDA_PATH=${pkgs.cudatoolkit}
    # export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib:${pkgs.stdenv.cc.cc.lib}/lib # WSL CASE
    export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
    export EXTRA_CCFLAGS="-I/usr/include"
      echo ================================================
      echo Nix devenv shell for Network Intrusion Detection
      echo To add python modules use 'poetry add'
      echo ================================================
  '';

  pre-commit.hooks.black.enable = true;

  env.DATASET_DIR = "${config.env.DEVENV_ROOT}/datasets";
}
