# pin nixpkgs version for reproducible builds
{ pkgs ? (import (builtins.fetchGit {
  url = https://github.com/NixOS/nixpkgs-channels.git;
  ref = "nixos-19.09";
  rev = "2de9367299f325c2b2021a44c2f63c810f8ad023";
}) {}) }:

pkgs.python2Packages.buildPythonPackage {
  pname = "learning";
  version = "0.1.0";
  src = ./.;
  checkInputs = with pkgs.python2Packages; [
    pytest
  ];
  propagatedBuildInputs = with pkgs.python2Packages; [
    numpy
  ];
  meta = with pkgs.stdenv.lib; {
    description = "A python machine learning library, with powerful customization for advanced users, and robust default options for quick implementation.";
    homepage = https://github.com/justinlovinger/learning;
    license = licenses.mit;
    maintainers = [ ];
  };
}
