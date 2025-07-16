{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    typix = {
      url = "github:loqusion/typix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs =
    {
      nixpkgs,
      flake-utils,
      typix,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (pkgs) lib;

        typixLib = typix.lib.${system};

        reportBuildArgs = {
          typstSource = "report.typ";
          src = lib.fileset.toSource {
            root = ./report;
            fileset = lib.fileset.unions [
              (lib.fileset.fromSource (typixLib.cleanTypstSource ./report))
              ./report/graphics
              ./report/references.bib
            ];
          };
        };

        report = typixLib.buildTypstProject reportBuildArgs;
        build-report = typixLib.buildTypstProjectLocal reportBuildArgs;
        watch-report = typixLib.watchTypstProject { typstSource = "report/report.typ"; };
      in
      {
        devShell = typixLib.devShell {
          packages = [
            build-report
            watch-report
          ];
        };

        checks = {
          inherit
            report
            build-report
            watch-report
            ;
        };

        packages = {
          inherit report;
          default = report;
        };

        apps =
          let
            build-report-app = flake-utils.lib.mkApp {
              drv = build-report;
            };
          in
          {
            build-report = build-report-app;
            default = build-report-app;

            watch-report = flake-utils.lib.mkApp {
              drv = watch-report;
            };
          };
      }
    );
}
