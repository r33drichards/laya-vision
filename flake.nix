{
  description = "Laya Vision: typed, calibrated decisions about an image (documentation site and browser demo)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        docsPython = pkgs.python3.withPackages (
          ps: with ps; [
            mkdocs
            mkdocs-mermaid2-plugin
          ]
        );

        # The browser demo (web-demo/) is published with the site, under demo/. Only the page itself: its tests,
        # fixtures and README stay behind, and the model files load from the Hub (thaitea/laya-vision-web).
        demoFiles = [ "index.html" "app.js" "worker.js" "laya.js" "style.css" "example.jpg" ];
      in
      {
        # nix build .#docs  -> ./result: the MkDocs site (site-docs/) with the browser demo under demo/
        packages.docs = pkgs.stdenvNoCC.mkDerivation {
          pname = "laya-vision-docs";
          version = "0.1.0";
          src = self;

          nativeBuildInputs = [ docsPython ];

          dontConfigure = true;
          strictDeps = true;

          buildPhase = ''
            runHook preBuild

            python3 -m mkdocs build --strict
            mkdir -p site/demo
            ${pkgs.lib.concatMapStringsSep "\n" (f: "cp web-demo/${f} site/demo/") demoFiles}

            runHook postBuild
          '';

          installPhase = ''
            runHook preInstall
            cp -R site "$out"
            runHook postInstall
          '';
        };

        packages.default = self.packages.${system}.docs;

        # nix develop  -> mkdocs with the same plugins, for `mkdocs serve`
        devShells.default = pkgs.mkShell {
          packages = [ docsPython ];
        };
      });
}
