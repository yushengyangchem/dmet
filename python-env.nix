{
  pkgs,
  pythonVersion,
  gitignoreSource,
}:
let
  pythonEnv = pkgs.${pythonVersion}.override {
    packageOverrides = self: super: {
      dmet = self.callPackage ./. { inherit gitignoreSource; };
    };
  };
in
pythonEnv
