{
  buildPythonPackage,
  hatchling,
  gitignoreSource,
  lib,
  numpy,
  pyscf,
}:
let
  versionFile = builtins.readFile ./src/dmet/__init__.py;
  versionLine = builtins.replaceStrings [ "\n" "\r" ] [ "" "" ] versionFile;
  versionMatch = builtins.match ''.*__version__ = "([^"]+)".*'' versionLine;
  version =
    if versionMatch == null then
      throw "Cannot find __version__ in src/dmet/__init__.py"
    else
      builtins.head versionMatch;
in
buildPythonPackage {
  pname = "dmet";
  inherit version;
  src = gitignoreSource ./.;
  pyproject = true;

  build-system = [ hatchling ];

  nativeBuildInputs = [ ];

  dependencies = [
    numpy
    pyscf
  ];

  doCheck = true;
  pythonImportsCheck = [ "dmet" ];
  meta = {
    description = "";
    homepage = "";
    license = lib.licenses.mit;
    mainProgram = "";
    maintainers = [
      {
        name = "yushengyangchem";
        email = "yushengyangchem@gmail.com";
        github = "yushengyangchem";
      }
    ];
  };
}
