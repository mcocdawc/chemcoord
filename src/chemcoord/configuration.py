from pathlib import Path
from typing import Final, Literal

import yaml
from attrs import define
from cattrs import structure, unstructure

from chemcoord.typing import PathLike

DEFAULT_RC_PATH: Final = Path("~/.chemcoord.yml")


@define(hash=True, kw_only=True)
class Defaults:
    #: Which atomic radius data to use.
    atomic_radius_data: str = "atomic_radius_cc"
    #: Which molecular viewer to use.
    viewer: str = "gv.exe"
    #: Which filetype to use when viewing a list of molecules.
    list_viewer_file: Literal["molden", "xyz"] = "molden"


@define(hash=True, kw_only=True)
class Settings:
    defaults: Defaults = Defaults()

    def write(self, path: PathLike = DEFAULT_RC_PATH) -> None:
        """Write the settings to a yaml file

        Args:
            path: The path to the settings file.
                Defaults to ``~/.chemcoord.yml``.
        """
        with open(Path(path).expanduser(), "w") as f:
            f.write("# Settings files for `chemcoord`.\n")
            f.write("# You can delete keys; in this case the default is taken.\n")
            yaml.dump(unstructure(self), stream=f, default_flow_style=False)


def _read(path: PathLike = DEFAULT_RC_PATH) -> Settings:
    with open(Path(path).expanduser()) as f:
        return structure(yaml.safe_load(stream=f), Settings)


if DEFAULT_RC_PATH.expanduser().exists():
    settings = _read()
else:
    settings = Settings()
