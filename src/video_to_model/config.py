"""Project-wide paths and defaults."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def from_project_root(cls, root: Path) -> "ProjectPaths":
        return cls(root=root.resolve())

    @property
    def inputs(self) -> Path:
        return self.root / "data" / "inputs"

    @property
    def outputs(self) -> Path:
        return self.root / "data" / "outputs"

    @property
    def checkpoints(self) -> Path:
        return self.root / "models" / "checkpoints"

    def default_output_path(self, input_path: Path) -> Path:
        return self.outputs / f"{input_path.stem}_depth.mp4"
