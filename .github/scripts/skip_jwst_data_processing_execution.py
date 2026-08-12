"""Skip input-dependent JWST data-processing notebooks during CI execution."""

import json
from pathlib import Path


NOTEBOOK_DIRECTORY = Path("notebooks/JWST_Data_Processing")
SKIP_TAGS = {"nbval-skip", "skip-execution"}


def main():
    repository_root = Path(__file__).resolve().parents[2]
    notebook_directory = repository_root / NOTEBOOK_DIRECTORY
    notebook_paths = sorted(notebook_directory.glob("*.ipynb"))

    for notebook_path in notebook_paths:
        with notebook_path.open(encoding="utf-8") as handle:
            notebook = json.load(handle)

        tagged_cells = 0
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue

            metadata = cell.setdefault("metadata", {})
            existing_tags = set(metadata.get("tags", []))
            metadata["tags"] = sorted(existing_tags | SKIP_TAGS)
            tagged_cells += 1

        with notebook_path.open("w", encoding="utf-8") as handle:
            json.dump(notebook, handle, ensure_ascii=False, indent=1)
            handle.write("\n")

        print(f"Marked {tagged_cells} code cells to skip in {notebook_path}")


if __name__ == "__main__":
    main()
