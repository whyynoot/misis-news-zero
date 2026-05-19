from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "manual_dataset_label_audit_and_relabel_queue.ipynb"


def main() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=3600,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"executed={NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
