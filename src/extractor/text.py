from pathlib import Path


def extract_text(path: Path):
    from unstructured.partition.text import partition_text
    from unstructured.documents.elements import NarrativeText, Title, ListItem

    keep = (NarrativeText, Title, ListItem)
    elements = partition_text(filename=str(path))
    return [el for el in elements if isinstance(el, keep)]
