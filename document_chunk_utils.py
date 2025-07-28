"""Utilities for building LangChain Documents from a DataFrame.

This module provides a helper function to perform hybrid chunking on text
from a pandas DataFrame and convert each chunk into a ``Document`` object.

Import the function like this::

    from document_chunk_utils import build_documents_hybrid_from_df
"""

from __future__ import annotations

import re
from typing import List

import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def _initial_split(text: str) -> List[str]:
    """Split text using hybrid rules without discarding separators."""
    pattern = re.compile(
        r"(?:\n\s*\n)+"      # multiple newlines
        r"|(?=【[^】]+】)"      # section headers
        r"|(?=\d+[\.\u3001])"  # ordered lists like 1. or 1、
    )
    parts = [part.strip() for part in pattern.split(text) if part.strip()]
    return parts


def build_documents_hybrid_from_df(
    df: pd.DataFrame, chunk_size: int = 300, chunk_overlap: int = 30
) -> List[Document]:
    """Convert DataFrame rows into Document objects using hybrid chunking.

    Args:
        df: The DataFrame containing a ``context`` column and metadata columns.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap size passed to ``RecursiveCharacterTextSplitter``.

    Returns:
        List[Document]: Documents built from all rows of ``df``.
    """
    documents: List[Document] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    for _, row in df.iterrows():
        if "context" not in row:
            continue
        text = str(row["context"])
        metadata = row.drop(labels=["context"]).to_dict()

        segments = _initial_split(text)
        for segment in segments:
            if len(segment) > chunk_size:
                sub_segments = splitter.split_text(segment)
            else:
                sub_segments = [segment]
            for sub in sub_segments:
                documents.append(Document(page_content=sub, metadata=metadata))

    return documents
