"""Drop the existing Milvus collection and recreate the hybrid-search schema."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.settings import settings
from app.retrieval import MilvusCollectionSchema, MilvusStore


def build_store() -> MilvusStore:
    return MilvusStore(
        schema=MilvusCollectionSchema(
            collection_name=settings.milvus_collection_name,
            dimension=settings.embedding_dimension,
            dense_metric_type=settings.milvus_dense_metric_type,
            sparse_vector_field_name=settings.milvus_sparse_field_name,
            analyzer_type=settings.milvus_analyzer_type,
            rrf_k=settings.hybrid_rrf_k,
        ),
        host=settings.milvus_host,
        port=settings.milvus_port,
    )


def reset_collection(store: MilvusStore, confirm_drop: bool = False) -> None:
    client = store.connect()
    collection_name = store.schema.collection_name
    if client.has_collection(collection_name=collection_name):
        if not confirm_drop:
            raise RuntimeError(
                f"Collection {collection_name!r} already exists. "
                "Re-run with --confirm-drop to delete it."
            )
        client.drop_collection(collection_name=collection_name)
        print(f"Dropped collection: {collection_name}")

    store.create_collection()
    print(f"Created hybrid-search collection: {collection_name}")
    print('Re-run POST /ingest with "force_reingest": true to load data into the empty collection.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-drop",
        action="store_true",
        help="Allow deletion of an existing collection. Existing data will not be preserved.",
    )
    args = parser.parse_args()
    reset_collection(build_store(), confirm_drop=args.confirm_drop)


if __name__ == "__main__":
    main()
