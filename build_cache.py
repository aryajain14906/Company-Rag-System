"""
Run this ONCE locally to pre-build the embeddings cache, then commit
the resulting policies/_embeddings_cache.pkl file to your git repo.

With the content-hash cache signature fix in extract.py, this cache
will be reused on every future Render deploy (as long as the PDFs in
policies/ don't change) instead of re-embedding from scratch every
single time — which is what was causing slow/timed-out deploys.

Usage:
    pip install -r requirements.txt
    python build_cache.py

Then:
    git add policies/_embeddings_cache.pkl
    git commit -m "Add precomputed embeddings cache"
    git push
"""

from extract import RagEngine

POLICY_FOLDER = "./policies"  # must match POLICY_FOLDER used by api.py

if __name__ == "__main__":
    print(f"Building RagEngine for '{POLICY_FOLDER}' (verbose)...")
    engine = RagEngine(POLICY_FOLDER, verbose=True)
    print(f"\nDone. Cache written to {POLICY_FOLDER}/_embeddings_cache.pkl")
    print("Commit that file to your repo so future deploys reuse it.")