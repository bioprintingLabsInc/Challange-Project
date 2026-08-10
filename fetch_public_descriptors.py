#!/usr/bin/env python3
"""Fetch auditable PubChem descriptors for the EPA Carstens chemical panel."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


PROPERTIES = (
    "CanonicalSMILES,IsomericSMILES,MolecularFormula,MolecularWeight,XLogP,"
    "TPSA,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,Complexity,Charge"
)
URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/"
    f"property/{PROPERTIES}/JSON"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carstens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser.parse_args()


def fetch(query: str) -> dict:
    url = URL.format(query=urllib.parse.quote(query, safe=""))
    request = urllib.request.Request(url, headers={"User-Agent": "DNT-six-domain-model/1.0"})
    last_error = ""
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return {"status": "resolved", "url": url, "payload": json.load(response)}
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {"status": "not_found", "url": url, "error": "HTTP 404"}
            last_error = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(1.5 * (attempt + 1))
    return {"status": "error", "url": url, "error": last_error}


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_excel(args.carstens)
    identity = data[["casn", "chnm", "dsstox_substance_id"]].drop_duplicates("casn")
    rows = []
    for number, row in enumerate(identity.itertuples(index=False), start=1):
        cas = str(row.casn).strip()
        cache = args.cache_dir / f"{cas.replace('/', '_')}.json"
        if cache.exists():
            record = json.loads(cache.read_text())
        else:
            record = fetch(cas)
            record["query"] = cas
            cache.write_text(json.dumps(record, indent=2))
            time.sleep(0.22)
        output = {
            "casn": cas,
            "chemical": row.chnm,
            "dsstox_substance_id": row.dsstox_substance_id,
            "pubchem_query": cas,
            "match_status": record.get("status", "error"),
            "source_url": record.get("url", ""),
            "error": record.get("error", ""),
        }
        try:
            output.update(record["payload"]["PropertyTable"]["Properties"][0])
        except (KeyError, IndexError, TypeError):
            pass
        rows.append(output)
        print(f"[{number:02d}/{len(identity)}] {cas}: {output['match_status']}", flush=True)
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {args.output}; resolved {(result.match_status == 'resolved').sum()}/{len(result)}")


if __name__ == "__main__":
    main()
