#!/usr/bin/env python
# coding: utf-8
"""How far did the nodata-as-claims bug reach?

THE BUG. Damages._extract_claims_from_grids selected claim cells with
np.nonzero(), which is true for the rasters' nodata (-9999) as readily as for
a real count. Any cell inside the EXPOSURE mask that is nodata in the CLAIM
grids therefore produced one phantom claim per date, with selection -9999 - or
a multiple of it once select_categories_type() summed the categories.

Found in the mobiliar set: 952 phantom claims at a single cell,
(2,595,500, 1,272,500), spread over 952 distinct dates, 9.3% of the whole
dataset. Fixed on 2026-10-19 (`np.nonzero(masked > 0)`), with
tests/test_claims_nodata.py pinning it.

WHY THIS SCRIPT EXISTS ANYWAY. Two reasons the fix is not the end of it:

  1. **The claims are cached.** Damages reloads from
     `damages_<name>_<y0>-<y1>.pickle`, so the phantoms stay in every existing
     pickle, and in everything built from one, until those are deleted and the
     rasters re-read. The fix changes nothing on its own.

  2. **Those 952 were visible only by luck.** Their cell has cid == 0, which
     kept them out of the catalogue and out of training. A nodata cell with a
     VALID cid would produce phantom claims that link to real events and
     become spurious positives - silent, and directly harmful. Nothing checked
     for that, so nothing knows whether it happened.

WHAT IT REPORTS, per dataset, from whatever is currently cached:

  * claims with selection <= 0, split by whether their cell has a cid - the
    second group is the one that reached training
  * which dates and cells they sit on
  * the same check on the exposure, where a negative value would corrupt
    nb_contracts (a model feature, and the Poisson head's offset)
  * what to delete to rebuild cleanly

Usage, from the swafi repository root:
    python scripts/data_analyses/audit_claims_nodata.py
    python scripts/data_analyses/audit_claims_nodata.py --dataset gvz
"""
import argparse
import sys

import numpy as np
import pandas as pd

from swafi.config import Config
from swafi.utils.logging_setup import setup_logging
from swafi.utils.use_common import get_damages

config = Config()


def audit(dataset):
    suffix = dataset.upper()
    y_start = config.get(f"YEAR_START_{suffix}")
    y_end = config.get(f"YEAR_END_{suffix}")

    print("=" * 78)
    print(f"{dataset.upper()}  {y_start}-{y_end}")
    print("=" * 78)

    damages, removed = get_damages(dataset, y_start, y_end)
    claims = damages.claims
    n = len(claims)

    bad = claims[claims["selection"] <= 0]
    print(f"  claims in the cached frame        {n:,}")
    print(f"  with selection <= 0 (nodata)      {len(bad):,}"
          f"  ({100 * len(bad) / n:.2f}%)")

    if len(bad):
        no_cid = bad[bad["cid"] == 0]
        with_cid = bad[bad["cid"] != 0]
        print(f"    of which cid == 0 (harmless: no events there)  "
              f"{len(no_cid):,}")
        print(f"    of which cid != 0  **THESE REACHED TRAINING**   "
              f"{len(with_cid):,}")
        print()
        print(f"    values: "
              + ", ".join(f"{v} x{c:,}" for v, c in
                          bad["selection"].value_counts().head(6).items()))
        cells = bad.value_counts(["x", "y"]).head(5)
        print(f"    cells ({bad.value_counts(['x', 'y']).shape[0]} distinct, "
              f"top 5):")
        for (x, y), c in cells.items():
            flag = ""
            match = bad[(bad["x"] == x) & (bad["y"] == y)]
            if (match["cid"] != 0).any():
                flag = "   <-- HAS A CID: spurious positives in training"
            print(f"      x {x:,.0f}  y {y:,.0f}   {c:,} claims{flag}")
        if len(with_cid):
            print()
            print("    !! Those claims were linked to real events and became")
            print("       positives. Every model trained on this dataset has")
            print("       learned from them. Rebuild before trusting any of it.")
    else:
        print("    none - this frame is clean (or already rebuilt)")

    # ---- the exposure side -------------------------------------------------
    print()
    exposure = damages.exposure
    bad_exp = exposure[exposure["selection"] < 0]
    print(f"  exposure rows                     {len(exposure):,}")
    print(f"  with selection < 0 (nodata)       {len(bad_exp):,}"
          f"  ({100 * len(bad_exp) / max(1, len(exposure)):.2f}%)")
    if len(bad_exp):
        print("    A negative exposure feeds get_contracts_number(), so it")
        print("    corrupts nb_contracts - a model feature, and the Poisson")
        print("    head's offset. _create_mask() marks a cell on max > 0, so a")
        print("    cell can be masked on one year and nodata in another.")
        aff = bad_exp.value_counts(["x", "y"]).shape[0]
        print(f"    {aff:,} distinct cells affected; "
              f"{len(set(bad_exp['cid']) - {0}):,} of them have a cid")
    else:
        print("    none")

    if removed is not None and len(removed):
        # STRICTLY negative, not <= 0: a removed claim is one whose SELECTED
        # categories sum to zero, so 0 is its normal value and testing <= 0
        # flags the whole frame. Only a negative is nodata here.
        bad_rm = removed[removed["selection"] < 0] if "selection" in removed else []
        print()
        print(f"  removed claims (other categories) {len(removed):,}, "
              f"of which nodata {len(bad_rm):,}")
    print()
    return len(bad), len(bad_exp)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="both",
                        choices=["both", "gvz", "mobiliar"])
    args = parser.parse_args()
    setup_logging(script_name="audit_claims_nodata")

    datasets = ["gvz", "mobiliar"] if args.dataset == "both" else [args.dataset]
    total = 0
    for ds in datasets:
        try:
            n_claims, _n_exp = audit(ds)
            total += n_claims
        except (FileNotFoundError, OSError, RuntimeError) as err:
            print(f"  {ds}: not available here ({err})")
            print()

    print("=" * 78)
    print("TO REBUILD")
    print("=" * 78)
    if not total:
        print("  Nothing to do: no nodata survived into these frames.")
        return 0
    print("  The fix only applies when the rasters are re-read. Delete, in")
    print("  PICKLES_DIR, and re-run the pipeline from the linkage:")
    for ds in datasets:
        suffix = ds.upper()
        y0, y1 = config.get(f"YEAR_START_{suffix}"), config.get(f"YEAR_END_{suffix}")
        print(f"    damages_{ds}_{y0}-{y1}.pickle"
              f"            <- the claims and exposure")
        print(f"    damages_{ds}_{y0}_{y1}_xr.pickle"
              f"         <- the assessment grid built from them")
        print(f"    damages_{ds}_linked_*.pickle"
              f"             <- the claim/event linkage")
        print(f"    events_to_remove_{ds}_linked_*.pickle")
        print(f"    events_{ds}_with_target_*.pickle"
              f"        <- the training samples")
    print()
    print("  Then: compute_claims_events_link.py, then")
    print("  check_claims_events_link.py to confirm, then re-train anything")
    print("  whose numbers are going to be quoted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
