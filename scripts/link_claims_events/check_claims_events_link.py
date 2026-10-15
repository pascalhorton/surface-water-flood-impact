#!/usr/bin/env python
# coding: utf-8
"""Is the linked events file the one it claims to be?

WHY THIS EXISTS. The claims-events linkage reads a catalogue and writes a
training sample, and nothing in the output records which catalogue it came
from. Three caches sat between the two, each keyed on the damage dataset, the
method and the precipitation dataset - and none on how the events were
DETECTED, which is recorded only in the name of the parquet:

    events_<tag>.pickle                   the parquet, already read
    damages_<dataset>_linked_<label>      the claims, already matched
    events_to_remove_<dataset>_linked_..  the duplicates to drop

So re-extracting a catalogue and re-running the linkage reloaded all three and
never opened the new parquet. The output carried the new name, the new file
date, and the old catalogue. That happened here: a 5-minute catalogue was
re-extracted with the detection step pinned, re-linked, and the result was
byte-identical to the one from five weeks earlier.

The caches are keyed properly now. This script is the acceptance test, because
a cache key is a promise and this file checks the outcome.

WHAT IT CHECKS, in order of how much it would hurt to get wrong:

  1. THE LINKING INVARIANT. The simple method matches a claim to an event
     whose i_max_date falls in [date_claim - 8 h, date_claim + 26 h]
     (SIMPLE_EVENT_HOURS_BEFORE/AFTER in damages.py). Every event carrying a
     claim must satisfy it. 'eid' is a POSITIONAL index into the catalogue
     (Events._add_event_id), so a linkage computed on a different catalogue
     maps each claim to whatever event now sits at that position - and those
     land anywhere in twenty years. This check catches that immediately, and
     it needs nothing but the linked file.

  2. CONTAINMENT. Every event in the linked file must exist in the parquet it
     is supposed to come from. The linked file is a subset - year and location
     selection, the removals, de-duplication - so equality is not expected,
     but containment is. Events the parquet does not hold came from another
     catalogue.

  3. WHICH CATALOGUE IT MATCHES BEST, when more than one parquet is given.
     The decisive form of check 2: hand it the old and the new catalogue and
     it says which one the file was actually built from.

Usage, from the swafi repository root:
    python scripts/link_claims_events/check_claims_events_link.py \
        --dataset gvz --precip-dataset 5min --label step1h_occurrence
    # and, to tell two catalogues apart:
    python scripts/link_claims_events/check_claims_events_link.py \
        --dataset gvz --precip-dataset 5min --label step1h_occurrence \
        --against <new.parquet> <old.parquet>

With no --against, the parquet named by the EVENTS_PATH_SIMPLE_* /
EVENTS_PATH_CLASSIC config entry is used, which is the one the linkage read.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from swafi.config import Config
from swafi.damages import SIMPLE_EVENT_HOURS_BEFORE, SIMPLE_EVENT_HOURS_AFTER
from swafi.events import get_events_filename, load_events_from_pickle
from swafi.utils.event_extraction import detection_tag_in

logger = logging.getLogger(__name__)
config = Config()

# How much of the linked file may be absent from the parquet before the two are
# called different catalogues. Not zero: a cell can be dropped from the parquet
# side by the column subset read here only if something is badly wrong, but
# floating-point date handling across parquet engines has surprised this
# project before, so a hair of tolerance keeps the verdict about catalogues.
CONTAINMENT_TOLERANCE_PCT = 0.01


def _config_key(event_method, precip_dataset):
    if event_method == 'classic':
        return 'EVENTS_PATH_CLASSIC'
    return f'EVENTS_PATH_SIMPLE_{precip_dataset.upper()}'


def _load_linked(dataset, label, event_method, precip_dataset):
    filename = get_events_filename(dataset, label, event_method, precip_dataset)
    print(f"  linked events file   {filename}")
    events = load_events_from_pickle(filename=filename).events
    print(f"    {len(events):,} events, {events['cid'].nunique():,} cells")
    if 'target' in events.columns:
        n_pos = int((pd.to_numeric(events['target'], errors='coerce') > 0).sum())
        print(f"    {n_pos:,} carrying damage "
              f"({100 * n_pos / max(1, len(events)):.3f}% base rate)")
    return events


def _load_catalogue(path, label):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{label}: no such file: {path}")
    want = ['cid', 'e_date', 'i_max_date']
    available = set(pq.read_schema(path).names)
    df = pd.read_parquet(path, columns=[c for c in want if c in available])
    tag = detection_tag_in(path) or '(none)'
    print(f"  {label}")
    print(f"    {path.name}")
    print(f"    {len(df):,} events, {df['cid'].nunique():,} cells, "
          f"detection tag {tag}")
    return df


def _keys(df):
    """The identity of an event: its cell and its peak instant.

    i_max_date rather than e_date, because it is the native-resolution
    timestamp the linkage matches on, so it survives the dating rule and tells
    two catalogues apart where the floored date cannot.
    """
    return set(zip(df['cid'].to_numpy(),
                   pd.to_datetime(df['i_max_date']).to_numpy()))


def check_linking_invariant(events):
    """Every event with a claim must sit in the window the claim was matched in.

    This is the check that catches a linkage computed on another catalogue,
    and it reads nothing but the file under test.
    """
    print("=" * 76)
    print("1. THE LINKING INVARIANT")
    print("=" * 76)

    if not {'date_claim', 'i_max_date'} <= set(events.columns):
        print("  no date_claim column: this file carries no claim dates, so the")
        print("  invariant cannot be checked. It is written by the linkage for")
        print("  events that matched a claim, so a file without it either")
        print("  predates that or holds no positives.")
        return None

    linked = events.dropna(subset=['date_claim']).copy()
    if linked.empty:
        print("  no event carries a claim date: nothing to check.")
        return None

    offset_h = ((pd.to_datetime(linked['i_max_date'])
                 - pd.to_datetime(linked['date_claim']))
                / pd.Timedelta(hours=1))
    lo, hi = -SIMPLE_EVENT_HOURS_BEFORE, SIMPLE_EVENT_HOURS_AFTER
    inside = offset_h.between(lo, hi)
    n_out = int((~inside).sum())

    print(f"  events carrying a claim   {len(linked):,}")
    print(f"  the matching window       i_max_date - date_claim in "
          f"[{lo:+.0f}, {hi:+.0f}] h")
    print(f"  observed                  {offset_h.min():+.1f} to "
          f"{offset_h.max():+.1f} h, median {offset_h.median():+.1f} h")
    print(f"  outside the window        {n_out:,} "
          f"({100 * n_out / len(linked):.2f}%)")

    if n_out == 0:
        print()
        print("  PASSES. Every claim sits in the window its event was matched")
        print("  in, so the claims were matched against THIS catalogue.")
        return True

    # Say how badly, because a handful and a uniform spread mean different things.
    spread = offset_h[~inside].abs()
    print(f"  of those, |offset|        median {spread.median():,.0f} h "
          f"({spread.median() / 24:,.0f} days), max {spread.max() / 24:,.0f} days")
    print()
    print("  FAILS. A claim cannot have been matched to an event outside its")
    print("  own matching window, so the 'eid' the claims carry are positions")
    print("  into a DIFFERENT catalogue than the events in this file. The")
    print("  cause is a linked-damages pickle, or an events-to-remove list,")
    print("  computed on another catalogue and reloaded:")
    print("    damages_<dataset>_linked_<label>.pickle")
    print("    events_to_remove_<dataset>_linked_<label>.pickle")
    print("  Delete both and re-run compute_claims_events_link.py. Everything")
    print("  trained on this file is measuring a scrambled target.")
    return False


def check_containment(events, catalogues):
    """Every event in the linked file must exist in the catalogue it came from.

    THE FIRST CATALOGUE IS THE ONE UNDER TEST - the config's, or the first
    --against. Any others are there to diagnose, and the verdict is about the
    first: a file that is cleanly contained in a catalogue is not vindicated
    by that, if the catalogue it is contained in is not the one the linkage
    was supposed to read. That is precisely the failure this script was
    written for, and judging on the best match would pass it.
    """
    print()
    print("=" * 76)
    print("2. CONTAINMENT, and which catalogue this file came from")
    print("=" * 76)

    if 'i_max_date' not in events.columns:
        print("  no i_max_date column: cannot identify the events.")
        return None

    linked_keys = _keys(events)
    print(f"  the linked file holds {len(linked_keys):,} distinct (cell, peak) "
          f"events")
    print()

    results = []
    for label, df in catalogues:
        cat_keys = _keys(df)
        missing = len(linked_keys - cat_keys)
        pct = 100 * missing / max(1, len(linked_keys))
        kept = 100 * len(linked_keys & cat_keys) / max(1, len(cat_keys))
        results.append((label, missing, pct, kept))
        print(f"  against {label}")
        print(f"    in the linked file but NOT in this catalogue   "
              f"{missing:,} ({pct:.2f}%)")
        print(f"    share of this catalogue that survived          {kept:.2f}%")

    print()
    under_test = results[0]
    best = min(results, key=lambda r: r[1])

    if len(results) > 1 and best is not under_test:
        print(f"  THIS FILE CAME FROM {best[0].upper()}, not from "
              f"{under_test[0]}:")
        print(f"    {best[0]:<14} accounts for all but {best[2]:.2f}% of it")
        print(f"    {under_test[0]:<14} leaves {under_test[2]:.2f}% unaccounted "
              f"for")
        print()
        print("  FAILS. The file is internally consistent - the claims were")
        print("  matched against the catalogue it really came from - so no")
        print("  check inside it can object. It is simply the wrong")
        print("  catalogue, which is what a cache keyed on too little gives")
        print("  you: the linkage reloaded a previous run instead of reading")
        print("  the parquet it was pointed at. Delete these three, which are")
        print("  keyed on the event definition now, and re-run:")
        print("    events_<tag><detection tag>.pickle")
        print("    damages_<dataset>_linked_<label><detection tag>.pickle")
        print("    events_to_remove_<dataset>_linked_<label><detection tag>.pickle")
        return False

    if len(results) > 1:
        print(f"  Best match is {best[0]}, the one under test.")
        worse = [r for r in results[1:] if r[1] > under_test[1]]
        if len(worse) < len(results) - 1:
            print("  But another catalogue accounts for it equally well, so")
            print("  these catalogues cannot be told apart on this file: they")
            print("  barely differ, or the file matches none of them.")
        print()

    if under_test[1] == 0:
        print("  PASSES. Every event in the linked file exists in the")
        print("  catalogue, so the linkage read it.")
        return True
    if under_test[2] <= CONTAINMENT_TOLERANCE_PCT:
        print(f"  PASSES, within tolerance ({under_test[2]:.3f}% of events "
              f"unaccounted for, {CONTAINMENT_TOLERANCE_PCT}% allowed).")
        return True

    print(f"  FAILS. The linked file holds events {under_test[0]} does not, so")
    print("  it was not built from it. The linkage reads the parquet named by")
    print("  the EVENTS_PATH_SIMPLE_* config entry and caches it in")
    print("  'events_<tag><detection tag>.pickle': check that the config")
    print("  points where you think, and delete that dump to force a re-read.")
    return False


def report_sample(events, catalogues):
    """What the linked sample is, next to the catalogue it came from."""
    print()
    print("=" * 76)
    print("3. WHAT THE LINKAGE KEPT")
    print("=" * 76)
    label, df = catalogues[0]
    cat_cells = set(df['cid'].unique())
    linked_cells = set(events['cid'].unique())
    print(f"  {label:<24} {len(df):,} events over {len(cat_cells):,} cells")
    print(f"  {'linked sample':<24} {len(events):,} events over "
          f"{len(linked_cells):,} cells")
    kept = 100 * len(linked_cells & cat_cells) / max(1, len(cat_cells))
    print(f"  {'cells kept':<24} {kept:.1f}% of the catalogue's")
    outside = len(linked_cells - cat_cells)
    if outside:
        print(f"  !! {outside:,} cells in the linked file are absent from the "
              f"catalogue")

    if 'e_date' in events.columns:
        dates = pd.to_datetime(events['e_date'])
        print(f"  period                   {dates.min().date()} to "
              f"{dates.max().date()}")
    print()
    print("  The linkage drops the years and cells without contracts, the")
    print("  events matched to a claim that another event matched better, the")
    print("  events left without contracts, and the duplicates. So a sample")
    print("  far smaller than the catalogue is expected; a sample holding")
    print("  events the catalogue does not is not.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='gvz', choices=['gvz', 'mobiliar'],
                        help="the damage dataset")
    parser.add_argument('--precip-dataset', default='5min',
                        choices=['hourly', '5min'],
                        help="the precipitation dataset the events came from")
    parser.add_argument('--event-method', default='simple',
                        choices=['simple', 'classic'],
                        help="the event extraction method")
    parser.add_argument('--label', default='default_occurrence',
                        help="the event file label the linkage wrote "
                             "(LABEL_RESULTING_FILE, e.g. step1h_occurrence)")
    parser.add_argument('--against', nargs='+', default=None, metavar='PARQUET',
                        help="the catalogue(s) to check against; default is "
                             "the one the config names. Give two to find out "
                             "which one the file came from.")
    args = parser.parse_args()

    if args.event_method == 'classic':
        print("NOTE: the invariant checked here is the simple method's "
              "matching window;\n      the classic method matches over "
              "window_days and is not checked.")
        print()

    print("=" * 76)
    print("FILES")
    print("=" * 76)
    events = _load_linked(args.dataset, args.label, args.event_method,
                          args.precip_dataset)
    print()

    paths = args.against
    if not paths:
        key = _config_key(args.event_method, args.precip_dataset)
        paths = [config.get(key)]
        print(f"  (catalogue from the config entry {key})")
    labels = ([f"catalogue {i + 1}" for i in range(len(paths))]
              if len(paths) > 1 else ["catalogue"])
    catalogues = [(label, _load_catalogue(path, label))
                  for label, path in zip(labels, paths)]
    print()

    invariant = check_linking_invariant(events)
    contained = check_containment(events, catalogues)
    report_sample(events, catalogues)

    print()
    print("=" * 76)
    print("VERDICT")
    print("=" * 76)
    failed = [name for name, ok in (("the linking invariant", invariant),
                                    ("containment", contained)) if ok is False]
    skipped = [name for name, ok in (("the linking invariant", invariant),
                                     ("containment", contained)) if ok is None]
    if failed:
        print(f"  FAILED: {', '.join(failed)}.")
        print("  Do not train on this file. Delete the caches named above and")
        print("  re-run compute_claims_events_link.py.")
        return 1
    if skipped:
        print(f"  INCONCLUSIVE: {', '.join(skipped)} could not be checked.")
        return 2
    print("  PASSES. The claims were matched against this catalogue and every")
    print("  event in the sample comes from it, so the file is what its name")
    print("  says it is.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
