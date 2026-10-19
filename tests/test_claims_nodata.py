"""
Nodata is not a claim.

The claim rasters carry -9999 for nodata. Damages._extract_claims_from_grids
selected cells with np.nonzero(), which is true for -9999 as readily as for 1,
so any cell inside the EXPOSURE mask that is nodata in the CLAIM grids
produced one phantom claim per date.

The two masks do not have to agree: the exposure mask marks cells with at
least one annual exposure, and a cell can carry exposure while the claim
rasters have no data there. One such cell existed in the mobiliar set, at
(2,595,500, 1,272,500), and it generated **952 phantom claims** - 9.3% of the
whole dataset - spread over 952 distinct dates between 2013 and 2022.

They were visible only because that cell also has cid == 0, which kept them
out of the catalogue and out of training. A nodata cell with a valid cid would
have produced spurious POSITIVES that link to real events, which is why this
is pinned rather than left to the reader.

The signature in the data was the value: selection -9999, and -19998, -29997,
-39996 where select_categories_type() summed two, three or four categories.
"""
import numpy as np
import pytest

from swafi.damages import Damages

NODATA = -9999


class _Reader(Damages):
    """The extraction machinery, without the file IO or the domain."""

    def __init__(self, mask):
        # Deliberately not calling super().__init__: this test is about one
        # method, and building a real Damages needs rasters, a CID file and a
        # domain that are not available here.
        self.mask = {'mask': mask, 'shape': mask.shape}


def _grids(shape, dates, claims, nodata_cells):
    """Claim grids holding some real claims and some nodata cells."""
    data = np.zeros((len(dates),) + shape, dtype=float)
    for t, (row, col), value in claims:
        data[t, row, col] = value
    for row, col in nodata_cells:
        data[:, row, col] = NODATA
    return data


def test_nodata_inside_the_exposure_mask_is_not_a_claim():
    """The regression: a masked cell that is nodata in the claim grids."""
    mask = np.zeros((4, 5), dtype=bool)
    mask[1, 1] = True          # a cell with real claims
    mask[2, 3] = True          # a cell that is nodata in the claim grids
    dates = ['2013-01-29', '2013-02-01', '2013-02-02']

    data = _grids((4, 5), dates,
                  claims=[(0, (1, 1), 1.0), (2, (1, 1), 3.0)],
                  nodata_cells=[(2, 3)])

    df = _Reader(mask)._extract_claims_from_grids(data, dates, 'selection')

    assert len(df) == 2, (
        f"expected the 2 real claims, got {len(df)} - the nodata cell "
        f"produced one phantom per date")
    assert (df['selection'] > 0).all(), "a claim is a count and is positive"
    assert sorted(df['selection']) == [1.0, 3.0]
    assert NODATA not in df['selection'].to_numpy()


def test_one_phantom_per_date_was_the_old_behaviour():
    """What the bug looked like, so the test cannot pass for the wrong reason.

    np.nonzero() over the same grids would have returned the two real claims
    plus one row per date for the nodata cell. If this ever matches the
    function's output again, the fix has been undone.
    """
    mask = np.zeros((4, 5), dtype=bool)
    mask[1, 1] = True
    mask[2, 3] = True
    dates = ['2013-01-29', '2013-02-01', '2013-02-02']
    data = _grids((4, 5), dates,
                  claims=[(0, (1, 1), 1.0), (2, (1, 1), 3.0)],
                  nodata_cells=[(2, 3)])

    masked = data[:, mask]
    buggy = len(np.nonzero(masked)[0])
    fixed = len(_Reader(mask)._extract_claims_from_grids(data, dates, 'selection'))

    assert buggy == 2 + len(dates), "the old selection took nodata as a claim"
    assert fixed < buggy


def test_a_cell_with_no_claim_at_all_contributes_nothing():
    mask = np.ones((3, 3), dtype=bool)
    dates = ['2020-06-01', '2020-06-02']
    data = np.zeros((2, 3, 3))

    df = _Reader(mask)._extract_claims_from_grids(data, dates, 'selection')

    assert df.empty
    assert list(df.columns) == ['date_claim', 'mask_index', 'selection']


def test_nodata_outside_the_mask_was_never_the_problem():
    """Cells outside the exposure mask are dropped before the selection, so
    they could not produce phantoms either way. Pinned so the fix is not
    mistaken for having addressed that case."""
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    dates = ['2020-06-01']
    data = np.full((1, 3, 3), NODATA, dtype=float)
    data[0, 0, 0] = 2.0

    df = _Reader(mask)._extract_claims_from_grids(data, dates, 'selection')

    assert len(df) == 1
    assert df['selection'].iloc[0] == 2.0


@pytest.mark.parametrize("value", [-9999, -19998, -29997, -39996])
def test_the_summed_nodata_multiples_are_also_refused(value):
    """select_categories_type() sums the categories, so a cell that is nodata
    in several rasters arrives as a multiple of -9999. Those were the values
    actually observed in the mobiliar claims frame."""
    mask = np.ones((2, 2), dtype=bool)
    dates = ['2020-06-01']
    data = np.zeros((1, 2, 2))
    data[0, 0, 0] = value

    df = _Reader(mask)._extract_claims_from_grids(data, dates, 'selection')

    assert df.empty, f"{value} is nodata, not {value} claims"


def test_extract_non_null_claims_agrees():
    """The legacy per-date path had the same selection and the same bug."""
    mask = np.ones((2, 2), dtype=bool)
    data = np.array([[5.0, NODATA], [0.0, 2.0]])

    indices, values = _Reader(mask)._extract_non_null_claims(data)

    assert sorted(values) == [2.0, 5.0]
    assert NODATA not in values
    assert len(indices) == 2
