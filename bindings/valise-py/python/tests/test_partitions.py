# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Partitioned collections, windowed views, search_view, forget_before.

Mirrors tests/db_partition.rs. Time routing uses record.created_at (set via
Record.at), so tests pin timestamps for determinism. NB: the engine requires
created_at to be non-decreasing in commit order — puts below are ascending.
"""

import pytest

import valise

from conftest import DIM, unit_vec

DAY = 86_400
# Three known periods: 1970-01-02, 1970-01-03, 1970-01-04.
D2, D3, D4 = 1 * DAY, 2 * DAY, 3 * DAY


def _make_partitioned(store, by=valise.Partition.BY_DAY):
    return store.partitioned("mem", valise.Schema().text("body"), by)


def _seed(store, p):
    with store.writer() as w:
        w.put_into(p, "old", valise.Record().text("body", "ancient quantum lore").at(D2))
        w.put_into(p, "mid", valise.Record().text("body", "recent quantum news").at(D3))
        w.put_into(p, "new", valise.Record().text("body", "fresh quantum update").at(D4))
        w.commit()


def test_routing_creates_day_partitions(store):
    p = _make_partitioned(store)
    _seed(store, p)
    assert sorted(p.all().collections) == [
        "mem:1970-01-02",
        "mem:1970-01-03",
        "mem:1970-01-04",
    ]
    names = [c.name for c in store.collections()]
    assert "mem:1970-01-03" in names


def test_view_range_and_partitions_windows(store):
    p = _make_partitioned(store)
    _seed(store, p)
    # Inclusive range window covering only day 3.
    v = p.view(valise.Window.range(D3, D3 + DAY - 1))
    assert v.collections == ["mem:1970-01-03"]
    # Explicit suffix selection; unknown suffixes are dropped.
    v = p.view(valise.Window.partitions(["1970-01-02", "1970-01-09"]))
    assert v.collections == ["mem:1970-01-02"]
    # last_days with a pinned "now" (deterministic): [now - 1d, now] spans
    # the day-3/day-4 boundary, so both intersecting partitions are in view.
    v = p.view(valise.Window.last_days(1), now=D4 + DAY // 2)
    assert sorted(v.collections) == ["mem:1970-01-03", "mem:1970-01-04"]


def test_search_view_fuses_globally_with_per_hit_collections(store):
    p = _make_partitioned(store)
    _seed(store, p)
    res = store.search_view(
        p.all(), valise.Search().text("body", "quantum").top_k(10)
    )
    assert len(res) == 3
    assert sorted(res.keys) == ["mid", "new", "old"]
    # Multi-collection result: per-hit collection names are carried.
    assert res.collection is None
    assert sorted(set(res.collections)) == [
        "mem:1970-01-02",
        "mem:1970-01-03",
        "mem:1970-01-04",
    ]
    for hit in res:
        assert hit.collection.startswith("mem:")
    # A narrower view scopes the same query.
    res = store.search_view(
        p.view(valise.Window.partitions(["1970-01-04"])),
        valise.Search().text("body", "quantum").top_k(10),
    )
    assert res.keys == ["new"]
    # Reader exposes the same entry point.
    res = store.reader().search_view(
        p.all(), valise.Search().text("body", "quantum").top_k(10)
    )
    assert len(res) == 3


def test_search_view_with_vectors(store):
    p = store.partitioned(
        "vmem",
        valise.Schema().text("body").vector(
            "dense", valise.Vector(dim=DIM, calibrate=valise.Auto(sample=8))
        ),
        valise.Partition.BY_DAY,
    )
    with store.writer() as w:
        for i in range(8):
            w.put_into(
                p,
                f"d{i}",
                valise.Record()
                .text("body", "doc")
                .vector("dense", unit_vec(700 + i))
                .at(D2 + i * 3600),
            )
        w.commit()
    res = store.search_view(
        p.all(), valise.Search().vector("dense", unit_vec(703)).top_k(3)
    )
    assert res.keys[0] == "d3"


def test_forget_before_tombstones_old_partitions(store):
    p = _make_partitioned(store)
    _seed(store, p)
    # Forget everything whose whole period precedes day 4.
    count = p.forget_before(D4)
    assert count == 2  # "old" + "mid"
    res = store.search_view(p.all(), valise.Search().text("body", "quantum").top_k(10))
    assert res.keys == ["new"]
    # Tombstoned bytes are reclaimable by compaction.
    st = store.stats()
    assert st.frames_total > st.frames_active
    report = store.compact()
    assert report.compacted
    res = store.search_view(p.all(), valise.Search().text("body", "quantum").top_k(10))
    assert res.keys == ["new"]


def test_month_partitioning(store):
    p = _make_partitioned(store, by=valise.Partition.BY_MONTH)
    with store.writer() as w:
        w.put_into(p, "jan", valise.Record().text("body", "january entry").at(15 * DAY))
        w.put_into(p, "feb", valise.Record().text("body", "february entry").at(45 * DAY))
        w.commit()
    assert sorted(p.all().collections) == ["mem:1970-01", "mem:1970-02"]
    assert p.forget_before(32 * DAY) == 1  # all of January precedes day 32


def test_window_requires_factory_and_typed_args(store):
    p = _make_partitioned(store)
    with pytest.raises(TypeError):
        valise.Window()
    with pytest.raises(TypeError):
        p.view("last week")
    with pytest.raises(TypeError):
        store.search_view("not a view", valise.Search().text("body", "q"))
