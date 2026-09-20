from shelf.fuse import Sighting, Station
from shelf.store import Store


def make(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    s.init()
    return s


def test_sighting_creates_an_item_when_the_class_is_new(tmp_path):
    s = make(tmp_path)
    iid = s.record_sighting(Sighting("cup", 0.9, 10.0, 2.0, 12.0, 30.0))
    items = s.items()
    assert len(items) == 1
    assert items[0]["id"] == iid
    assert items[0]["cls_name"] == "cup"
    assert items[0]["source"] == "fusion"


def test_second_sighting_updates_rather_than_duplicates(tmp_path):
    s = make(tmp_path)
    a = s.record_sighting(Sighting("cup", 0.9, 10.0, 2.0, 12.0, 30.0))
    b = s.record_sighting(Sighting("cup", 0.7, 20.0, 3.0, 40.0, 50.0))
    assert a == b
    assert len(s.items()) == 1
    assert s.items()[0]["confidence"] == 0.7


def test_stations_round_trip(tmp_path):
    s = make(tmp_path)
    s.upsert_station(Station("s1", "Aisle", 50.0, 34.0, 90.0))
    got = s.stations()
    assert len(got) == 1
    assert got[0].heading == 90.0


def test_concurrent_sightings_of_one_class_make_one_item(tmp_path):
    """Task 6 writes from the capture thread; check-then-act would duplicate.

    Asserting on row count alone is not enough: unlocked, the likely failure is
    an IntegrityError from two threads minting the same millisecond-based id,
    which kills one worker silently and still leaves exactly one row.
    """
    import threading
    s = make(tmp_path)
    barrier = threading.Barrier(8)
    errors = []

    def worker():
        barrier.wait()                      # maximise the overlap
        try:
            s.record_sighting(Sighting("cup", 0.9, 10.0, 2.0, 12.0, 30.0))
        except Exception as e:              # noqa: BLE001 - the point is to see any
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert errors == [], f"worker raised: {errors!r}"
    assert len(s.items()) == 1


def test_tags_round_trip_as_a_list(tmp_path):
    s = make(tmp_path)
    s.record_sighting(Sighting("cup", 0.9, 10.0, 2.0, 12.0, 30.0))
    assert s.items()[0]["tags"] == ["cup"]
