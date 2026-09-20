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
