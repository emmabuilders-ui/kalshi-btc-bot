import gzip

from kalshi_btc_bot.dashboard import read_price_hour


def test_cut_off_member_does_not_merge_lines(tmp_path):
    d = tmp_path / "2026-11-05"
    d.mkdir()
    f = d / "12.csv.gz"
    body = b"".join(b"%d,80000.5,brti\n" % t for t in range(1793880000, 1793881851)) + b"1793881851,80010"
    first = gzip.compress(body)[:-8]         # a run killed mid-write: no gzip trailer, ends in half a line
    second = gzip.compress(b"1793883000,79661.83,brti\n1793883001,79662.0,brti\n")
    f.write_bytes(first + second)
    pts = read_price_hour(f, 1)
    assert all(1793880000 <= t < 1793883600 for t, _ in pts)
    assert pts[-1] == [1793883001, 79662.0]
    assert [1793880000, 80000.5] in pts


def test_times_outside_the_hour_are_dropped(tmp_path):
    d = tmp_path / "2026-11-05"
    d.mkdir()
    f = d / "12.csv.gz"
    f.write_bytes(gzip.compress(b"1793880005,1.5,x\n17938818511793883000,2.5,x\n1793883600,3.5,x\n"))
    assert read_price_hour(f, 5) == [[1793880005, 1.5]]
