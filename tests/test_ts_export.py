from kalshi_btc_bot.ts_export import read_ts_export


def test_reads_data_window_export(tmp_path):
    f = tmp_path / "SPY.txt"
    f.write_text('"Date","Time","Open","High","Low","Close","Up","Down"\n'
                 '12/01/2025,09:31,680.10,680.50,680.00,680.40,1200,800\n'
                 '12/01/2025,09:30,680.00,680.20,679.90,680.10,1500,900\n')
    df = read_ts_export(f)
    assert str(df.index[0]) == "2025-12-01 09:30:00-05:00"
    assert df["Volume"].tolist() == [2400, 2000]
    assert df["Close"].iloc[-1] == 680.40


def test_daily_export_without_time(tmp_path):
    f = tmp_path / "ES.csv"
    f.write_text("Date,Open,High,Low,Close,Vol,OI\n12/02/2025,6800,6850,6790,6840,100,5\n")
    df = read_ts_export(f)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume", "OpenInterest"]
