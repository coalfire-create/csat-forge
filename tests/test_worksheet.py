from csat_forge.cli import main


def test_worksheet_cli(tmp_path, capsys):
    out = tmp_path / "w.html"
    assert main(["worksheet", "-n", "4", "--killer", "1", "--seed", "1", "--out", str(out)]) == 0
    h = out.read_text(encoding="utf-8")
    assert h.count('class="no"') == 5 and "정답 및 해설" in h
