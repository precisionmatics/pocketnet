from pocketnet.cli import main


def test_predict_cli(example, tmp_path, capsys):
    assert main(["predict", str(example), "-o", str(tmp_path), "--device", "cpu", "--formats", "csv,json"]) == 0
    assert (tmp_path / "3POZ_pockets.csv").exists() and (tmp_path / "3POZ_pockets.json").exists() and "pockets" in capsys.readouterr().out


def test_bad_format_and_missing_input(example, tmp_path):
    assert main(["predict", str(example), "-o", str(tmp_path), "--formats", "xyz"]) == 2
    assert main(["predict", str(tmp_path / "empty_dir_that_does_not_exist.pdb"), "-o", str(tmp_path), "--device", "cpu", "-q"]) == 1


def test_info(capsys):
    assert main(["info"]) == 0 and "bundled models" in capsys.readouterr().out
