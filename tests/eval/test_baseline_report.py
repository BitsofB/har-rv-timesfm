import pandas as pd

from src.eval.baseline_report import compute_baseline_metrics, write_baseline_report


def test_compute_baseline_metrics_perfect_prediction_is_zero_qlike():
    y = pd.Series([1.0, 2.0, 3.0])
    metrics = compute_baseline_metrics({"perfect": (y, y)})

    assert list(metrics.columns) == ["qlike", "mse", "mae", "r2"]
    assert metrics.loc["perfect", "qlike"] == 0.0
    assert metrics.loc["perfect", "mse"] == 0.0
    assert metrics.loc["perfect", "r2"] == 1.0


def test_write_baseline_report_writes_markdown_table(tmp_path):
    metrics = compute_baseline_metrics({
        "perfect": (pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0])),
    })
    out_path = tmp_path / "baseline_metrics.md"

    write_baseline_report(metrics, out_path=str(out_path))

    content = out_path.read_text()
    assert "qlike" in content
    assert "perfect" in content


def test_write_baseline_report_writes_notes_when_provided(tmp_path):
    metrics = compute_baseline_metrics({
        "perfect": (pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0])),
    })
    out_path = tmp_path / "baseline_metrics.md"
    notes = "## Notes\n\nData feed: Alpaca IEX, not SIP consolidated tape."

    write_baseline_report(metrics, out_path=str(out_path), notes=notes)

    content = out_path.read_text()
    assert "qlike" in content
    assert notes in content


def test_write_baseline_report_omits_notes_section_when_not_provided(tmp_path):
    """No notes arg (or None) must behave exactly as before -- no regression."""
    metrics = compute_baseline_metrics({
        "perfect": (pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0])),
    })
    out_path = tmp_path / "baseline_metrics.md"

    write_baseline_report(metrics, out_path=str(out_path))

    content = out_path.read_text()
    expected = "# Baseline metrics\n\n" + metrics.to_markdown() + "\n"
    assert content == expected


def test_write_baseline_report_creates_nested_directories(tmp_path):
    """Verify that write_baseline_report creates parent directories if they don't exist."""
    metrics = compute_baseline_metrics({
        "perfect": (pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0])),
    })
    nested_path = tmp_path / "nested" / "subdir" / "baseline_metrics.md"

    # nested/subdir/ doesn't exist yet
    assert not nested_path.parent.exists()

    write_baseline_report(metrics, out_path=str(nested_path))

    # Verify the file was created and contains expected content
    assert nested_path.exists()
    content = nested_path.read_text()
    assert "qlike" in content
    assert "perfect" in content
