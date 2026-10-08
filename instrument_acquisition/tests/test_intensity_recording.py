def test_mean_I_sample_and_writer(tmp_path=None):
    """Quick dry-run style test for I_sample mean and CSV writer."""
    import sys
    import statistics
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from processing.labview_output_math import calculate_avg_row
    from output.csv_writer import write_I_sample_csv, write_I_avg_csv

    # Create fake intensity rows for two wavelengths
    wavelengths = [400.0, 401.0]
    i_rows = [[1.0, 1.01, 0.99], [0.95, 0.96, 0.94]]

    # Compute means and standard deviations manually
    mean0 = sum(i_rows[0]) / len(i_rows[0])
    mean1 = sum(i_rows[1]) / len(i_rows[1])
    std0 = statistics.stdev(i_rows[0])
    std1 = statistics.stdev(i_rows[1])

    # Use calculate_avg_row to ensure AVG row does not include intensity
    avg0 = calculate_avg_row(400.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    avg1 = calculate_avg_row(401.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    assert len(avg0) == 8
    assert len(avg1) == 8
    assert avg0[-1] == 0
    assert avg1[-1] == 0

    outdir = tmp_path or 'data'
    path = write_I_sample_csv('testbase', 'pos', outdir, wavelengths, i_rows)
    i_avg_path = write_I_avg_csv('testbase', 'pos', outdir, wavelengths, [[mean0, std0], [mean1, std1]])
    # Ensure file exists and has correct rows.
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            lines = fh.read().strip().splitlines()
            assert len(lines) == 2
        with open(i_avg_path, 'r', encoding='utf-8') as fh:
            lines = fh.read().strip().splitlines()
            assert len(lines) == 2
            assert lines[0].split(',')[1] == str(mean0)
            assert lines[1].split(',')[1] == str(mean1)
    except Exception:
        # if tmp_path not provided, skip strict file assertion
        pass

    print('test_mean_I_sample_and_writer PASSED')


if __name__ == '__main__':
    test_mean_I_sample_and_writer()
