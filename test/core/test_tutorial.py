import pytest
import uxarray as ux


def test_available_datasets():
    assert "quad-hexagon" in ux.tutorial.available_datasets()


def test_file_path():
    path = ux.tutorial.file_path("quad-hexagon", component="grid")
    assert path.exists()


def test_open_dataset():
    uxds = ux.tutorial.open_dataset("quad-hexagon")
    assert uxds.uxgrid is not None


def test_open_grid():
    uxgrid = ux.tutorial.open_grid("outCSne30")
    assert uxgrid.n_face > 0


def test_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown tutorial dataset"):
        ux.tutorial.open_dataset("not-a-dataset")


def test_grid_only_dataset_open_dataset_error():
    with pytest.raises(ValueError, match="does not define a data file"):
        ux.tutorial.open_dataset("outCSne30")


def test_open_mfdataset():
    uxds = ux.tutorial.open_mfdataset("quad-hexagon-random")
    assert uxds.uxgrid is not None


def test_open_dataset_rejects_multifile():
    with pytest.raises(ValueError, match="multiple data files"):
        ux.tutorial.open_dataset("quad-hexagon-random")


def test_file_paths():
    paths = ux.tutorial.file_paths("quad-hexagon-random")
    assert len(paths) == 3
    assert all(path.exists() for path in paths)
