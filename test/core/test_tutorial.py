from pathlib import Path

import pytest
import uxarray as ux
import uxarray.tutorial as tutorial


@pytest.fixture(autouse=True)
def set_tutorial_data_dir(monkeypatch):
    meshfiles = Path(__file__).resolve().parents[1] / "meshfiles"
    monkeypatch.setenv("UXARRAY_DATA_DIR", str(meshfiles))


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


def test_file_path_prefers_env_var(monkeypatch, tmp_path):
    data_dir = tmp_path / "meshfiles"
    grid = data_dir / "ugrid" / "quad-hexagon" / "grid.nc"
    grid.parent.mkdir(parents=True)
    grid.write_text("dummy")

    monkeypatch.setenv("UXARRAY_DATA_DIR", str(data_dir))

    path = ux.tutorial.file_path("quad-hexagon", component="grid")
    assert path == grid


def test_file_path_falls_back_to_cache(monkeypatch, tmp_path):
    cached = tmp_path / "cached-grid.nc"
    cached.write_text("dummy")

    monkeypatch.delenv("UXARRAY_DATA_DIR", raising=False)
    monkeypatch.setattr(tutorial, "_ensure_cached", lambda parts: cached)
    monkeypatch.setattr(tutorial, "_local_meshfiles_path", lambda: None)

    path = ux.tutorial.file_path("quad-hexagon", component="grid")
    assert path == cached
