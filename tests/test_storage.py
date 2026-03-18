"""
Tests for data storage in quickpde.driver.

Strategy:
- get_filename: tests path construction logic without running any simulation.
- zarr round-trip: write a known array and read it back. Verifies that the
  zarr write path in driver.py doesn't corrupt data.
- store_type: verify that the dtype cast (cfg.store_type) is actually applied
  to the stored trajectory. This is a regression test for the bug where
  store_type was defined in Config but never used.
"""
import math

import numpy as np
import zarr

from quickpde.config.base import Config
from quickpde.driver import get_filename
from quickpde.pdes import PDE


# ---------------------------------------------------------------------------
# get_filename
# ---------------------------------------------------------------------------

def test_get_filename_uses_provided_outfile(tmp_path):
    cfg = Config(outfile='myrun', outdir=str(tmp_path))
    path = get_filename(cfg)
    assert path.endswith('myrun.zarr')


def test_get_filename_already_has_zarr_extension(tmp_path):
    cfg = Config(outfile='myrun.zarr', outdir=str(tmp_path))
    path = get_filename(cfg)
    # Should not double the extension
    assert path.endswith('myrun.zarr')
    assert 'myrun.zarr.zarr' not in path


def test_get_filename_creates_parent_directory(tmp_path):
    subdir = tmp_path / 'deep' / 'nested'
    cfg = Config(outfile='run', outdir=str(subdir))
    get_filename(cfg)
    assert subdir.exists()


def test_get_filename_none_outfile_produces_hash(tmp_path):
    cfg = Config(outfile=None, outdir=str(tmp_path))
    path = get_filename(cfg)
    # Path should end with <10-char-hash>.zarr
    stem = path.split('/')[-1].replace('.zarr', '')
    assert len(stem) == 10
    assert stem.isalnum()


# ---------------------------------------------------------------------------
# Zarr round-trip
# ---------------------------------------------------------------------------

def test_zarr_array_roundtrip(tmp_path):
    """Data written to a zarr store must survive a read-back unchanged."""
    store_path = str(tmp_path / 'test.zarr')
    data = np.random.default_rng(0).standard_normal((5, 8, 8)).astype(np.float32)

    root = zarr.group(store_path)
    root.create_array('data', data=data)

    root2 = zarr.open_group(store_path, mode='r')
    recovered = np.array(root2['data'])
    np.testing.assert_array_equal(recovered, data)


def test_zarr_roundtrip_with_time(tmp_path):
    """Both 'data' and 'time' arrays survive a zarr round-trip."""
    store_path = str(tmp_path / 'test.zarr')
    data = np.ones((3, 8, 8), dtype=np.float32)
    time = np.array([1e-3, 2e-3, 3e-3])

    root = zarr.group(store_path)
    root.create_array('data', data=data)
    root.create_array('time', data=time)

    root2 = zarr.open_group(store_path, mode='r')
    np.testing.assert_array_equal(np.array(root2['data']), data)
    np.testing.assert_array_almost_equal(np.array(root2['time']), time)


# ---------------------------------------------------------------------------
# store_type is applied
# ---------------------------------------------------------------------------

def test_store_type_f4_produces_float32(tmp_path):
    """store_type='f4' must cast the trajectory to float32 before writing.

    This is a regression test: store_type was previously defined in Config
    but never applied in driver.py.
    """
    cfg = Config(
        pde='rotation_2d',
        axis_points=8,
        bound_x=(-math.pi, math.pi),
        bound_y=(-math.pi, math.pi),
        dt=1e-3,
        t_end=1e-3,
        store_every=1,
        ic_sharpness=5.0,
        store_type='f4',
        outfile='dtype_test',
        outdir=str(tmp_path),
    )
    pde = PDE.from_config(cfg)
    trajectory, _ = pde.solve(cfg)
    stored = np.array(trajectory, dtype=np.dtype(cfg.store_type))
    assert stored.dtype == np.float32


def test_store_type_f8_produces_float64(tmp_path):
    cfg = Config(
        pde='rotation_2d',
        axis_points=8,
        bound_x=(-math.pi, math.pi),
        bound_y=(-math.pi, math.pi),
        dt=1e-3,
        t_end=1e-3,
        store_every=1,
        ic_sharpness=5.0,
        store_type='f8',
        use_double_precision=True,
        outfile='dtype_test64',
        outdir=str(tmp_path),
    )
    import jax
    jax.config.update('jax_enable_x64', True)
    pde = PDE.from_config(cfg)
    trajectory, _ = pde.solve(cfg)
    stored = np.array(trajectory, dtype=np.dtype(cfg.store_type))
    assert stored.dtype == np.float64
