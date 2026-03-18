reload_docs:
	python -m sphinx_autobuild docs docs/_build/html

cfm_vorticity_medium:
	.venv/bin/python quickpde/driver.py --multi -cn vorticity vorticity.bump_angle=range\(0,3000\) +env=nrel

cfm_vorticity_grf:
	.venv/bin/python quickpde/driver.py --multi -cn vorticity_grf vorticity.random_seed=range\(0,3000\) outdir='data/pyqg' +env=nrel

cfm_vorticity_grf_small:
	.venv/bin/python quickpde/driver.py --multi -cn vorticity_grf vorticity.random_seed=range\(0,10\) outdir='data/pyqg_small' axis_points=128

cfm_zarrs:
	.venv/bin/python quickpde/driver.py --multi -cn vorticity_grf vorticity.random_seed=range\(0,10\) outdir='data/pyqg_zarr' axis_points=128

	.venv/bin/python quickpde/driver.py --multi -cn vorticity_grf vorticity.random_seed=range\(0,10\) outdir='data/pyqg_small' axis_points=128
