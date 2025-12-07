reload_docs:
	python -m sphinx_autobuild docs docs/_build/html

cfm_vorticity_medium:
	.venv/bin/python quickpde/driver.py --multi -cn vorticity vorticity.bump_angle=range\(0,1000\) +env=nrel
