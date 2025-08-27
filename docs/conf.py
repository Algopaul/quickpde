import os
import sys

sys.path.insert(0, os.path.abspath(".."))  # so `import your_pkg` works

project = "quickpde"
extensions = [
    "myst_parser",  # Markdown support
    "sphinx.ext.autodoc",  # Pulls in docstrings
    "sphinx.ext.autosummary",  # Summary pages for objects
    "sphinx.ext.napoleon",  # Google/NumPy docstrings
    "sphinx_autodoc_typehints",  # Nice type hints formatting
    "sphinx.ext.viewcode",  # Link to highlighted source
    "sphinx.ext.intersphinx",  # Cross-link to other projects (optional)
    "sphinx.ext.mathjax",  # Cross-link to other projects (optional)
    "sphinx_copybutton",
    "sphinx_mathjax_offline",
]
autosummary_generate = True  # auto-generate summary pages
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

html_theme = "furo"
add_module_names = False  # cleaner API page titles

# Optional: cross-links to the Python stdlib types in rendered docs
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# If you prefer Markdown as the primary content format:
myst_enable_extensions = ["deflist", "colon_fence"]

copybutton_prompt_text = r">>> |\.\.\. "  # Strip Python prompts
copybutton_prompt_is_regexp = True

myst_enable_extensions = [
    "amsmath",
    "dollarmath",  # support $...$ and $$...$$
]

mathjax3_config = {
    "tex": {
        "macros": {
            "R": r"\mathbb{R}",
            "C": r"\mathbb{C}",
            # with arguments
            "frobsq": [r"\left\lVert #1 \right\rVert_F^2", 1],
        }
    }
}
