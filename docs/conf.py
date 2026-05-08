from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "hrtfpykit"
author = "Ariel Alvarez-Martinez"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "*.ipynb_checkpoints"]
html_static_path = []

html_theme = "alabaster"
html_title = "hrtfpykit documentation"
html_logo = "assets/images/hrtfpykit-logo.png"
html_static_path = ["assets"]
html_theme_options = {
    "show_powered_by": False,
    "show_relbars": True,
}
