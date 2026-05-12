from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrtfpykit import __version__

project = "hrtfpykit"
author = "Ariel Alvarez-Martinez"
copyright = "Ariel Alvarez-Martinez"
version = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "bysource"
toc_object_entries = True
toc_object_entries_show_parents = "hide"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "*.ipynb_checkpoints"]
html_static_path = ["assets"]
html_css_files = ["hrtfpykit-furo.css"]

html_theme = "furo"
html_title = "hrtfpykit documentation"
html_logo = "assets/images/hrtfpykit-logo.png"
html_favicon = "assets/images/hrtfpykit-icon.png"
html_show_sourcelink = False
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#0c7b93",
        "color-brand-content": "#0a9396",
        "color-link": "#872ee0",
        "color-link--hover": "#6d23b6",
        "color-link--visited": "#872ee0",
        "color-link-underline": "#872ee0",
        "color-link-underline--hover": "#6d23b6",
        "color-link-underline--visited": "#872ee0",
        "color-link--visited--hover": "#6d23b6",
        "color-link-underline--visited--hover": "#6d23b6",
        "color-api-name": "#005f73",
        "color-api-pre-name": "#005f73",
    },
    "dark_css_variables": {
        "color-brand-primary": "#94d2bd",
        "color-brand-content": "#94d2bd",
        "color-link": "#b27aeb",
        "color-link--hover": "#d8b4fe",
        "color-link--visited": "#b27aeb",
        "color-link-underline": "#b27aeb",
        "color-link-underline--hover": "#d8b4fe",
        "color-link-underline--visited": "#b27aeb",
        "color-link--visited--hover": "#d8b4fe",
        "color-link-underline--visited--hover": "#d8b4fe",
        "color-api-name": "#2dd4bf",
        "color-api-pre-name": "#2dd4bf",
    },
}
