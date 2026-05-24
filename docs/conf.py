project = "NovaNN CMake Documentation"
copyright = "2026, NovaNN Contributor"
author = "Juan José Medina"
release = "5.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.moderncmakedomain",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = []

intersphinx_mapping = {
    "cmake": ("https://cmake.org/cmake/help/latest", None),
}
