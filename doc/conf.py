from __future__ import absolute_import, division, print_function

import os
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('.'))

base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
about = {}
with open(os.path.join(base_dir, "src", "pybnb", "__about__.py")) as f:
    exec(f.read(), about)
project = about["__title__"]
copyright = about["__author__"]
version = release = about["__version__"]

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.viewcode',
              'sphinx.ext.intersphinx',
              'sphinxcontrib.napoleon',
              'sphinxcontrib.spelling']

nitpicky = False
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build']
pygments_style = 'sphinx'
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
htmlhelp_basename = 'pybnbdoc'
autodoc_member_order = 'bysource'
spelling_word_list_filename='spelling_wordlist.txt'
