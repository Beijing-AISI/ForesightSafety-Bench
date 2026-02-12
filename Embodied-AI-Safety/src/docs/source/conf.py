# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-informatione
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))       # 包目录（projectdir/src）
project = 'Panda-Guard'
copyright = '2024, Floyed Shen, etc.'
author = 'Floyed Shen, etc.'
release = 'v0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',   # 自动从代码的 docstring 生成文档
    'sphinx.ext.napoleon',  # 支持 Google 和 NumPy 风格的 docstring
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]

language = 'en'
locale_dirs = ['../locales/']
gettext_compact = False      # optional

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
