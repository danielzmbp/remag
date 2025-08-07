#!/usr/bin/env python3
"""
Update Bioconda recipe after a new PyPI release
"""
import os
import sys
import hashlib
import requests
import time
import json
from pathlib import Path

def get_pypi_info(package_name, version):
    """Get package info from PyPI."""
    url = f"https://pypi.org/pypi/{package_name}/{version}/json"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def calculate_sha256(url):
    """Download file and calculate SHA256."""
    response = requests.get(url)
    response.raise_for_status()
    return hashlib.sha256(response.content).hexdigest()

def create_bioconda_recipe(package_name, version):
    """Create or update Bioconda recipe."""
    pypi_info = get_pypi_info(package_name, version)
    
    # Find the source distribution
    source_url = None
    for file_info in pypi_info['urls']:
        if file_info['packagetype'] == 'sdist':
            source_url = file_info['url']
            break
    
    if not source_url:
        raise ValueError("No source distribution found on PyPI")
    
    sha256 = calculate_sha256(source_url)
    
    recipe = f'''{{% set name = "remag" %}}
{{% set version = "{version.lstrip('v')}" %}}

package:
  name: {{{{ name|lower }}}}
  version: {{{{ version }}}}

source:
  url: https://pypi.io/packages/source/{{{{ name[0] }}}}/{{{{ name }}}}/{{{{ name }}}}-{{{{ version }}}}.tar.gz
  sha256: {sha256}

build:
  number: 0
  noarch: python
  script: {{{{ PYTHON }}}} -m pip install . -vv --no-deps --no-build-isolation
  entry_points:
    - remag = remag.cli:main

requirements:
  host:
    - python >=3.8
    - pip
    - setuptools >=61.0
    - wheel
  run:
    - python >=3.8
    - hdbscan >=0.8.28
    - matplotlib-base >=3.5.0
    - numpy >=1.21.0
    - pandas >=1.3.0
    - pysam >=0.18.0
    - rich-click >=1.5.0
    - pytorch >=1.11.0
    - loguru >=0.6.0
    - scikit-learn >=1.0.0
    - tqdm >=4.62.0
    - umap-learn >=0.5.0
    - xgboost >=1.6.0
    - joblib >=1.1.0
    - psutil >=5.8.0
    - leidenalg >=0.9.0
    - python-igraph >=0.10.0
    - biopython

test:
  imports:
    - remag
    - remag.xgbclass
  commands:
    - remag --help

about:
  home: https://github.com/danielzmbp/remag
  summary: 'Recovery of high-quality eukaryotic genomes from complex metagenomes'
  description: |
    REMAG is a specialized metagenomic binning tool designed for recovering 
    high-quality eukaryotic genomes from mixed prokaryotic-eukaryotic samples. 
    It uses contrastive learning with Siamese neural networks to generate 
    meaningful contig embeddings for clustering.
  license: MIT
  license_family: MIT
  license_file: LICENSE
  doc_url: https://github.com/danielzmbp/remag
  dev_url: https://github.com/danielzmbp/remag

extra:
  recipe-maintainers:
    - danielzmbp
'''
    
    return recipe

def create_pr_to_bioconda():
    """Create a pull request to bioconda-recipes."""
    # This would typically fork bioconda-recipes, create a branch,
    # add the recipe, and create a PR. For now, we'll just save the recipe.
    
    github_ref = os.environ.get('GITHUB_REF', '')
    
    # Handle different ref types
    if github_ref.startswith('refs/tags/'):
        version = github_ref.replace('refs/tags/', '')
    elif github_ref.startswith('refs/heads/'):
        # If triggered from a branch, try to get the latest tag
        print(f"Warning: Running from branch {github_ref}, looking for latest release...")
        # For manual runs, we should specify version explicitly
        version = os.environ.get('RELEASE_VERSION', '')
        if not version:
            print("No version specified. Use RELEASE_VERSION environment variable.")
            sys.exit(1)
    else:
        version = github_ref
    
    # Clean up version (remove 'v' prefix if present)
    if version.startswith('v'):
        version = version[1:]
    
    if not version:
        print("No version tag found")
        sys.exit(1)
    
    print(f"Creating recipe for version: {version}")
    
    # Wait a bit for PyPI to update if this is a fresh release
    time.sleep(10)
    
    recipe = create_bioconda_recipe('remag', version)
    
    # Save recipe for manual submission or automated PR
    recipe_path = Path('.github/bioconda-recipe/meta.yaml')
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path.write_text(recipe)
    
    print(f"Bioconda recipe created for version {version}")
    print("Recipe saved to .github/bioconda-recipe/meta.yaml")
    print("\nTo submit to Bioconda:")
    print("1. Fork https://github.com/bioconda/bioconda-recipes")
    print("2. Create a new branch")
    print("3. Add this recipe to recipes/remag/meta.yaml")
    print("4. Create a pull request")

if __name__ == "__main__":
    create_pr_to_bioconda()