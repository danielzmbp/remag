# Release Process for REMAG

This document describes the automated release process for REMAG.

## Prerequisites

1. **GitHub Repository Secrets**
   - `PYPI_API_TOKEN`: PyPI API token for package upload
   - `BIOCONDA_TOKEN`: GitHub token for Bioconda PR (optional)
   - `ZENODO_TOKEN`: Zenodo API token for DOI generation

2. **PyPI Setup**
   - Create an account at https://pypi.org
   - Generate an API token: Account Settings → API tokens
   - Add token as GitHub secret named `PYPI_API_TOKEN`

3. **Zenodo Setup**
   - Create account at https://zenodo.org
   - Generate token: Applications → Personal access tokens
   - Add token as GitHub secret named `ZENODO_TOKEN`
   - First release will create a concept DOI for all versions

4. **Bioconda Setup**
   - Fork https://github.com/bioconda/bioconda-recipes
   - The workflow will generate a recipe, but manual PR is needed

## Release Steps

1. **Update Version**
   ```bash
   # Update version in pyproject.toml
   # Commit changes
   git add pyproject.toml
   git commit -m "Bump version to X.Y.Z"
   ```

2. **Create Release Tag**
   ```bash
   # Create annotated tag
   git tag -a v0.2.0 -m "Release version 0.2.0"
   
   # Push tag to trigger release workflow
   git push origin v0.2.0
   ```

3. **Automated Process**
   The GitHub Actions workflow will:
   - Build source and wheel distributions
   - Upload to PyPI
   - Create GitHub release with changelog
   - Upload to Zenodo and get DOI
   - Generate Bioconda recipe

4. **Manual Bioconda Submission**
   - Check `.github/bioconda-recipe/meta.yaml`
   - Submit PR to bioconda-recipes repository
   - Recipe will be automatically tested

## Version Numbering

Follow semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Incompatible API changes
- MINOR: New functionality, backwards compatible
- PATCH: Bug fixes, backwards compatible

## Release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in pyproject.toml
- [ ] Create release tag
- [ ] Verify PyPI upload
- [ ] Verify Zenodo DOI
- [ ] Submit Bioconda PR
- [ ] Update citation info with new DOI

## Troubleshooting

### PyPI Upload Fails
- Check `PYPI_API_TOKEN` is valid
- Ensure version doesn't already exist
- Verify package metadata in pyproject.toml

### Zenodo Upload Fails
- Check `ZENODO_TOKEN` permissions
- First upload creates deposition ID
- Subsequent uploads update existing record

### Bioconda Recipe Issues
- Test locally: `conda build recipes/remag`
- Check dependencies match conda-forge names
- Ensure all dependencies are available

## Manual Release (Emergency)

If automation fails:

```bash
# Build distributions
python -m build

# Upload to PyPI
python -m twine upload dist/*

# Create GitHub release manually
# Upload to Zenodo manually
```