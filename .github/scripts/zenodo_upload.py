#!/usr/bin/env python3
"""
Upload release to Zenodo and get DOI
"""
import os
import sys
import json
import requests
from pathlib import Path

ZENODO_URL = "https://zenodo.org/api"
SANDBOX_URL = "https://sandbox.zenodo.org/api"  # For testing

def get_zenodo_deposition(token, use_sandbox=False):
    """Create or get existing deposition."""
    base_url = SANDBOX_URL if use_sandbox else ZENODO_URL
    headers = {"Authorization": f"Bearer {token}"}
    
    # Check if we have a previous deposition ID stored
    deposition_file = Path(".zenodo_deposition_id")
    if deposition_file.exists():
        deposition_id = deposition_file.read_text().strip()
        # Get the deposition
        r = requests.get(f"{base_url}/deposit/depositions/{deposition_id}", headers=headers)
        if r.status_code == 200:
            deposition = r.json()
            # If it's published, create a new version
            if deposition.get('state') == 'done':
                print(f"Creating new version of deposition {deposition_id}")
                r = requests.post(
                    f"{base_url}/deposit/depositions/{deposition_id}/actions/newversion",
                    headers=headers
                )
                if r.status_code != 201:
                    print(f"Failed to create new version: {r.status_code} - {r.text}")
                    # Fallback: create completely new deposition
                    print("Falling back to creating new deposition...")
                    r = requests.post(f"{base_url}/deposit/depositions", headers=headers, json={})
                    r.raise_for_status()
                    deposition = r.json()
                    deposition_file.write_text(str(deposition['id']))
                    return deposition
                r.raise_for_status()
                # Get the new draft
                new_version_url = r.json()['links']['latest_draft']
                r = requests.get(new_version_url, headers=headers)
                r.raise_for_status()
                new_deposition = r.json()
                # Update stored ID to the new draft ID
                deposition_file.write_text(str(new_deposition['id']))
                return new_deposition
            return deposition
    
    # Create new deposition
    r = requests.post(f"{base_url}/deposit/depositions", headers=headers, json={})
    r.raise_for_status()
    deposition = r.json()
    
    # Save deposition ID for future releases
    deposition_file.write_text(str(deposition['id']))
    
    return deposition

def update_metadata(deposition, version):
    """Update deposition metadata."""
    metadata = {
        "title": f"REMAG: Recovering high-quality Eukaryotic genomes from complex metagenomes v{version}",
        "upload_type": "software",
        "description": """<p>REMAG is a specialized metagenomic binning tool designed for recovering high-quality eukaryotic genomes from mixed prokaryotic-eukaryotic samples.</p>

<p><strong>Key Features:</strong></p>
<ul>
<li>Bacterial filtering using 4CAC XGBoost classifier</li>
<li>Contrastive learning with Siamese neural networks</li>
<li>HDBSCAN clustering for genome binning</li>
<li>Quality assessment using eukaryotic core genes</li>
<li>Iterative refinement for contamination removal</li>
</ul>

<p><strong>What's new in this version:</strong></p>
<p>See the release notes on GitHub for detailed changes.</p>""",
        "creators": [
            {
                "name": "Gómez-Pérez, Daniel",
                "affiliation": "Earlham Institute",
                "orcid": "0000-0002-9938-9444"
            }
        ],
        "keywords": [
            "metagenomics",
            "binning",
            "eukaryotes",
            "machine learning",
            "contrastive learning",
            "bioinformatics"
        ],
        "license": "MIT",
        "related_identifiers": [
            {
                "relation": "isSupplementTo",
                "identifier": "https://github.com/danielzmbp/remag",
                "resource_type": "software"
            }
        ],
        "access_right": "open",
        "version": version.lstrip('v')
    }
    
    return metadata

def upload_files(deposition, token, use_sandbox=False):
    """Upload distribution files to Zenodo."""
    base_url = SANDBOX_URL if use_sandbox else ZENODO_URL
    headers = {"Authorization": f"Bearer {token}"}
    
    # First, delete any existing files if this is a new version
    if 'files' in deposition and deposition['files']:
        for file_info in deposition['files']:
            print(f"Removing old file: {file_info['filename']}")
            r = requests.delete(
                f"{base_url}/deposit/depositions/{deposition['id']}/files/{file_info['id']}",
                headers=headers
            )
            # Don't fail if file deletion fails (might already be deleted)
            if r.status_code not in [204, 404]:
                print(f"Warning: Could not delete file {file_info['filename']}: {r.status_code}")
    
    # Upload all files in dist/
    dist_path = Path("dist")
    if not dist_path.exists():
        print("No dist/ directory found")
        return
    
    for file_path in dist_path.glob("*"):
        if file_path.is_file():
            print(f"Uploading {file_path.name}...")
            
            with open(file_path, 'rb') as f:
                # Try the bucket upload method first (for new versions)
                if 'links' in deposition and 'bucket' in deposition['links']:
                    bucket_url = deposition['links']['bucket']
                    print(f"Trying bucket upload to: {bucket_url}/{file_path.name}")
                    r = requests.put(
                        f"{bucket_url}/{file_path.name}",
                        headers=headers,
                        data=f
                    )
                    if r.status_code in [200, 201]:
                        print(f"Successfully uploaded {file_path.name} via bucket")
                        continue
                    else:
                        print(f"Bucket upload failed: {r.status_code} - {r.text}")
                
                # Fallback to traditional file upload
                f.seek(0)  # Reset file pointer
                files = {'file': (file_path.name, f)}
                r = requests.post(
                    f"{base_url}/deposit/depositions/{deposition['id']}/files",
                    headers=headers,
                    files=files
                )
                if r.status_code != 201:
                    print(f"Upload failed with status {r.status_code}")
                    print(f"Response: {r.text}")
                    print(f"URL: {base_url}/deposit/depositions/{deposition['id']}/files")
                r.raise_for_status()

def publish_deposition(deposition, token, use_sandbox=False):
    """Publish the deposition to get DOI."""
    base_url = SANDBOX_URL if use_sandbox else ZENODO_URL
    headers = {"Authorization": f"Bearer {token}"}
    
    r = requests.post(
        f"{base_url}/deposit/depositions/{deposition['id']}/actions/publish",
        headers=headers
    )
    if r.status_code != 202:
        print(f"Publish failed with status {r.status_code}")
        print(f"Response: {r.text}")
    r.raise_for_status()
    
    return r.json()

def main():
    """Main function."""
    token = os.environ.get('ZENODO_TOKEN')
    if not token:
        print("ZENODO_TOKEN not set")
        sys.exit(1)
    
    version = os.environ.get('GITHUB_REF', '').replace('refs/tags/', '')
    if not version:
        print("No version tag found")
        sys.exit(1)
    
    use_sandbox = os.environ.get('ZENODO_SANDBOX', 'false').lower() == 'true'
    
    try:
        # Get or create deposition
        print("Getting Zenodo deposition...")
        deposition = get_zenodo_deposition(token, use_sandbox)
        print(f"Using deposition ID: {deposition['id']}")
        print(f"Deposition state: {deposition.get('state', 'unknown')}")
        print(f"Available links: {list(deposition.get('links', {}).keys())}")
        if 'bucket' in deposition.get('links', {}):
            print(f"Bucket URL: {deposition['links']['bucket']}")
        
        # Update metadata
        print("Updating metadata...")
        metadata = update_metadata(deposition, version)
        base_url = SANDBOX_URL if use_sandbox else ZENODO_URL
        headers = {"Authorization": f"Bearer {token}"}
        
        r = requests.put(
            f"{base_url}/deposit/depositions/{deposition['id']}",
            headers=headers,
            json={"metadata": metadata}
        )
        r.raise_for_status()
        
        # Upload files
        print("Uploading files...")
        upload_files(deposition, token, use_sandbox)
        
        # Publish to get DOI
        print("Publishing deposition...")
        published = publish_deposition(deposition, token, use_sandbox)
        
        doi = published.get('doi', 'Unknown')
        doi_url = published.get('doi_url', '')
        
        print(f"\nSuccess! Your release has been published to Zenodo")
        print(f"DOI: {doi}")
        print(f"URL: {doi_url}")
        
        # Save DOI for badge in README
        Path(".zenodo_doi").write_text(doi)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()