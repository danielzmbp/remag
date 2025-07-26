#!/usr/bin/env python3
"""
Generate Bioconda recipe and create PR instructions
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    # Run the update script to generate recipe
    subprocess.run([sys.executable, ".github/scripts/update_bioconda.py"])
    
    # Get version from GITHUB_REF
    version = os.environ.get('GITHUB_REF', '').replace('refs/tags/', '')
    if not version:
        print("No version tag found")
        sys.exit(1)
    
    # Read the generated recipe
    recipe_path = Path(".github/bioconda-recipe/meta.yaml")
    if not recipe_path.exists():
        print("Recipe not generated")
        sys.exit(1)
    
    recipe_content = recipe_path.read_text()
    
    # Generate PR instructions
    print(f"""
═══════════════════════════════════════════════════════════════════════════════
BIOCONDA RECIPE GENERATED SUCCESSFULLY!
═══════════════════════════════════════════════════════════════════════════════

Version: {version}
Recipe saved to: .github/bioconda-recipe/meta.yaml

TO SUBMIT TO BIOCONDA:
═══════════════════════════════════════════════════════════════════════════════

1. Go to: https://github.com/danielzmbp/bioconda-recipes
2. Make sure you're on the main/master branch
3. Click on 'recipes' folder → 'remag' folder
4. Click on 'meta.yaml' → Edit (pencil icon)
5. Replace the entire content with the recipe below
6. Commit with message: "Update remag to {version}"
7. Create PR with title: "Update remag to {version}"

OR use GitHub CLI:
═══════════════════════════════════════════════════════════════════════════════

gh repo fork bioconda/bioconda-recipes --clone
cd bioconda-recipes
git checkout -b update-remag-{version}
mkdir -p recipes/remag
cat > recipes/remag/meta.yaml << 'EOF'
{recipe_content}EOF
git add recipes/remag/meta.yaml
git commit -m "Update remag to {version}"
git push origin update-remag-{version}
gh pr create --title "Update remag to {version}" --body "Update REMAG to {version}"

═══════════════════════════════════════════════════════════════════════════════
""")

if __name__ == "__main__":
    main()