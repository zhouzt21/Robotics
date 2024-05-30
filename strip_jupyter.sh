# Description: Strip output from Jupyter notebooks before committing to git
# Usage: Run this script from the root of your git repository
# If you don't want to stripe output of all notebooks, remove the last line and add it to .gitattributes at each folder manually

git config filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'
echo "*.ipynb filter=strip-notebook-output" >> .git/info/attributes
