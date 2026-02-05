rm -f a3.zip 
pushd submission; zip -r ../a3.zip . --exclude "*__pycache__*"; popd