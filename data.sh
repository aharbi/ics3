# Download and extract the OpenEarthMap data
mkdir data

curl -L -O https://zenodo.org/records/7223446/files/OpenEarthMap.zip\?download\=1
unzip OpenEarthMap.zip -d data

rm OpenEarthMap.zip
