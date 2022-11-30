mkdir -p data
cd data && wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
cd data && tar -xvf nyudepthv2.tar.gz && rm -f nyudepthv2.tar.gz
cd data && mv nyudepthv2 nyudepth_hdf5