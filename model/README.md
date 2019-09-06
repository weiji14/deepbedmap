# Neural Network model related files

This folder contains the files which are directly related to the training of the neural network.

- train/
  - [tiles_3031.geojson](train/tiles_3031.geojson)  (Bounding boxes of the raster training tiles in EPSG:3031 projection)
  - [tiles_4326.geojson](train/tiles_4326.geojson)  (Reprojected bounds in EPSG:4326, rendered as a slippy map in Github)
  - \*_data.npy  (\*hidden in git, but available at https://quiltdata.com/package/weiji14/deepbedmap, preprocessed raster tiles from data_prep.ipynb)

- weights/
  - srgan_generator_model_architecture.dot (\* hidden in git but available at https://www.comet.ml/weiji14/deepbedmap under Graph definition, Chainer model architecture of Generator Network in Graphviz DOT format)
  - srgan_generator_model_weights.npz  (\*hidden in git but available at https://www.comet.ml/weiji14/deepbedmap under experiment assets, trained neural network weights)
