# Neural Network model related files

This folder contains the files which are directly related to the training of the neural network.

- train/
  - [tiles_3031.geojson](train/tiles_3031.geojson)  (Bounding boxes of the raster training tiles in EPSG:3031 projection)
  - [tiles_4326.geojson](train/tiles_4326.geojson)  (Reprojected bounds in EPSG:4326, rendered as a slippy map in Github)
  - \*_data.npy  (\*hidden in git, preprocessed raster tiles from data_prep.ipynb)

- weights/
  - [srgan_generator_model_architecture.json](weights/srgan_generator_model_architecture.json)  (Keras model architecture of Generator Network in JSON)
  - srgan_generator_model_weights.hdf5  (\*hidden in git, neural network weights)
  - srgan_generator_model.hdf5  (\*hidden in git, contains both neural network model architecture and weights)
