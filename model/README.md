# Neural Network model related files

This folder contains the files which are directly related to the training of the neural network.

- train/
  - [tiles_3031.geojson](train/tiles_3031.geojson)  (Bounding boxes of the raster training tiles in EPSG:3031 projection)
  - [tiles_4326.geojson](train/tiles_4326.geojson)  (Reprojected bounds in EPSG:4326, rendered as a slippy map in Github)
  - \*_data.npy  (\*hidden in git, preprocessed raster tiles from data_prep.ipynb)

- weights/
  - [srgan_generator_model_architecture.onnx.txt](weights/srgan_generator_model_architecture.onnx.txt)  (Chainer model architecture of Generator Network in ONNX text format)
  - srgan_generator_model_architecture.onnx  (\*hidden in git, Chainer model architecture of Generator Network in ONNX binary format)
  - srgan_generator_model_weights.npz  (\*hidden in git but available at https://www.comet.ml/weiji14/deepbedmap under experiment assets, trained neural network weights)
