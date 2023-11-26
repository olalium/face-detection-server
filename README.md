# Face detection Rust server 👤👤👤
Implementation of a fast face detection server using [actix](https://actix.rs/) and [ort](https://crates.io/crates/ort). The model in use is the [Ultra-Light-Fast-Generic-Face-Detector](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB).

## Architecture
![image](https://github.com/olalium/face-detection-server/assets/18397540/c30ba4f5-a243-4a58-945b-0710aea2ef67)

## Requirements

### Model
You need to download the Ultra face detection model (version-RFB-640.onnx) and pass it through the environmental variable `ULTRA_MODEL_PATH`.

### Environmental variables
You need to pass the following environmental variables in an .env file.

| Environmental variable | description                                                              |
|------------------------|--------------------------------------------------------------------------|
| ULTRA_MODEL_PATH       | path to the onnx model, ie. model/version-RFB-640.onnx                   |
| ULTRA_THREADS          | number of threads to use when running the ultra face detection neural net|
