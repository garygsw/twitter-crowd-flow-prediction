Note: *Only tested on `Windows Server 2012 R2` with `Theano`. If you run experiments on `Linux/Mac OS`, or with `tensorflow`, you may need to change the hyper-parameters (e.g. learning_rate).*

1. Install [**DeepST**](https://github.com/lucktroy/DeepST)

2. Download [**BikeNYC**](https://github.com/lucktroy/DeepST/tree/master/data/BikeNYC) data

3. Reproduce the result of ST-ResNet 

    ```
    THEANO_FLAGS="device=gpu,floatX=float32" python exptBikeNYC.py
    ```