# DEEPATTERN+

The repo contains the source code of DeePattern+, a framework developed from:

Haoyu Yang, Piyush Pathak, Frank Gennari, Ya-Chieh Lai and Bei Yu, “DeePattern: Layout Pattern Generation with Transforming Convolutional Auto-Encoder”, ACM/IEEE Design Automation Conference (DAC), Las Vegas, NV, June 2–6, 2019. 



## How To
### DeePattern (TCAE) Train

use ``` make train% ``` to train TCAE model for a the given data set, e.g. ```make traintc1```

- currently we only support catalogs that ```cX,cY <= 24```

- trained models will be stored at ```./models/tc1/```

- some reconstruction samples during training will be available at ```./models/tc1/sample```



### DeePattern (TCAE) Test

use ``` make test% ``` to do inference and generated 1M patterns with noise perturbation, e.g. ```make testtc1```

- a msgpack containing 1M generated patterns will be put at ```./models/tc1/test/noise_data.msgpack```


### DeePattern+ (TCAE+GAN) Train

```make gandata%```

- generate the training set for GAN based on ``` make test% ``` results.

```make gantrain%```

- train gan

### DeePattern+ (TCAE+GAN) Train

```make gantest%```

- connect the generator with reconstion unit do the inference.

