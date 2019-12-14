# DEEPATTERN

The repo contains the source code for our DAC'19 paper:

Haoyu Yang, Piyush Pathak, Frank Gennari, Ya-Chieh Lai and Bei Yu, “DeePattern: Layout Pattern Generation with Transforming Convolutional Auto-Encoder”, ACM/IEEE Design Automation Conference (DAC), Las Vegas, NV, June 2–6, 2019. 

Due to IP issue, we do not have the original layouts/designs included here.
A test layout generator is instead provided in src/fakegen.py, which will generate the training and testing data.

## How To
### Train

use ``` make df ``` to create train.msgpack and test.msgpack

- you can also create your own msgpack(s) for practical usage by simply replacing the files located in ```./data```

- currently we only support catalogs that ```cX,cY <= 16```

use ``` make euv ``` to train the TCAE.

### Test

``` make test ``` will do the job.
