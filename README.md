# neuronshapley
Neuron Shapley: Discovering the Responsible Neurons


### Prerequisites

Required python libraries:

```
  Tensorflow 1.12: https://www.tensorflow.org/
```

1- You need to download the inception_v3 checkpoint and save it as inception_v3.ckpt: https://drive.google.com/open?id=1by_aFmyImtM-pVYq_BtPTvmCF3K9iyQe

2- Imagenet Validation set should be saved in './imagenet' in the following structure: The folder should containt 1000 subfolders where images of each ImageNet class are in a separate folder (named exactly as the class names in the val_images.txt file). Images should be resized to 299 * 299.

### Getting started
How to run the code
1- The cb_run.py runs the iterations of the algorithm. Run it as follows:
```
python3 cb_run.py [class_name] [metric] [number_of_validation_images] [True if adversarial setting and False if real data]
```
Note that you can (and should) run as many cb_run.py scripts in parallell as possible (simply run the above script several times in the background).
2- The cb_aggregate.py has to run alongside cb_run.py. This script constantly aggregates the parallel results and updates the set of filters that there needs to be more adaptive sampling for. In other words, it runs the multi-armed-bandit section of the algorithm. Running only cb_run.py will not make use of the multi-armed-bandit speed-up.

Example bash script that runs NeuronShaley for the overall performance of the network:
```
python cb_aggregate.py all accuracy 25000 False
for i in $(seq 0 10)
do
    python3 cb_run.py all accuracy 25000 False
done
```


## Authors

* **Amirata Ghorbani** - [Website](http://web.stanford.edu/~amiratag)
* **James Zou** - [Website](https://www.james-zou.com/)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
