# Transfer Learning Script guide

### Arguments
--config_file: Path to configuration file (CSV) (arguments not sepcified will be set to default)
--model_name: Name of the pre-trained model to use (mobilenet, efficientnet, etc.)
--num_models: Number of models to train
--num_epochs: Number of epochs to train each model
--dataset_dir: Directory where the dataset that will be used for training is located
--batch_size: Batch size for training
--learning_rate: Learning rate for the optimizer
--optimizer: Optimizer to use
--save_dir: Directory to save the models

### Types, default arguments, and choices
--config_file: type=str, default = None
--model_name: type=str, default = mobilenet, choices = [mobilenet, efficientnet, resnet50, inceptionv3]
--num_models: type=int, default = 1
--num_epochs: type=int, default = 2
--dataset_dir: type=str, default = None (required)
--batch_size: type=int, default = 32
--learning_rate: type=float, default = 0.001
--optimizer: type=str, default = adam  choices=["adam", "sgd"]
--save_dir: type=str, default = "saved_models"

### Common Error
TensorFlow has a bug with no apparent solution at the moment. TensorFlow caches the model used in training in a temporary folder, if the model isn't used for a while, i.e. the weights aren't updated, it depricates and you will get a ValueError notifying you that the model does not exist. This error was first spotted when performing transfer learning on mobilenet and then performing it again one week later.

Workaround#1 = The easiest but slightly annoying fix is to delete the folder that the ValueError mentions. If you do this, a new folder will be created when you rin the script again and evrything should run smootly.

Workaround#2 = The second fix that was identified online was downloading the models locally and changing the URLs to point to the directory where they are located. This has not yet been tried but seems to be a "permanent" fix, but this would involve downloading all the models locally which takes up space and changing the script code.