from Utils.utils import *
from datetime import datetime
from perceiver_pytorch import Perceiver
import sys

# tensorboard --logdir=E:\DeepLearning\Logs\LogOneWheele3 --host localhost --port 8088

if __name__ == "__main__":

    print(sys.argv[1])

    datasetBasePath=sys.argv[1]
    cudaNum=int(sys.argv[2])

    lossSpeed= 0.005
    logPath=sys.argv[4]
    #"C:/Users/josef/Desktop/DeepLearning/Logs/LogOneWheele256/"

    hidden_size = 300
    hidden_layer_depth = 1

    learning_rate = 0.0001

    sequence_length = 1024
    hidden_size_FC = 1
    number_of_features = 18
    batch_size = 64
    n_epochs = 500

    print("Program start time =", datetime.now().strftime("%H:%M:%S"))
    train_dataset, mul, muw, stdl, stdw = open_data_pickle_one_wheele2('trainUpdated', speed_vect=False, base=datasetBasePath)
    test_dataset, mul, muw, stdl, stdw = open_data_pickle_one_wheele2('testJa', speed_vect=False, base=datasetBasePath,
                                               mul=mul, muw=muw, stdl=stdl, stdw=stdw)

    # test_dataset = downscale(test_dataset, 256)
    # train_dataset = downscale(train_dataset, 256)

    test_dataset_sub=TensorDataset(test_dataset.tensors[0][:, :, :], test_dataset.tensors[1], test_dataset.tensors[2], test_dataset.tensors[3])
    train_dataset_sub=TensorDataset(train_dataset.tensors[0][:, :, :], train_dataset.tensors[1], train_dataset.tensors[2], train_dataset.tensors[3])

    model = Perceiver(
        input_channels=18,  # number of channels for each token of the input
        input_axis=1,  # number of axis for input data (2 for images, 3 for video)
        num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
        max_freq=10.,  # maximum frequency, hyperparameter depending on how fine the data is
        depth=6,  # depth of net
        num_latents=256,
        # number of latents, or induced set points, or centroids. different papers giving it different names
        cross_dim=512,  # cross attention dimension
        latent_dim=128,  # latent dimension
        cross_heads=1,  # number of heads for cross attention. paper said 1
        latent_heads=8,  # number of heads for latent self attention, 8
        cross_dim_head=64,
        latent_dim_head=64,
        num_classes=1,  # output number of classes
        attn_dropout=0.2,
        ff_dropout=0.3,
        weight_tie_layers=False  # whether to weight tie layers (optional, as indicated in the diagram)
    )

    print("Learning started =", datetime.now().strftime("%H:%M:%S"))

    # inp=torch.rand(1, 254, 18)
    # x=model(inp)
    model.fit(n_epochs, train_dataset_sub, test_dataset_sub, shuffleTrain=True, train_verif=0.1,
              mess=sys.argv[3], logerPath=logPath, cudaNum=cudaNum, batch_size=batch_size)

    print("Learning finished =", datetime.now().strftime("%H:%M:%S"))
    stop = 0
    # z_run = vrae.transform(test_dataset)
    #os.system("shutdown /s /t 1")
