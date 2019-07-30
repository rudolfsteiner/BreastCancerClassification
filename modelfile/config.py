import os

class Config():

    # the root path for all the data
    ROOT_PATH = "/mnt/windows/projects/IDC_regular_ps50_idx5/idc"

    # the training, validation, and testing directories, predefined, not randomly selected
    TRAIN_PATH = os.path.sep.join([ROOT_PATH, "train"])
    VAL_PATH = os.path.sep.join([ROOT_PATH, "val"])
    TEST_PATH = os.path.sep.join([ROOT_PATH, "test"])
    
    # num of epochs for training
    epochs = 10  
    lr = 1e-2 #learning reate
    batch_size = 128 
    plot_name = "plot.png" # filename to save the training history

