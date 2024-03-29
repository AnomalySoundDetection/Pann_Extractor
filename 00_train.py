"""
 @file   00_train.py
 @brief  Script for training
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import default python-library
########################################################################
import os
import glob
import sys
import time
import gc
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
# from import
from tqdm import tqdm
# original lib
import common as com
import random
from Feature_Extractor import load_extractor
from Dataset import AudioDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
########################################################################

########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save png file path.

        return : None
        """
        self.plt.savefig(name)

########################################################################
# main 00_train.py
########################################################################
if __name__ == "__main__":
    
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    param = com.load_yaml()
        
    # make output directory
    os.makedirs(param["model_directory"], exist_ok=True)

    # initialize the visualizer
    visualizer = visualizer()

    # training device
    device = torch.device('cuda')

    # training info
    epochs = int(param["fit"]["epochs"])
    batch_size = int(param["fit"]["batch_size"])
    
    # load base_directory list
    machine_list = com.get_machine_list(param["dev_directory"])
    print("=====================================")
    print("Train Machine List: ", machine_list)
    print("=====================================")

    # loop of the base directory
    for idx, machine in enumerate(machine_list):
        print("\n===========================")
        print("[{idx}/{total}] {machine}".format(machine=machine, idx=idx+1, total=len(machine_list)))
        
        root_path = param["dev_directory"] + "/" + machine
        
        model_file_path = "{model}/model_{machine}.pt".format(model=param["model_directory"],
                                                                     machine=machine)
        history_img = "{model}/history_{machine}.png".format(model=param["model_directory"],
                                                                  machine=machine)

        if os.path.exists(model_file_path):
            com.logger.info("model exists")
            continue

        data_list = com.select_dirs(param=param, machine=machine)
        id_list = com.get_machine_id_list(target_dir=root_path, dir_type="train")

        print("Current Machine: ", machine)
        print("Machine ID List: ", id_list)

        train_list = []
        val_list = []

        for path in data_list:
            if random.random() < 0.85:
                train_list.append(path)
            else:
                val_list.append(path)
        
        for _id in id_list:
            # generate dataset
            print("\n----------------")
            print("Generating Dataset of Current ID: ", _id)

            train_dataset = AudioDataset(data=train_list, _id=_id, root=root_path, sample_rate=param["feature"]["sample_rate"])
            val_dataset = AudioDataset(data=val_list, _id=_id, root=root_path, sample_rate=param["feature"]["sample_rate"])
            
            train_dl = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            val_dl = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

            print("------ DONE -------")
            
            
            # train model
            print("\n----------------")
            print("Start Model Training...")

            train_loss_list = []
            val_loss_list = []


            # TODO: loss function and optimizer update
            #loss_function =     
            #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            extractor = load_extractor(sample_rate=param["feature"]["sample_rate"],
                                       window_size=param["feature"]["n_fft"],
                                       hop_size=param["feature"]["hop_length"],
                                       mel_bins=param["feature"]["n_mels"],
                                       fmin=param["feature"]["fmin"],
                                       fmax=param["feature"]["fmax"])
            

            extractor = extractor.to(device=device, dtype=torch.float32)
            extractor.eval()

            #flow_model = None

            for epoch in range(1, 2):
                train_loss = 0.0
                eval_loss = 0.0
                print("Epoch: {}".format(epoch))

                #TODO: Update optimizer and loss function
                #TODO: Update flow model after finishing model setup

                #flow_model.train()

                for batch in tqdm(train_dl):
                    #optimizer.zero_grad()
                    batch = batch.to(device)
                    feature = extractor(batch)

                    #loss = loss_function(batch)
                    #loss.backward()
                    #optimizer.step()
                    #train_loss += loss.item()

                    del batch
                #train_loss /= len(train_dl)
                #train_loss_list.append(train_loss)

                #flow_model.eval()
                
                with torch.no_grad():
                    for batch in tqdm(val_dl):
                        batch = batch.to(device)
                        feature = extractor(batch)
                        #loss = loss_function(batch)
                        #val_loss += loss.item()
                        del batch
                #val_loss /= len(val_dl)
                #val_loss_list.append(val_loss)
            
            visualizer.loss_plot(train_loss_list, val_loss_list)
            visualizer.save_figure(history_img)
            #torch.save(flow_model.state_dict(), model_file_path)
            com.logger.info("save_model -> {}".format(model_file_path))

            del train_dataset, val_dataset, train_dl, val_dl, extractor
            
            gc.collect()

            time.sleep(5)