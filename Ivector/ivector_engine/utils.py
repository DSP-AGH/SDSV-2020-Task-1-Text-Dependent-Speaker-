import pandas as pd
import numpy as np
from datetime import datetime


def text_file_to_np_array(directory):
    """
    parses content of text file into np array

    :param directory: directory of text file dedicated for text files in SdSV docs directory

    :return: numpy array containing data in text file (to be processed in train, enrollment, trail procedures)
    """
    df = pd.read_csv(directory, sep=' ', index_col=None, delimiter=None)
    array_ = df.to_numpy()

    return array_

def generate_system_output_file(scores_arr, file_name = 'output_file.txt'):
    """
    generates text file with LLR scores according to docs/trials.txt

    :param scores_arr: 1D numpy array containing LLR scores
    :param file_name: name of the file in directory ./Ivector/output_files
    :return:
    """

    now = datetime.now()
    date_str = now.strftime("%d_%m_%Y_%H_%M_%S")
    date_str+='_'
    base_directory = './output_files/'
    output_file_name = base_directory+date_str+'_'+file_name
    np.savetxt(output_file_name, scores_arr, delimiter='\n')




