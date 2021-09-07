import glob
import os

import pandas as pd
import numpy as np


class DataProcess():

    def get_email_class(self, file_path):
        """
        This method sets the labels for each email.

        Note:
            spam - 0
            ham - 1

        :return: l1 - email class
        """
        lbl = 0

        directory_name = os.path.dirname(file_path)
        dir_name_list = directory_name.split('/')
        dir_name = dir_name_list[3]

        if os.path.isdir(directory_name) and os.path.exists(directory_name):
            if dir_name == 'spam':
                lbl = 0
            elif dir_name == 'ham':
                lbl = 1
            else:
                lbl = 'CHECK'

        return lbl

    def data_creation(self):
        emails_ds = pd.DataFrame(columns=['FILE_NAME', 'LABEL'])

        file_names, subjects, body, labels = [], [], [], []

        for main_folder in glob.glob('input/enron/*'):
            main_folder = main_folder + '/*'
            for sub_folder in glob.glob(main_folder):

                sub_folder = sub_folder + '/*'
                for file in glob.glob(sub_folder):

                    label1 = self.get_email_class(file)

                    with open(file, 'r') as f:
                        fname = os.path.basename(file)
                        file_names.append(fname)
                        labels.append(label1)

        df = pd.DataFrame({'FILE_NAME': file_names,
                           'LABEL': labels
                           })

        df.index = np.arange(1, len(df) + 1)
        emails_ds = emails_ds.append(df, ignore_index=False)

        emails_ds.to_csv('input/EMAIL_DATASET.csv', index=True, index_label='SR_NO')
