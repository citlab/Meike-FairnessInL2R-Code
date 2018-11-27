import numpy as np
import pandas as pd
from learning import Globals

########top_male_bottom_female#######
#directory = '../../sample/synthetic/top_male_bottom_female/GAMMA=0/'
#####GAMMA=0######
#file_directory = '../../sample/synthetic/top_male_bottom_female/GAMMA=0/sample_test_data_scoreAndGender_separated.txt'

#####GAMMA=75######
#file_directory = '../../sample/synthetic/top_male_bottom_female/GAMMA=75/sample_test_data_scoreAndGender_separated.txt'

#####GAMMA=150######
#file_directory = '../../sample/synthetic/top_male_bottom_female/GAMMA=150/sample_test_data_scoreAndGender_separated.txt'

########synthetic_multinomial#######

#####GAMMA=0######
directory = '../../sample/synthetic_multinomial/GAMMA=0/'
file_directory = 'testdaten.csv'

#####GAMMA=6.5######
# directory = '../../sample/synthetic_multinomial/GAMMA=6.5/'
# file_directory = '../../sample/synthetic_multinomial/GAMMA=6.5/sample_test_data_scoreAndgroups_separated.txt'

#####GAMMA=8######
# directory = '../../sample/synthetic_multinomial/GAMMA=8/'
# file_directory = '../../sample/synthetic_multinomial/GAMMA=8/sample_test_data_scoreAndgroups_separated.txt'

#####GAMMA=75######
# directory = '../../sample/synthetic_multinomial/GAMMA=75/'
# file_directory = '../../sample/synthetic_multinomial/GAMMA=75/sample_test_data_scoreAndgroups_separated.txt'

########LawStud#######
#####GAMMA=0######
# directory = '../../sample/LWStud/GAMMA=0/'
# file_directory = '../../sample/LWStud/GAMMA=0/LSAT_AllInOne_test.csv'

#####GAMMA=75######
# directory = '../../sample/LWStud/GAMMA=75/'
# file_directory = '../../sample/LWStud/GAMMA=75/LSAT_AllInOne_test.csv'

#####GAMMA=150######
# directory = '../../sample/LWStud/GAMMA=150/'
# file_directory = '../../sample/LWStud/GAMMA=150/LSAT_AllInOne_test.csv'

#####GAMMA=300######
# directory = '../../sample/LWStud/GAMMA=300/'
# file_directory = '../../sample/LWStud/GAMMA=300/LSAT_AllInOne_test.csv'

#####GAMMA=600######
# directory = '../../sample/LWStud/GAMMA=600/'
# file_directory = '../../sample/LWStud/GAMMA=600/LSAT_AllInOne_test.csv'

with open(directory+'model.txt','r',encoding='utf-8-sig') as file:
    for line in file:
        omega = np.asarray(line.split(" "), dtype='float64')
        print(omega)
        df = pd.read_csv(file_directory, decimal=',')
        df = df.apply(pd.to_numeric, errors='ignore')
        print(df)
        list_id = np.asarray(df.iloc[:, 0])
        X = np.asarray(df.iloc[:, 1:(df.shape[1] - 1)])
        y = np.reshape(np.asarray(df.iloc[:, df.shape[1] - 1]), (X.shape[0], 1))
        # with open(file_directory, 'r') as test_file:
        #     list_id = []
        #     y = []
        #     X = []
        #     for row in test_file:
        #         currentline = row.split(",")
        #         id = currentline[0]
        #         X1 = currentline[1:len(currentline) - 1]
        #         y1 = currentline[len(currentline) - 1]
        #
        #         list_id.append(id)
        #         X.append(X1)
        #         y.append(y1)
        # list_id = np.asarray(list_id).astype('int64')
        # X = np.asarray(X).astype('float64')
        # y = np.reshape(np.asarray(y).astype('float64'), (len(y), 1))
        z = np.dot(X,omega)
        doc_ids = np.arange(1,np.size(z)+1)

        # also write y for later evaluation
        y = np.transpose(np.array([list_id, doc_ids, y.reshape(-1), X[:,Globals.PROT_COL]]))
        np.savetxt(directory+'trainingScores_ORIG.pred', y, delimiter=',', fmt=['%d','%d','%1.2f','%d'])

        # unsorted prediction
        z_unsorted = np.transpose(np.array([list_id, doc_ids, z, X[:,Globals.PROT_COL]]))
        np.savetxt(directory+'predictions_UNSORTED.pred', z_unsorted, delimiter=',', fmt=['%d','%d','%f','%d'])

        #sorted prediction
        z_sorted = z.argsort()[::-1]
        z_sorted = np.transpose(np.array([list_id[z_sorted], doc_ids[z_sorted],z[z_sorted],X[:,Globals.PROT_COL][z_sorted]]))
        np.savetxt(directory+'predictions_SORTED.pred', z_sorted, delimiter=',', fmt=['%d', '%d', '%f', '%d'])