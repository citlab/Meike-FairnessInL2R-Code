from preprocessing.datasetDescription import DatasetDescription
import preprocessing.rerank_with_fair as rerank
import pandas as pd

class Preprocessing():

    def __init__(self, dataset, p):
        self.dataset = dataset
        if p == "p_minus":
            self.p = -0.1
            self.description_classifier = "RERANKED_PMinus"
        elif p == "p_plus":
            self.p = 0.1
            self.description_classifier = "RERANKED_PPlus"
        else:
            self.p = 0.0
            self.description_classifier = "RERANKED"

    def preprocess_dataset(self):

        if self.dataset == "trec":
            """
            TREC Data
        
            """
            print("Start reranking of TREC Data")
            query_id = "query_id"
            protected_attribute = "gender"
            protected_attribute_value = 1
            protected_group = "female"
            header = ["query_id", "gender", "match_body_email_subject_score_norm", "match_body_email_subject_df_stdev",
                      "match_body_email_subject_idf_stdev", "match_body_score_norm", "match_subject_score_norm", "judgment"]
            header_in_file = None
            header_to_write = ["query_id", "gender", "match_body_email_subject_score_norm", "match_body_email_subject_df_stdev",
                               "match_body_email_subject_idf_stdev", "match_body_score_norm", "match_subject_score_norm", "judgment"]
            judgment = "judgment"
            k = 200

            for fold in ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5", "fold_6"]:
                print("Reranking for " + fold)
                path = "../octave-src/sample/TREC/" + fold + "/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train.csv"
                description = "../octave-src/sample/TREC/" + fold + "/features_withListNetFormat_withGender_withZscore_candidateAmount-200_train_" + self.description_classifier + ".csv"
                TRECData = DatasetDescription(description, path, query_id, protected_attribute, protected_attribute_value,
                                              protected_group, header, header_in_file, header_to_write, judgment, k)

                rerank.rerank_featurevectors(TRECData, self.p)

        if self.dataset == "law":
            """
            LSAT Data - Gender
            
            """
            print("Start reranking of LSAT Data")
            print("protected attribute: sex")

            description = "../octave-src/sample/LawStudents/gender/LawStudents_Gender_train_" + self.description_classifier + ".txt"
            path = "../octave-src/sample/LawStudents/gender/LawStudents_Gender_train.txt"
            query_id = "query_id"
            protected_attribute = "sex"
            protected_attribute_value = 1
            protected_group = "female"
            header = ["query_id", "sex", "LSAT", "UGPA", "ZFYA"]
            header_in_file = None
            header_to_write = ["query_id", "sex", "LSAT", "UGPA", "ZFYA"]
            judgment = "ZFYA"
            k = len(pd.read_csv(path, header=None))

            LSATGenderData = DatasetDescription(description, path, query_id, protected_attribute,
                                                protected_attribute_value, protected_group,
                                                header, header_in_file, header_to_write, judgment, k)

            rerank.rerank_featurevectors(LSATGenderData, self.p)

            """
            LSAT Data - Race - Asian
            
            """
            print("Start reranking of LSAT Data")
            print("protected attribute: race - protected group: asian")

            description = "../octave-src/sample/LawStudents/race_asian/LawStudents_Race_train_" + self.description_classifier + ".txt"
            path = "../octave-src/sample/LawStudents/race_asian/LawStudents_Race_train.txt"
            query_id = "query_id"
            protected_attribute = "race"
            protected_attribute_value = 1
            protected_group = "asian"
            header = ["query_id", "race", "LSAT", "UGPA", "ZFYA"]
            header_in_file = None
            header_to_write = ["query_id", "race", "LSAT", "UGPA", "ZFYA"]
            judgment = "ZFYA"
            k = len(pd.read_csv(path, header=None))

            LSATRaceAsianData = DatasetDescription(description, path, query_id, protected_attribute,
                                                protected_attribute_value, protected_group,
                                                header, header_in_file, header_to_write, judgment, k)

            rerank.rerank_featurevectors(LSATRaceAsianData, self.p)

            """
            LSAT Data - Race - Black
            
            """
            print("Start reranking of LSAT Data")
            print("protected attribute: race - protected group: black")

            description = "../octave-src/sample/LawStudents/race_black/LawStudents_Race_train_" + self.description_classifier + ".txt"
            path = "../octave-src/sample/LawStudents/race_black/LawStudents_Race_train.txt"
            query_id = "query_id"
            protected_attribute = "race"
            protected_attribute_value = 1
            protected_group = "black"
            header = ["query_id", "race", "LSAT", "UGPA", "ZFYA"]
            header_in_file = None
            header_to_write = ["query_id", "race", "LSAT", "UGPA", "ZFYA"]
            judgment = "ZFYA"
            k = len(pd.read_csv(path, header=None))

            LSATRaceBlackData = DatasetDescription(description, path, query_id, protected_attribute,
                                                protected_attribute_value, protected_group,
                                                header, header_in_file, header_to_write, judgment, k)

            rerank.rerank_featurevectors(LSATRaceBlackData, self.p)

            """
            LSAT Data - Race - Hispanic
            
            """
            print("Start reranking of LSAT Data")
            print("protected attribute: race - protected group: hispanic")

            description = "../octave-src/sample/LawStudents/race_hispanic/LawStudents_Race_train_" + self.description_classifier + ".txt"
            path = "../octave-src/sample/LawStudents/race_hispanic/LawStudents_Race_train.txt"
            query_id = "query_id"
            protected_attribute = "race"
            protected_attribute_value = 1
            protected_group = "hispanic"
            header = ["query_id", "race", "LSAT", "UGPA", "ZFYA"]
            header_in_file = None
            header_to_write = ["query_id", "race", "LSAT", "UGPA", "ZFYA"]
            judgment = "ZFYA"
            k = len(pd.read_csv(path, header=None))

            LSATRaceHispanicData = DatasetDescription(description, path, query_id, protected_attribute,
                                                protected_attribute_value, protected_group,
                                                header, header_in_file, header_to_write, judgment, k)

            rerank.rerank_featurevectors(LSATRaceHispanicData, self.p)

            """
            LSAT Data - Race - mexican
            
            """
            print("Start reranking of LSAT Data")
            print("protected attribute: race - protected group: mexican")

            description = "../octave-src/sample/LawStudents/race_mexican/LawStudents_Race_train_" + self.description_classifier + ".txt"
            path = "../octave-src/sample/LawStudents/race_mexican/LawStudents_Race_train.txt"
            query_id = "query_id"
            protected_attribute = "race"
            protected_attribute_value = 1
            protected_group = "mexican"
            header = ["query_id", "race", "LSAT", "UGPA", "ZFYA"]
            header_in_file = None
            header_to_write = ["query_id", "race", "LSAT", "UGPA", "ZFYA"]
            judgment = "ZFYA"
            k = len(pd.read_csv(path, header=None))

            LSATRaceMexicanData = DatasetDescription(description, path, query_id, protected_attribute,
                                                protected_attribute_value, protected_group,
                                                header, header_in_file, header_to_write, judgment, k)

            rerank.rerank_featurevectors(LSATRaceMexicanData, self.p)

            """
            LSAT Data - Race - puertorican
            
            """
            print("Start reranking of LSAT Data")
            print("protected attribute: race - protected group: puertorican")

            description = "../octave-src/sample/LawStudents/race_puertorican/LawStudents_Race_train_" + self.description_classifier + ".txt"
            path = "../octave-src/sample/LawStudents/race_puertorican/LawStudents_Race_train.txt"
            query_id = "query_id"
            protected_attribute = "race"
            protected_attribute_value = 1
            protected_group = "puertorican"
            header = ["query_id", "race", "LSAT", "UGPA", "ZFYA"]
            header_in_file = None
            header_to_write = ["query_id", "race", "LSAT", "UGPA", "ZFYA"]
            judgment = "ZFYA"
            k = len(pd.read_csv(path, header=None))

            LSATRacePuertoricanData = DatasetDescription(description, path, query_id, protected_attribute,
                                                protected_attribute_value, protected_group,
                                                header, header_in_file, header_to_write, judgment, k)

            rerank.rerank_featurevectors(LSATRacePuertoricanData, self.p)

        if self.dataset == "engineering-NoSemi":

            """
            ChileUniversity Data - NoSemi - gender
        
            """
            print("Start reranking of ChileUniversity Data - NoSemi - gender")
            query_id = "ano_in"
            protected_attribute = "hombre"
            protected_attribute_value = 1
            protected_group = "female"
            header = ['ano_in', 'hombre', 'psu_mat', 'psu_len', 'psu_cie', 'nem', 'score']
            header_in_file = None
            header_to_write = ['ano_in', 'hombre', 'psu_mat', 'psu_len', 'psu_cie', 'nem', 'score']
            judgment = "score"

            fold_count = 1
            for fold in ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]:

                print("Reranking for " + fold)
                path = "../octave-src/sample/ChileUni/NoSemi/gender/" + fold + "/chileDataL2R_gender_nosemi_fold" + str(fold_count) + "_train.txt"
                description = "../octave-src/sample/ChileUni/NoSemi/gender/" + fold + "/chileDataL2R_gender_nosemi_fold" + str(fold_count) + "_train_" + self.description_classifier + ".txt"
                k = len(pd.read_csv(path, header=None))
                ChileUniversityData = DatasetDescription(description, path, query_id, protected_attribute, protected_attribute_value,
                                              protected_group, header, header_in_file, header_to_write, judgment, k)

                rerank.rerank_featurevectors(ChileUniversityData, self.p)

                fold_count += 1

            """
            ChileUniversity Data - NoSemi - highschool
        
            """
            print("Start reranking of ChileUniversity Data - NoSemi - highschool")
            query_id = "ano_in"
            protected_attribute = "highschool_type"
            protected_attribute_value = 1
            protected_group = "highschool"
            header = ['ano_in', 'highschool_type', 'psu_mat', 'psu_len', 'psu_cie', 'nem', 'score']
            header_in_file = None
            header_to_write = ['ano_in', 'highschool_type', 'psu_mat', 'psu_len', 'psu_cie', 'nem', 'score']
            judgment = "score"

            fold_count = 1
            for fold in ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]:

                print("Reranking for " + fold)
                path = "../octave-src/sample/ChileUni/NoSemi/highschool/" + fold + "/chileDataL2R_highschool_nosemi_fold" + str(fold_count) + "_train.txt"
                description = "../octave-src/sample/ChileUni/NoSemi/highschool/" + fold + "/chileDataL2R_highschool_nosemi_fold" + str(fold_count) + "_train_" + self.description_classifier + ".txt"
                k = len(pd.read_csv(path, header=None))
                ChileUniversityData = DatasetDescription(description, path, query_id, protected_attribute, protected_attribute_value,
                                              protected_group, header, header_in_file, header_to_write, judgment, k)

                rerank.rerank_featurevectors(ChileUniversityData, self.p)

                fold_count += 1