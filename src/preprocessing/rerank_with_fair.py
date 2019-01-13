import preprocessing.fair.post_processing_methods.fair_ranker.create as fair
from preprocessing.fair.dataset_creator.candidate import Candidate

import pandas as pd


def rerank_featurevectors(dataDescription, p_deviation=0.0):
    file = open(dataDescription.path, 'r')
    data = None
    if dataDescription.header_in_file == None:
        data = pd.read_csv(file, header=dataDescription.header_in_file, names=dataDescription.header)
    else:
        data = pd.read_csv(file)

    data['uuid'] = 'empty'

    reranked_features = pd.DataFrame()

    # re-rank with fair for every query
    for query in data[dataDescription.query_id].unique():
        #print("Rerank for query " + str(query))
        data_query = data.query(dataDescription.query_id + "==" + str(query))

        data_query, protected, nonProtected = create(data_query, dataDescription)

        p = (len(data_query.query(dataDescription.protected_attribute + "==" + str(dataDescription.protected_attribute_value))) / len(data_query) + p_deviation)

        if("TREC" in dataDescription.path):
            p = 0.105 + p_deviation

        fairRanking, fairNotSelected = fair.fairRanking(dataDescription.k, protected, nonProtected, p, dataDescription.alpha)
        fairRanking = setNewQualifications(fairRanking)

        # swap original qualification with fair qualification
        for candidate in fairRanking:
            # candidate_row = pd.DataFrame(data_query.query("uuid=='" + str(candidate.uuid) + "'"))
            candidate_row = data_query[data_query.uuid == candidate.uuid]

            candidate_row.iloc[0, data_query.columns.get_loc(dataDescription.judgment)] = (candidate.qualification / len(fairRanking) )

            reranked_features = reranked_features.append(candidate_row.iloc[0])

    # sort by judgement to ease evaluation of output
    reranked_features_sorted = pd.DataFrame()
    for query in data[dataDescription.query_id].unique():
        sorted = reranked_features.query(dataDescription.query_id + "==" + str(query)).sort_values(by=dataDescription.judgment, ascending=False)
        reranked_features_sorted = reranked_features_sorted.append(sorted)

    reranked_features_sorted.update(reranked_features_sorted[dataDescription.query_id].astype(int).astype(str))
    reranked_features_sorted.to_csv(dataDescription.description, sep=',', index=False, columns=dataDescription.header_to_write, header=False)


def create(data, dataDescription):
    protected = []
    nonProtected = []

    for row in data.itertuples():
        # change to different index in row[.] to access other columns from csv file
        if row[data.columns.get_loc(dataDescription.protected_attribute) + 1] == 0.:
            candidate = Candidate(row[data.columns.get_loc(dataDescription.judgment) + 1], [])
            nonProtected.append(candidate)
            data.ix[row.Index, "uuid"] = candidate.uuid

        else:
            candidate = Candidate(row[data.columns.get_loc(dataDescription.judgment) + 1], dataDescription.protected_group)
            protected.append(candidate)
            data.ix[row.Index, "uuid"] = candidate.uuid

    # sort candidates by judgment
    protected.sort(key=lambda candidate: candidate.qualification, reverse=True)
    nonProtected.sort(key=lambda candidate: candidate.qualification, reverse=True)

    return data, protected, nonProtected


def setNewQualifications(fairRanking):
    qualification = len(fairRanking)
    for candidate in fairRanking:
        candidate.qualification = qualification
        qualification -= 1
    return fairRanking

