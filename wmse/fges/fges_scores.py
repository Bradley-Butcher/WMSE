from wmse.scores import weighted_mse, bic, qnml, BDeu, DiscreteData

import numpy as np
import pandas as pd
class WMSEScore:

    def __init__(self, dataset):
        self.dataset = dataset

    def local_score(self, child: int, parents: list):

        data = self.dataset

        parents = np.array(parents)

        return weighted_mse(data, child, parents)

    def local_score_diff_parents(self, node1, node2, parents):
        return self.local_score(node2, parents + [node1]) - self.local_score(node2, parents)

    def local_score_diff(self, node1, node2):
        return self.local_score_diff_parents(node1, node2, [])

class BICScore:

    def __init__(self, dataset):
        self.dataset = dataset

    def local_score(self, child: int, parents: list):

        data = self.dataset

        parents = np.array(parents)

        return bic(data, child, parents)

    def local_score_diff_parents(self, node1, node2, parents):
        return self.local_score(node2, parents + [node1]) - self.local_score(node2, parents)

    def local_score_diff(self, node1, node2):
        return self.local_score_diff_parents(node1, node2, [])


class qNMLScore:

    def __init__(self, dataset):
        self.dataset = dataset

    def local_score(self, child: int, parents: list):

        data = self.dataset

        parents = np.array(parents)

        return qnml(data, child, parents)

    def local_score_diff_parents(self, node1, node2, parents):
        return self.local_score(node2, parents + [node1]) - self.local_score(node2, parents)

    def local_score_diff(self, node1, node2):
        return self.local_score_diff_parents(node1, node2, [])


class BDeuScore:
    def __init__(self, dataset, equivalent_sample_size=1.0):
        self.columns = dataset.columns
        self.dataset = dataset
        dataset = dataset.apply(lambda col: pd.Categorical(col))
        data = DiscreteData(data_source=dataset)
        self.bdeu = BDeu(alpha=equivalent_sample_size, data=data)
    
    def local_score(self, child, parents):
        if parents is not None and len(parents) > 0:
            return self.bdeu.bdeu_score(self.columns[child], tuple(self.columns[p] for p in parents))[0]
        else:
            return self.bdeu.bdeu_score(self.columns[child], tuple([]))[0]



    def local_score_diff_parents(self, node1, node2, parents):
        """
        Method to compute the change in score resulting
        from adding node1 to the list of parents.
        :param node1: int representing the node to add
                      to list of parents.
        :param node2: int representing the node in question.
        :param parents: list of ints representing the parent nodes.
        """
        return self.local_score(node2, parents + [node1]) - self.local_score(node2, parents)

    def local_score_diff(self, node1, node2):
        """
        Method to compute the change in score resulting
        from having node1 as a parent.
        :param node1: int representing the parent node.
        :param node2: int representing the node in question.
        """
        return self.local_score_diff_parents(node1, node2, [])