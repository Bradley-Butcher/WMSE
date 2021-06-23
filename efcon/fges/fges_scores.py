import numpy as np
from numba import njit, float64, int64
from baynet import DAG
import pandas as pd

from typing import Dict
from efcon.score_function import weighted_mse, bic, qnml

from math import lgamma, log, pi
from itertools import combinations

from scipy.special import digamma, gammaln
from scipy.stats import norm, entropy

from numba import jit, njit

adtree_available = False



# @njit
# def cartesian(arrays, out=None):
#     arrays = [np.asarray(x) for x in arrays]
#     dtype = arrays[0].dtype

#     n_i = [x.size for x in arrays]
#     n = 1
#     for item in n_i:
#         n *= item
#     if out is None:
#         out = np.zeros((n, len(arrays)), dtype=np.int64)

#     m = int(n / arrays[0].size)
#     out[:, 0] = np.repeat(arrays[0], m)
#     if arrays[1:]:
#         cartesian(arrays[1:], out=out[0:m, 1:])
#         for j in range(1, arrays[0].size):
#             out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
#     return out

# @njit
# def nb_mean(array, axis):
#     return np_apply_along_axis(np.mean, axis, array)

# @njit
# def nb_sum(array, axis):
#     return np_apply_along_axis(np.sum, axis, array)

# @njit
# def nb_all(array, axis):
#     return np_apply_along_axis(np.all, axis, array)

# @njit
# def np_apply_along_axis(func1d, axis, arr):
#     assert arr.ndim == 2
#     assert axis in [0, 1]
#     if axis == 0:
#         result = np.empty(arr.shape[1])
#         for i in range(len(result)):
#             result[i] = func1d(arr[:, i])
#     else:
#         result = np.empty(arr.shape[0])
#         for i in range(len(result)):
#             result[i] = func1d(arr[i, :])
#     return result

# @njit
# def nb_argwhere_all(data, value):
#     indicies = np.zeros(data.shape[0], dtype=np.bool_)
#     for i in range(data.shape[0]):
#         eval = data[i] == value
#         prod = 1
#         if data.ndim > 1:
#             for e in eval:
#                 prod *= e
#         else:
#             prod *= eval
#         indicies[i] = prod
#     return np.where(indicies)[0]

# @njit
# def get_pygx_px(data: np.ndarray, y_idx: int, x_idxs: np.ndarray):
#     parent_levels = [np.unique(data[:, x_idxs[i]]) for i in range(len(x_idxs))]
#     child_levels = np.unique(data[:, y_idx])
#     parent_combinations = cartesian(parent_levels)

#     p_ykgxjs = np.zeros((len(child_levels), len(parent_combinations)), dtype=np.float64)
#     p_xjs = np.zeros(len(parent_combinations), dtype=np.float64)

#     for a, j in enumerate(parent_combinations):
#         # parent_rows = np.argwhere(np.all(data[:, x_idxs] == j, axis=1)).flatten()
#         parent_rows = nb_argwhere_all(data[:, x_idxs], j)
#         parent_data = data[parent_rows, :]
#         if parent_data.shape[0] == 0:
#             p_xjs[a] = 0
#             p_ykgxjs[:, a] = 0
#             continue
#         p_xjs[a] = len(parent_rows) / len(data)
#         for b, k in enumerate(child_levels):
#             child_rows = nb_argwhere_all(parent_data[:, y_idx], k)
#             # child_rows = np.argwhere(data[parent_rows, y_idx] == k).flatten()
#             p_ykgxjs[b, a] = len(child_rows) / len(parent_rows)

#     return p_ykgxjs, p_xjs

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

class Data:
    """
    Complete data (either discrete or continuous)

    This is an abstract class
    """

    def rawdata(self):
        '''
        The data without any information about variable names.

        Returns:
         numpy.ndarray: The data
        '''
        return self._data

    def variables(self):
        '''
        Returns:
         list : The variable names
        '''
        return self._variables

    def varidx(self):
        '''
        Returns:
         dict : Maps a variable name to its position in the list of variable names.
        '''
        return self._varidx

class DiscreteData(Data):
    """
    Complete discrete data
    """

    _value_type = np.uint8
    _arity_type = np.uint8
    _count_type = np.uint32
    
    def __init__(self, data_source, varnames = None, arities = None):
        '''Initialises a `DiscreteData` object.

        If  `data_source` is a filename then it is assumed that:

            #. All values are separated by whitespace
            #. Empty lines are ignored
            #. Comment lines start with a '#'
            #. The first line is a header line stating the names of the 
               variables
            #. The second line states the arities of the variables
            #. All other lines contain the actual data

        Args:
          data_source (str/array_like/Pandas.DataFrame) : 
            Either a filename containing the data or an array_like object or
            Pandas data frame containing it.

          varnames (iter) : 
           Variable names corresponding to columns in the data.
           Ignored if `data_source` is a filename or Pandas DataFrame (since they 
           will supply the variable names). Otherwise if not supplied (`=None`)
           then variables names will be: X1, X2, ...

          arities (iter) : 
           Arities for the variables corresponding to columns in the data.
           Ignored if `data_source` is a filename or Pandas DataFrame (since they 
           will supply the arities). Otherwise if not supplied (`=None`)
           the arity for each variable will be set to the number of distinct values
           observed for that variable in the data.
        '''

        if type(data_source) == str:
            with open(data_source, "r") as file:
                line = file.readline().rstrip()
                while len(line) == 0 or line[0] == '#':
                    line = file.readline().rstrip()
                varnames = line.split()
                line = file.readline().rstrip()
                while len(line) == 0 or line[0] == '#':
                    line = file.readline().rstrip()
                arities = np.array([int(x) for x in line.split()],dtype=self._arity_type)

                for arity in arities:
                    if arity < 2:
                        raise ValueError("This line: '{0}' is interpreted as giving variable arities but the value {1} is less than 2.".format(line,arity))

                # class whose instances are callable functions 'with memory'
                class Convert:
                    def __init__(self):
                        self._last = 0 
                        self._dkt = {}

                    def __call__(self,s):
                        try:
                            return self._dkt[s]
                        except KeyError:
                            self._dkt[s] = self._last
                            self._last += 1
                            return self._dkt[s]


                converter_dkt = {}
                for i in range(len(varnames)):
                    # trick to create a function 'with memory'
                    converter_dkt[i] = Convert()
                data = np.loadtxt(file,
                                  dtype=self._value_type,
                                  converters=converter_dkt,
                                  comments='#')

        elif type(data_source) == pd.DataFrame:
            data, arities, varnames = fromdataframe(data_source)
        else:
            data = np.array(data_source,dtype=self._value_type)
        self._data = data
        if arities is None:
            self._arities = np.array([x+1 for x in data.max(axis=0)],dtype=self._arity_type)
        else:
            self._arities = np.array(arities,dtype=self._arity_type)

        # ensure _variables is immutable _varidx is always correct.
        if varnames is None:
            self._variables = tuple(['X{0}'.format(i) for i in range(1,len(self._arities)+1)])
        else:
            # order of varnames determined by header line in file, if file used
            self._variables = tuple(varnames)

        self._unique_data, counts = np.unique(self._data, axis=0, return_counts=True)
        self._unique_data_counts = np.array(counts,self._count_type)
            
        self._maxflatcontabsize = 1000000

        self._varidx = {}
        for i, v in enumerate(self._variables):
            self._varidx[v] = i
        self._data_length = data.shape[0]

        # create AD tree, if possible
        if adtree_available:
            #dd = np.transpose(np.vstack((self._arities,self._data))).astype(np.uint8,casting='safe')
            #print(dd)
            self._adtree = return_adtree(2000,1000,1000,np.transpose(np.vstack((self._arities,self._data))).astype(np.uint8,casting='safe'))
        
        
    def data(self):
        '''
        The data with all values converted to unsigned integers.

        Returns:
         pandas.DataFrame: The data
        '''

        df = pd.DataFrame(self._data,columns=self._variables)
        arities = self._arities
        for i, (name, data) in enumerate(df.items()):
            # ensure correct categories are recorded even if not
            # all observed in data
            df[name] = pd.Categorical(data,categories=range(arities[i]))
        return df
    
    def data_length(self):
        '''
        Returns:
         int: The number of datapoints in the data
        '''
        return self._data_length

    def arities(self):
        '''
        Returns:
         numpy.ndarray: The arities of the variables.
        '''
        return self._arities

    def arity(self,v):
        '''
        Args:
         v (str) : A variable name
        Returns:
         int : The arity of `v`
        '''

        return self._arities[self._varidx[v]]


    def contab(self,variables):
        cols = np.array([self._varidx[v] for v in variables], dtype=np.uint32)
        cols.sort() 
        return make_contab(self._unique_data,self._unique_data_counts,cols,self._arities[cols],self._maxflatcontabsize)[0]

    def make_contab_adtree(self,variables):
        '''
        Compute a marginal contingency table from data or report
        that the desired contingency table would be too big.
        
        Args:
         variables (iter): The variables in the marginal contingency table.


        Returns:
         tuple: 1st element is of type ndarray: 
          If the contingency table would have too many then the array is empty
          (and the 2nd element of the tuple should be ignored)
          else an array of counts of length equal to the product of the `arities`.
          Counts are in lexicographic order of the joint instantiations of the columns (=variables)
          2nd element: the 'strides' for each column (=variable)
        '''
        cols = np.array([self._varidx[v] for v in variables], dtype=np.uintc)
        cols.sort()
        p = len(cols)
        idx = p-1
        stride = 1
        arities = self._arities[cols]
        maxsize = self._maxflatcontabsize
        strides = np.empty(p,dtype=np.uint32)
        while idx > -1:
            strides[idx] = stride
            stride *= arities[idx]
            if stride > maxsize:
                return np.empty(0,dtype=np.uint32), strides
            idx -= 1
        #print(variables,cols,int(stride),flush=True)
        flatcontab = np.empty(stride,dtype=np.uintc)
        makecontab(self._adtree,cols,flatcontab)
        #print(flatcontab,flush=True)
        return flatcontab, strides

class BDeu(DiscreteData):
    """
    Discrete data with attributes and methods for BDeu scoring
    """

    def __init__(self,data,alpha=1.0):
        '''Initialises a `BDeu` object.

        Args:
         data (DiscreteData): data
         
         alpha (float): The *equivalent sample size*
        '''
        self.__dict__.update(data.__dict__)
        self.alpha = alpha
        self._cache = {}

        # for upper bounds
        self._atoms = get_atoms(self._data,self._arities)

        
    @property
    def alpha(self):
        '''float: The *equivalent sample size* used for BDeu scoring'''
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        '''Set the *equivalent sample size* for BDeu scoring
        
        Args:
         alpha (float): the *equivalent sample size* for BDeu scoring

        Raises:
         ValueError: If `alpha` is not positive
        '''
        if not alpha > 0:
            raise ValueError('alpha (equivalent sample size) must be positive but was give {0}'.format(alpha))
        self._alpha = alpha

    def clear_cache():
        '''Empty the cache of stored BDeu component scores

        This should be called, for example, if new scores are being computed
        with a different alpha value
        '''
        self._cache = {}
        
    def upper_bound_james(self,child,parents,alpha=None):
        """
        Compute an upper bound on proper supersets of parents

        Args:
         child (str) : Child variable.
         parents (iter) : Parent variables
         alpha (float) : ESS value for BDeu score. If not supplied (=None)
          then the value of `self.alpha` is used.

        Returns:
         float : An upper bound on the local score for parent sets
         for `child` which are proper supersets of `parents`

        """
        if alpha is None:
            alpha = self._alpha
        child_idx = self._varidx[child]
        pa_idxs = sorted([self._varidx[v] for v in parents])
        for pa_idx in pa_idxs:
            alpha /= self._arities[pa_idx]
        r = self._arities[child_idx]

        # each element of atoms_ints is a tuple of ints:
        # (fullinst,childvalcounts,sum(childvalcounts),ok_first)
        # each element of atoms_floats is:
        # sum_n n*log(n/tot), where sum is over childvalcounts
        # and tot = sum(childvalcounts)
        atoms_ints, atoms_floats = self._atoms[0][child_idx], self._atoms[1][child_idx]

        if len(atoms_floats) == 0:
            return 0.0

        # remove cols corresponding to non-parents and order
        p = len(self._arities)
        end_idxs = list(range(p,p+r+2))
        atoms_ints_redux = atoms_ints[:,pa_idxs+end_idxs]
        if len(pa_idxs) > 0:
            idxs = np.lexsort([atoms_ints_redux[:,col] for col in range(len(pa_idxs))])
        else:
            idxs = list(range(len(atoms_floats)))

        return upper_bound_james_fun(atoms_ints_redux,atoms_floats,len(pa_idxs),alpha,r,idxs)
        
    def bdeu_score_component(self,variables,alpha=None):
        '''Compute the BDeu score component for a set of variables
        (from the current dataset).

        The BDeu score for a child v having parents Pa is 
        the BDeu score component for Pa subtracted from that for v+Pa
        
        Args:
         variables (iter) : The names of the variables
         alpha (float) : The effective sample size parameter for the BDeu score.
          If not supplied (=None)
          then the value of `self.alpha` is used.

        Returns:
         float : The BDeu score component.
        '''
        if alpha is None:
            alpha = self._alpha

        if len(variables) == 0:
            return lgamma(alpha) - lgamma(alpha + self._data_length), 1
        else:
            cols = np.array(sorted([self._varidx[x] for x in list(variables)]), dtype=np.uint32)
            return compute_bdeu_component(
                self._unique_data,self._unique_data_counts,cols,
                np.array([self._arities[i] for i in cols], dtype=self._arity_type),
                alpha,self._maxflatcontabsize)

    def _bdeu_score_component_cache(self,s):
        s_set = frozenset(s)
        try:
            score_non_zero_count = self._cache[s_set]
        except KeyError:
            score_non_zero_count = self.bdeu_score_component(s_set)
            self._cache[s_set] = score_non_zero_count
        return score_non_zero_count

    def bdeu_score(self, child, parents):

        parent_score, _ = self._bdeu_score_component_cache(parents)
        family_score, non_zero_count = self._bdeu_score_component_cache((child,)+parents)

        simple_ub = -log(self.arity(child)) * non_zero_count

        #james_ub = self.upper_bound_james(child,parents)
        
        #return parent_score - family_score, min(simple_ub,james_ub)
        return parent_score - family_score, simple_ub

        
    def bdeu_scores(self,palim=None,pruning=True,alpha=None):
        """
        Exhaustively compute all BDeu scores for all child variables and all parent sets up to size `palim`.
        If `pruning` delete those parent sets which have a subset with a better score.
        Return a dictionary dkt where dkt[child][parents] = bdeu_score
        
        Args:
         palim (int/None) : Limit on parent set size
         pruning (bool) : Whether to prune
         alpha (float) : ESS for BDeu score. 
          If not supplied (=None)
          then the value of `self.alpha` is used.



        Returns:
         dict : dkt where dkt[child][parents] = bdeu_score
        """
        if alpha is None:
            alpha = self._alpha
        
        if palim == None:
            palim = self._arities.size - 1

        score_dict = {}
        # Initialisation
        # Need to create dict for every child
        # also its better to do the zero size parent set calc here
        # so that we don't have to do a check for every parent set
        # to make sure it is not of size 0 when calculating score component size
        no_parents_score_component = lgamma(alpha) - lgamma(alpha + self._data_length)
        for c, child in enumerate(self._variables):
            score_dict[child] = {
                frozenset([]):
                no_parents_score_component
            }
        
        for pasize in range(1,palim+1):
            for family in combinations(self._variables,pasize): 
                score_component = self.bdeu_score_component(family,alpha)
                family_set = frozenset(family)
                for child in self._variables:
                    if child in family_set:
                        parent_set = family_set.difference([child])
                        score_dict[child][parent_set] -= score_component
                        if pruning and prune_local_score(score_dict[child][parent_set],parent_set,score_dict[child]):
                            del score_dict[child][parent_set]
                    else:
                        score_dict[child][family_set] = score_component 
                
        # seperate loop for maximally sized parent sets
        for vars in combinations(self._variables,palim+1):
            score_component = self.bdeu_score_component(vars,alpha)
            vars_set = frozenset(vars)
            for child in vars:
                parent_set = vars_set.difference([child])
                score_dict[child][parent_set] -= score_component
                if pruning and prune_local_score(score_dict[child][parent_set],parent_set,score_dict[child]):
                    del score_dict[child][parent_set]

            
        return score_dict

def get_atoms(data,arities):
    '''
    Args: 
        data(np.array): Discrete data as a 2d array of ints

    Returns:
        list: a list `atoms`, where `atoms[i]` is a dictionary mapping instantations
         of variables other than i to a tuple with 3 elements:

          1. the child counts (n1, n2, .., nr) for that instantations
          2. n * the entropy for the empirical distribution (n1/n, n2/n, .., nr/n)
          3. Whether the chi-squared statistic for (n1, n2, .., nr) exceeds r-1 
        
        Chi-squared test comes from "Compound Multinomial Likelihood Functions are Unimodal: 
        Proof of a Conjecture of I. J. Good". Author(s): Bruce Levin and James Reeds
        Source: The Annals of Statistics, Vol. 5, No. 1 (Jan., 1977), pp. 79-87
    
        At least two of the ni must be positive for the inst-tuple pair to be included
        in the dictionary since only in that case is the inst-tuple useful for getting
        a good upper bound.
       
    '''
    fullinsts = []
    for i in range(data.shape[1]):
        fullinsts.append({})
    for row in data:
        row = tuple(row)
        for i, val in enumerate(row):
            fullinsts[i].setdefault(row[:i]+(0,)+row[i+1:],  # add dummy value for ith val
                                    [0]*arities[i])[val] += 1

    # now, for each child i, delete full insts which are 'deterministic'
    # i.e. where only one child value is non-zero
    newdkts_ints = []
    newdkts_floats = []
    for i, dkt in enumerate(fullinsts):
        #print('old',len(dkt))
        #newdkt = {}
        newdkt_ints = []
        newdkt_floats = []
        th = arities[i] - 1
        for inst, childcounts in dkt.items():
            if len([v for v in childcounts if v > 0]) > 1:
                newdkt_ints.append(inst+tuple(childcounts)+(sum(childcounts),chisq(childcounts) > th))
                #newdkt[inst] = (
                #    np.array(childcounts,np.uint64),sum(childcounts),
                #    chisq(childcounts) > th,
                #    h(childcounts))
                newdkt_floats.append(h(childcounts))
        newdkts_ints.append(np.array(newdkt_ints,np.uint64))
        newdkts_floats.append(np.array(newdkt_floats,np.float64))
        #print('new',len(newdkt))
    #fullinsts = newdkts

    #print(newdkts_ints[0])
    #sys.exit()

    #for x in newdkts_ints:
    #    print(x.shape)
    
    return newdkts_ints, newdkts_floats
    

def save_local_scores(local_scores, filename):
    variables = local_scores.keys()
    with open(filename, "w") as scores_file:
        scores_file.write(str(len(variables)))
        for child, dkt in local_scores.items():
            scores_file.write("\n" + child + " " + str(len(dkt.keys())))
            for parents, score in dkt.items():
                #highest_sup = None
                scores_file.write("\n" + str(score) + " " + str(len(parents)) +" "+ " ".join(parents))

def prune_local_score(this_score, parent_set, child_dkt):
    for other_parent_set, other_parent_set_score in child_dkt.items():
        if other_parent_set_score >= this_score and other_parent_set < parent_set:
            return True
    return False

def fromdataframe(df):
    cols = []
    arities = []
    varnames = []
    for varname, vals in df.items():
        varnames.append(varname)
        cols.append(vals.cat.codes)
        arities.append(len(vals.cat.categories))
    return np.transpose(np.array(cols,dtype=np.uint32)), arities, varnames


def chisq(counts):
    tot = sum(counts)
    t = len(counts)
    mean = tot/t #Python 3 - this creates a float
    chisq = 0.0
    for n in counts:
        chisq += (n - mean)**2
    return chisq/mean


@jit(nopython=True)
def marginalise_uniques_contab(unique_insts, counts, cols):
    '''
    Marginalise a contingency table represented by unique insts and counts
    '''
    marg_uniqs, indices = np.unique(unique_insts[:,cols], return_inverse=True)
    marg_counts = np.zeros(len(marg_uniqs),dtype=np.uint32)
    for i in range(len(counts)):
        marg_counts[indices[i]] += counts[i]
    return marg_uniqs, marg_counts

@jit(nopython=True)
def make_contab(data, counts, cols, arities, maxsize):
    '''
    Compute a marginal contingency table from data or report
    that the desired contingency table would be too big.

    All inputs except the last are arrays of unsigned integers

    Args:
     data (numpy array): the unique datapoints as a 2-d array, each row is a datapoint, assumed unique
     counts (numpy array): the count of how often each unique datapoint occurs in the original data
     cols (numpy array): the columns (=variables) for the marginal contingency table.
      columns must be ordered low to high
     arities (numpy array): the arities of the variables (=columns) for the contingency table
      order must match that of `cols`.
     maxsize (int): the maximum size (number of cells) allowed for a contingency table

    Returns:
     tuple: 1st element is of type ndarray: 
      If the contingency table would have more elements than `maxsize' then the array is empty
      (and the 2nd element of the tuple should be ignored)
      else an array of counts of length equal to the product of the `arities`.
      Counts are in lexicographic order of the joint instantiations of the columns (=variables)
      2nd element: the 'strides' for each column (=variable)
    '''
    p = len(cols)
    #if arities = (2,3,3) then strides = 9,3,1
    #if row is (2,1,2) then index is 2*9 + 1*3 + 2*1
    strides = np.empty(p,dtype=np.uint32)
    idx = p-1
    stride = 1
    while idx > -1:
        strides[idx] = stride
        stride *= arities[idx]
        if stride > maxsize:
            return np.empty(0,dtype=np.uint32), strides
        idx -= 1
    contab = np.zeros(stride,dtype=np.uint32)
    for rowidx in range(data.shape[0]):
        idx = 0
        for i in range(p):
            idx += data[rowidx,cols[i]]*strides[i]
        contab[idx] += counts[rowidx]
    return contab, strides

@jit(nopython=True)
def _compute_ll_from_flat_contab(contab,strides,child_idx,child_arity):
    child_stride = strides[child_idx]
    ll = 0.0
    child_counts = np.empty(child_arity,dtype=np.int32)
    contab_size = len(contab)
    for i in range(0,contab_size,child_arity*child_stride):
        for j in range(i,i+child_stride):
            n = 0
            for k in range(child_arity):
                count = contab[j]
                child_counts[k] = count
                n += count
                j += child_stride
            if n > 0:
                for c in child_counts:
                    if c > 0:
                        ll += c * log(c/n)
    return ll

@jit(nopython=True)
def _compute_ll_from_unique_contab(data,counts,n_uniqs,pa_idxs,child_arity,orig_child_col):
    child_counts = np.zeros((n_uniqs,np.int64(child_arity)),dtype=np.uint32)
    for i in range(len(data)):
        child_counts[pa_idxs[i],data[i,orig_child_col]] += counts[i]
    ll = 0.0
    for i in range(len(child_counts)):
        #n = child_counts[i,:].sum() #to consider
        n = 0
        for k in range(child_arity):
            n += child_counts[i,k]
        for k in range(child_arity):
            c = child_counts[i,k]
            if c > 0:
                ll += c * log(c/n)
    return ll

@jit(nopython=True)
def compute_bdeu_component(data, counts, cols, arities, alpha, maxflatcontabsize):
    contab = make_contab(data, counts, cols, arities, maxflatcontabsize)[0]
    if len(contab) > 0:
        alpha_div_arities = alpha / len(contab)
        non_zero_count = 0
        score = 0.0
        for count in contab:
            if count != 0:
                non_zero_count += 1
                score -= lgamma(alpha_div_arities+count) 
        score += non_zero_count*lgamma(alpha_div_arities)  
        return score, non_zero_count


# START functions for upper bounds

@jit(nopython=True)
def lg(n,x):
    return lgamma(n+x) - lgamma(x)

def h(counts):
    '''
    log P(counts|theta) where theta are MLE estimates computed from counts
    ''' 
    tot = sum(counts)
    #print(tot)
    res = 0.0
    for n in counts:
        if n > 0:
            res += n*log(n/tot)
    return res

def chisq(counts):
    tot = sum(counts)
    t = len(counts)
    mean = tot/t #Python 3 - this creates a float
    chisq = 0.0
    for n in counts:
        chisq += (n - mean)**2
    return chisq/mean

def hsum(distss):
    res = 0.0
    for dists in distss:
        res += sum([h(d) for d in dists])
    return res

@jit(nopython=True)
def fa(dist,sumdist,alpha,r):
    #res = -lg(sumdist,alpha)
    res = lgamma(alpha) - lgamma(sumdist+alpha)
    alphar = alpha/r
    k = 0
    for n in dist:
        if n > 0:
            #res += lg(n,alphar)
            #res += (lgamma(n+alphar) - lgamma(alphar))
            res += lgamma(n+alphar)
            k += 1
    return res - k*lgamma(alphar)

def diffa(dist,alpha,r):
    """Compute the derivative of local-local BDeu score

    numba does not support the scipy.special functions -- we are using the
    digamma function in defining the entropy of the Chisq
    distribution. For this end I added a python script which contains code
    for a @vectorize-d digamma function. If we need to use anything from
    scipy.special we will have to write it up ourselves.

    """
    args = [n+alpha/r for n in dist] + [alpha,sum(dist)+alpha,alpha/r]
    z = digamma(args)
    return sum(z[:r+1]) - z[r+1] - r*z[r+2] 

def onepositive(dist):
    """
    Is there only one positive count in `dist`?
    """
    npos = 0
    for n in dist:
        if n > 0:
            npos += 1
            if npos > 1:
                return False
    return True

@njit
def array_equal(a,b,pasize):
    for i in range(pasize):
        if a[i] != b[i]:
            return False
    return True

#@njit
#def get_elems(a,idxs):
#    return np.a

@njit
def upper_bound_james_fun(atoms_ints,atoms_floats,pasize,alpha,r,idxs):
    ub = 0.0
    local_ub = 0.0
    best_diff = 0.0
    pasize_r = pasize+r
    lr = -log(r)
    oldrow = atoms_ints[idxs[0]]
    for i in idxs:
        row = atoms_ints[i]
        
        if not array_equal(oldrow,row,pasize):
            ub += min(local_ub+best_diff,lr)
            best_diff = 0.0
            local_ub = 0.0
            oldrow = row

        mle = atoms_floats[i]
        local_ub += mle
        if row[-1]: # if chi-sq condition met
            diff = fa(row[pasize:pasize_r],row[pasize_r],alpha/2.0,r) - mle
            if diff < best_diff:
                best_diff = diff
    ub += min(local_ub+best_diff,lr)
    return ub


#@jit(nopython=True)
def ub(dists,alpha,r):
    '''
    Args:
        dists (iter): list of (child-value-counts,chisqtest,mles) lists, for some
          particular instantiation ('inst1') of the current parents.
          There is a child-value-counts list for each non-zero instantiation ('inst2') of the biggest
          possible superset of the current set of parents. inst1 and each inst2 have the same
          values for the current parents.
        alpha (float): ESS/(num of current parent insts)
        r (int): arity of the child

    Returns:
       float: An upper bound on the BDeu local score for (implicit) child
        over all possible proper supersets of the current parents
    '''
    naives = 0.0
    best_diff = 0.0
    for (dist,sumdist,ok_first,naive_ub) in dists:
        naives += naive_ub
        #iffirst_ub = naive_ub
        #iffirst_ub = min(len(dist)*-log(r),naive_ub)
        if ok_first:
            diff = fa(dist,sumdist,alpha/2.0,r) - naive_ub
            #iffirst_ub = min(iffirst_ub,fa(dist,sumdist,alpha/2.0,r))
            #diff = iffirst_ub - naive_ub
            if diff < best_diff:
                best_diff = diff
    return best_diff + naives
