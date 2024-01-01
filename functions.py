import uuid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress, combinations
from scipy.stats import chi2, binom
from sklearn import metrics
from collections import Counter


def generate_job_id():
    return str(uuid.uuid4())


def merge_matr_dicts(dict1, dict2): # merging the dictionaries for the matrix terms
  res = dict(dict1, **dict2)
  res.update((k, dict1[k] + dict2[k]) for k in set(dict1).intersection(dict2))
  return res

# the return table is our table of interest in which we want to change the names
# it should contain the columns 'protein A', 'protein B'
# the 'file' contains two columns: 'original' and 'gene' (if we want this the other way )
# Function to calculate the spokes term (table of relevant sub-terms) of the SA formula
# but also incorporating experimental confidence
# Input should contain the columns: bait, prey, identifier and conf
def update_spoke_terms(old_spoke_data, new_pair, diff_exp=False):
    ## old_spoke_data i the record of this function
    ## new_pair comes in two columns, first column is bait, second is prey
    # diff_exp=True indicates whether or not the data comes from the same experiment as already accounted
    ind_b = (old_spoke_data['bait'] == new_pair[0])
    ind_p = (old_spoke_data['prey'] == new_pair[1])
    if diff_exp:
        old_spoke_data['tot_pur'] = old_spoke_data['tot_pur'] + 1
        old_spoke_data.loc[ind_b, 'n_i_bait'] = old_spoke_data.loc[ind_b, 'n_i_bait'] + 1
    old_spoke_data['all_pairs'] = old_spoke_data['all_pairs'] + 1
    old_spoke_data.loc[(ind_b) & (ind_p),'O_spoke'] = old_spoke_data.loc[(ind_b) & (ind_p),'O_spoke'] + 1
    old_spoke_data.loc[ind_b, 'n_i_prey'] = old_spoke_data.loc[ind_b, 'n_i_prey'] + 1
    old_spoke_data.loc[ind_p, 'n_j_prey'] = old_spoke_data.loc[ind_p, 'n_j_prey'] + 1
    return old_spoke_data

def complete_S(s_data):
    s_data['f_i_bait'] = s_data['n_i_bait'].div(s_data.tot_pur)
    s_data['f_j_prey'] = s_data['n_j_prey'].div(s_data.all_pairs)
    s_data['f_i_prey'] = s_data['n_i_prey'].div(s_data.n_i_bait)
    s_data['E_spoke'] = (s_data.iloc[:, 3:6].prod(axis=1)).multiply(s_data.tot_pur)
    s_data['s_ij'] = np.log(s_data['O_spoke'].div(s_data['E_spoke']))
    s_data['Lambda_ij'] = s_data['O_spoke'] * s_data['s_ij']
    return s_data

## In case I want to add multiple purifications at the same time,
# Do I need also to update the prey_id_dict? I don't think so...
def add_to_matr_list(fn_new_data, old_list): # new_data is a df w/ columns [bait, prey, identifier]
    # old_list = {[matrix pair]: count]}
    new_pairs = [';'.join(sorted([x, y], reverse = True)) for n, g in
                 fn_new_data.iloc[:,[1,2]].groupby('identifier').prey
                 for x, y in combinations(g, 2)]
    for i in new_pairs:
        if i in list(old_list.keys()):
            old_list[i] = old_list[i] + 1
        else:
            old_list[i] = 1
    return old_list

def compute_spoke_ij_conf_weighted(data, use_conf = True, do_conf_intervals = False, with_shrinkage=False, shrink_param = 0.5):
    # 1. f_i^{bait} = fraction of purifications where (protein/ trait) i was bait & n_bait = total_purifications
    bait_identif = data.drop_duplicates(subset = ['identifier', 'bait']) # for gwas4 these are pubmed and trait; for Intact they are protein A and interaction ID
    total_purifications = np.shape(bait_identif)[0] # i.e. # of unique experiments (by ID), meaning, repeated baits sometimes (= n_bait)
    n_i_bait = pd.DataFrame.from_dict(Counter(bait_identif['bait']),orient='index').unstack().reset_index()[['level_1',0]].rename(columns={'level_1': 'bait', 0: 'freq'})
    f_i_bait = n_i_bait.copy()
    f_i_bait['freq'] = f_i_bait['freq'].div(total_purifications)
    # 2. n_i^{prey} = number of preys retrieved for each particular bait (i)
    purif_preys = data.groupby(['identifier','bait']).size().reset_index(name="n_i_prey") # purification identifiers (e.g. pubmed+trait) + # of preys
    result = pd.merge(bait_identif,purif_preys,on=['identifier','bait']) #
    n_i_prey = (result[['bait', 'n_i_prey']].groupby(['bait']).sum()).reset_index()
    n_i_prey = pd.merge(n_i_prey, n_i_bait, on = 'bait')
    n_i_prey['n_i_prey'] = n_i_prey['n_i_prey'].div(n_i_prey['freq'])
    n_i_prey = n_i_prey.iloc[:,[0,1]]
    # 3. f_j^{prey} = fraction of all retrieved preys that were (protein/ gene) j
    if use_conf:
        f_j_prey = data.groupby(['prey'], as_index = False).sum().rename(columns={'conf':"f_j_prey"}) # prey in GWAS is gene and in Intact - protein B
    else:
        f_j_prey = data.groupby(['prey']).size().reset_index(name="f_j_prey")
    f_j_prey['f_j_prey'] = f_j_prey['f_j_prey'].div(sum(f_j_prey['f_j_prey']))
    # 4. n_{i,j} = number of times that i retrieves j when i is tagged
    if use_conf:
        sa = data[['bait', 'prey','conf']].groupby(['bait', 'prey']).sum().reset_index().rename(columns={'conf':'O_spoke'}).sort_values(by = 'n',ascending=False)
    else:
        sa = data.groupby(['bait', 'prey']).size().reset_index(name="O_spoke").sort_values(by = 'O_spoke',ascending=False)
    # 5. Summary: putting it all together
    s = pd.merge(sa, f_j_prey, how = "left", on="prey")
    s = pd.merge(s, n_i_prey, how = "left", on="bait")
    s = pd.merge(s, f_i_bait, how = "left", on="bait").rename(columns={'freq': 'f_i_bait'})
    s['E_spoke'] = (s.iloc[:,3:6].prod(axis=1)).multiply(total_purifications)
    if with_shrinkage:
        s['s_ij'] = np.log((s['O_spoke']+shrink_param).div(s['E_spoke']+shrink_param)) # ln by default, if another base: np.log2 or np.log10
    else:
        s['s_ij'] = np.log(s['O_spoke'].div(s['E_spoke']))
    s['Lambda_ij'] = s['O_spoke']*s['s_ij']
    ####### Previously, in the function compute_spoke_ij_conf_weighted_v2 I implemented a change for s['Lambda_ij'], so that:
    ####### s['Lambda_ij'] = s['n']*s['f_i_bait']*s['s_ij']
    # Everything below implements a CI for the expectation in the denominator. binomial where possible, everywhere else based on normality assumption and the SE
    if do_conf_intervals:
        s['SE'] = np.sqrt((s.iloc[:,4:6].prod(axis=1)).multiply(total_purifications)).multiply(s.iloc[:,3].multiply(1-s.iloc[:,3]))
        s['s_high'] = np.log(s['O_spoke'].div(binom.ppf(0.05, (s.iloc[:,4:6].prod(axis=1)).multiply(total_purifications), s.iloc[:,3])))
        s['s_low'] = np.log(s['O_spoke'].div(binom.ppf(0.95, (s.iloc[:,4:6].prod(axis=1)).multiply(total_purifications), s.iloc[:,3])))
        s.replace([np.inf, -np.inf], np.nan, inplace=True)
        s['s_high'] = s['s_high'].fillna(np.log(s['O_spoke'].div((s.iloc[:,3:6].prod(axis=1)).multiply(total_purifications)-1.64*s['SE'])))
        s['s_low'] = s['s_low'].fillna(np.log(s['O_spoke'].div((s.iloc[:,3:6].prod(axis=1)).multiply(total_purifications)+1.64*s['SE'])))
        s['Lambda_high'] = s['O_spoke']*s['s_high']
        s['Lambda_low'] = s['O_spoke']*s['s_low']
    return s, result

def add_to_A(small_s_of_interest, s, new_matr_list):
    ### II.2.1.i $n_{i,j}^{prey}$ = number of times that $i$ and $j$ are seen together in matrix
    # this is matr_list
    ### II.2.1.ii. $n_{prey}$ = number of preys observed with a particular bait (excluding itself)
    temp = s[['bait', 'n_i_prey']].drop_duplicates().rename(
        columns={'n_i_prey': 'preys'}).reset_index(drop=True)
    n_prey = pd.DataFrame({'bait': temp['bait'],
                           'term': (temp['preys'].mul(temp['preys'].subtract(1))).div(2)})
    total_sum = sum(n_prey['term'])
    ### II.2.1.iii. combining all matrix terms
    m = small_s_of_interest[['bait', 'prey', 'interactors']].rename(columns={'bait': 'prey_x',
                                                                             'prey': 'prey_y'})
    m['O_matrix'] = m['interactors'].map(new_matr_list)

    f_j_prey = small_s_of_interest[['prey', 'f_j_prey']].drop_duplicates(subset='prey', ignore_index=True)
    m = pd.merge(m, f_j_prey.rename(columns={'prey': 'prey_x', 'f_j_prey': 'f_x_prey'}), how="left", on="prey_x")
    m = pd.merge(m, f_j_prey.rename(columns={'prey': 'prey_y', 'f_j_prey': 'f_y_prey'}), how="left", on="prey_y")
    m = pd.merge(m, n_prey.rename(columns={'bait': 'prey_x'}), how="left", on="prey_x").rename(
        columns={'term': 'term_x'})
    m = pd.merge(m, n_prey.rename(columns={'bait': 'prey_y'}), how="left", on="prey_y").rename(
        columns={'term': 'term_y'})

    m[['term_x', 'term_y']] = m[['term_x', 'term_y']].fillna(0)
    m['binomial'] = total_sum - (m['term_x'] + m['term_y'])
    m['E_matrix'] = m[['f_x_prey', 'f_y_prey', 'binomial']].prod(axis=1)

    m['m_ij'] = np.log(m['O_matrix'].div(m['E_matrix']))
    m['m_ij'] = m['m_ij'].replace(np.nan, 0)
    m['Lambda_m'] = m['O_matrix'] * m['m_ij']

    """
    if with_shrinkage:
        m['m_ij'] = np.log((m['O_matrix'] + shrink_param).div(m['E_matrix'] + shrink_param))
    else:
        m['m_ij'] = np.log(m['O_matrix'].div(m['E_matrix']))
    m['m_ij'] = m['m_ij'].replace(np.nan, 0)
    m['Lambda_m'] = m['O_matrix'] * m['m_ij']

    if do_conf_intervals:
        m['SEm'] = np.sqrt(np.sqrt(m['binomial']).multiply(m[['f_x_prey', 'f_y_prey']].prod(axis=1)))
        m['m_high'] = np.log(m['O_matrix'].div(binom.ppf(0.05, m['binomial'], m['f_x_prey'] * m['f_y_prey'])))
        m['m_low'] = np.log(m['O_matrix'].div(binom.ppf(0.95, m['binomial'], m['f_x_prey'] * m['f_y_prey'])))
        m.replace([np.inf, -np.inf], np.nan, inplace=True)
        m['m_high'] = m['m_high'].fillna(
            np.log(m['O_matrix'].div(m[['f_x_prey', 'f_y_prey', 'binomial']].prod(axis=1) - 1.64 * m['SEm'])))
        m['m_low'] = m['m_low'].fillna(
            np.log(m['O_matrix'].div(m[['f_x_prey', 'f_y_prey', 'binomial']].prod(axis=1) + 1.64 * m['SEm'])))
        m['Lambda_m_high'] = m['O_matrix'] * m['m_high']
        m['Lambda_m_low'] = m['O_matrix'] * m['m_low']
    """

    ## Combine all to get final SA score
    """
    if do_conf_intervals:
        s_ij = pd.DataFrame(
            s[['bait', 'prey', 's_ij', 'Lambda_ij', 'O_spoke', 'E_spoke', 'Lambda_low', 'Lambda_high']]).rename(
            columns={'Lambda_ij':
                         's_ij_lrs', 'Lambda_low': 'Lambda_ij_low', 'Lambda_high': 'Lambda_ij_high'})
    else:
        s_ij = pd.DataFrame(s[['bait', 'prey', 's_ij', 'Lambda_ij', 'O_spoke', 'E_spoke']]).rename(
            columns={'Lambda_ij': 's_ij_lrs'})
    """
    s_ij = pd.DataFrame(small_s_of_interest).rename(columns={'Lambda_ij': 's_ij_lrs'}) #[['bait', 'prey', 's_ij', 'Lambda_ij', 'O_spoke', 'E_spoke','interactors']]
    #s_ij['interactors'] = s_ij.apply(lambda row: ';'.join(sorted([row['bait'], row['prey']], reverse=True)), axis=1)
    A = pd.DataFrame(columns=['protein_x', 'protein_y', 'WeSA', 'SA', 'O_spoke_x', 'O_spoke_y', 'O_matrix'])
    if (np.shape(s_ij)[0] > 0):
        s_ij['ij_tru'] = s_ij.apply(lambda row: ';'.join([row['bait'], row['prey']]), axis=1)
        s_ij['ij_rev'] = s_ij.apply(lambda row: ';'.join([row['prey'], row['bait']]), axis=1)
        """
        if do_conf_intervals:
            s_ji = s_ij[
                ['bait', 'prey', 's_ij', 's_ij_lrs', 'O_spoke', 'E_spoke', 'ij_rev', 'Lambda_ij_high', 'Lambda_ij_low',
                 'interactors']].rename(
                columns={'ij_rev': 'ij_tru', 's_ij': 's_ji', 's_ij_lrs': 's_ji_lrs', 'Lambda_ij_high': 'Lambda_ji_high',
                         'Lambda_ij_low': 'Lambda_ji_low'})
            m_ij = pd.DataFrame(
                m[['interactors', 'm_ij', 'Lambda_m', 'O_matrix', 'E_matrix', 'Lambda_m_low', 'Lambda_m_high']].rename(
                    columns={'Lambda_m': 'm_ij_lrs'}))
            A = pd.merge(s_ij[['bait', 'prey', 's_ij', 's_ij_lrs', 'O_spoke', 'E_spoke', 'interactors', 'ij_tru',
                               'Lambda_ij_low', 'Lambda_ij_high']], s_ji, how="outer", on="ij_tru")
        else:
            s_ji = s_ij[['bait', 'prey', 's_ij', 's_ij_lrs', 'O_spoke', 'E_spoke', 'ij_rev', 'interactors']].rename(
                columns={'ij_rev': 'ij_tru', 's_ij': 's_ji', 's_ij_lrs': 's_ji_lrs'})
            m_ij = pd.DataFrame(
                m[['interactors', 'm_ij', 'Lambda_m', 'O_matrix', 'E_matrix']].rename(columns={'Lambda_m': 'm_ij_lrs'}))
            A = pd.merge(s_ij[['bait', 'prey', 's_ij', 's_ij_lrs', 'O_spoke', 'E_spoke', 'interactors', 'ij_tru']], s_ji,
                         how="outer", on="ij_tru")
        print('Number of bait-prey pairs for which a SA score is calculated = ',
              f'{np.shape(A)[0]:,}')  # Pairs from only the matrix are ignored
        A['interactors_x'] = A['interactors_x'].fillna(A['interactors_y'])
        A[['bait_x', 'prey_x']] = A[['bait_x', 'prey_x']].fillna(
            A[['bait_y', 'prey_y']].rename(columns={'bait_y': 'bait_x', 'prey_y': 'prey_x'}))
    
        if do_conf_intervals:
            A = pd.merge(A[['bait_x', 'prey_x', 'O_spoke_x', 'O_spoke_y', 'E_spoke_x', 'E_spoke_y', 'interactors_x', 's_ij',
                            's_ji', 's_ij_lrs', 's_ji_lrs', 'Lambda_ij_low', 'Lambda_ji_low', 'Lambda_ij_high',
                            'Lambda_ji_high']].rename(
                columns={'interactors_x': 'interactors', 'bait_x': 'bait', 'prey_x': 'prey'}), m_ij, how="left",
                         on="interactors")  # .drop_duplicates(subset = 'interactors', ignore_index=True)
        else:
            A = pd.merge(A[['bait_x', 'prey_x', 'O_spoke_x', 'O_spoke_y', 'E_spoke_x', 'E_spoke_y', 'interactors_x', 's_ij',
                            's_ji', 's_ij_lrs', 's_ji_lrs']].rename(
                columns={'interactors_x': 'interactors', 'bait_x': 'bait', 'prey_x': 'prey'}), m_ij, how="left",
                         on="interactors")  # .drop_duplicates(subset = 'interactors', ignore_index=True)
        """

        s_ji = s_ij.drop(columns='ij_tru').rename(columns={'ij_rev': 'ij_tru', 's_ij': 's_ji', 's_ij_lrs': 's_ji_lrs'}) # [['bait', 'prey', 's_ij', 's_ij_lrs', 'O_spoke', 'E_spoke', 'ij_rev', 'interactors']]
        m_ij = pd.DataFrame(m.rename( #[['interactors', 'm_ij', 'Lambda_m', 'O_matrix', 'E_matrix']]
            columns={'Lambda_m': 'm_ij_lrs'}))
        A = pd.merge(s_ij, s_ji, how="outer", on="ij_tru") ##[['bait', 'prey', 's_ij', 's_ij_lrs', 'O_spoke', 'E_spoke', 'interactors', 'ij_tru']]
        A['interactors_x'] = A['interactors_x'].fillna(A['interactors_y'])
        A[['bait_x', 'prey_x']] = A[['bait_x', 'prey_x']].fillna(
            A[['bait_y', 'prey_y']].rename(columns={'bait_y': 'bait_x', 'prey_y': 'prey_x'}))
        A = pd.merge(A.rename( #[['bait_x', 'prey_x', 'O_spoke_x', 'O_spoke_y', 'E_spoke_x', 'E_spoke_y','interactors_x', 's_ij','s_ji', 's_ij_lrs', 's_ji_lrs']]
            columns={'interactors_x': 'interactors', 'bait_x': 'bait', 'prey_x': 'prey'}), m_ij,
            how="left", on="interactors") #.drop_duplicates(subset = 'interactors', ignore_index=True)

        A = A.drop_duplicates(subset=['interactors'])
        A['a_ij'] = A[['s_ij', 's_ji', 'm_ij']].sum(axis=1)
        A['Lambda_ij'] = A[['s_ij_lrs', 's_ji_lrs', 'm_ij_lrs']].sum(axis=1)
        """
        if do_conf_intervals:
            A['Lambda_ij_low'] = A['Lambda_ij_low'].fillna(2 * A['s_ij_lrs'] - A['Lambda_ij_high'])
            A['Lambda_ij_high'] = A['Lambda_ij_high'].fillna(2 * A['s_ij_lrs'] - A['Lambda_ij_low'])
            A['Lambda_ji_low'] = A['Lambda_ji_low'].fillna(2 * A['s_ji_lrs'] - A['Lambda_ji_high'])
            A['Lambda_ji_high'] = A['Lambda_ji_high'].fillna(2 * A['s_ji_lrs'] - A['Lambda_ji_low'])
            A['Lambda_m_low'] = A['Lambda_m_low'].fillna(2 * A['m_ij_lrs'] - A['Lambda_m_high'])
            A['Lambda_m_high'] = A['Lambda_m_high'].fillna(2 * A['m_ij_lrs'] - A['Lambda_m_low'])
            A['Lambda_low'] = A[['Lambda_ij_low', 'Lambda_ji_low', 'Lambda_m_low']].sum(axis=1)
            A['Lambda_high'] = A[['Lambda_ij_high', 'Lambda_ji_high', 'Lambda_m_high']].sum(axis=1)
        """
        A = A.fillna(0)
        A = A.sort_values('a_ij', ascending=False, ignore_index=True)
        """if do_conf_intervals:
            A = A[['bait', 'prey', 'a_ij', 'Lambda_ij', 'Lambda_low', 'Lambda_high'] + 
                  [c for c in A if c not in ['bait','prey','a_ij','Lambda_low','Lambda_ij','Lambda_high']]]
        else:
        """
        #A = A[['bait', 'prey', 'a_ij', 'Lambda_ij'] + [c for c in A if c not in ['bait', 'prey', 'a_ij',
        #                                                                             'Lambda_ij']]]  # , 'O_spoke_x', 'E_spoke_x', 'O_spoke_y', 'E_spoke_y', 'O_matrix', 'E_matrix'

        #total_purifications = np.shape(data.drop_duplicates(subset=['identifier', 'bait']))[0]  # i.e. # of unique experiments (by ID), meaning, repeated baits sometimes (= n_bait)
        # A[['s_ij_lrs','s_ji_lrs']] = A[['s_ij_lrs','s_ji_lrs']]*total_purifications
        # A['Lambda_ij'] = A[['s_ij_lrs','s_ji_lrs', 'm_ij_lrs']].sum(axis=1)

        A.replace([np.inf, -np.inf], np.nan, inplace=True)
        A = A.dropna(subset=["a_ij", "Lambda_ij"], how="all")
        A = A.fillna(0)

        A = A[['bait', 'prey', 'Lambda_ij', 'a_ij', 'O_spoke_x', 'O_spoke_y', 'O_matrix', 'E_spoke_x',
               'E_spoke_y', 'E_matrix', 's_ij_lrs', 's_ji_lrs', 'm_ij_lrs', 's_ij', 's_ji', 'm_ij',
               'f_i_prey_x', 'f_i_bait_x', 'n_i_prey_x', 'n_i_bait_x', 'n_j_prey_x','f_i_prey_y',
               'f_i_bait_y', 'n_i_prey_y', 'n_i_bait_y', 'n_j_prey_y','all_pairs_x', 'tot_pur_x', 'f_x_prey',
               'f_y_prey', 'term_x', 'term_y', 'binomial', 'interactors']].rename(
            columns={'bait':'protein_x', 'prey':'protein_y','tot_pur_x':'tot_pur', 'Lambda_ij': 'WeSA',
                     'a_ij': 'SA','m_ij':'m', 'all_pairs_x': 'all_pairs', 'm_ij_lrs':'m_lrs'}).sort_values(by='WeSA', ascending=False)#.to_csv(fn_path_to_A, sep='\t', index=False)
    return A


def create_A_dict_v(data, fn_path_to_A, use_conf=False, fn_matr_dict = {}, fn_matr_list = {}, matrix_available=True,
                    matr_from_file = False, matr_dict_name='mydict_matrix.pkl', matr_list_name='matr_list.pkl',
                    save_data = False, with_shrinkage=False, shrink_param=0.5, scale_weight=True):
    # A is created using a dictionary for the matrix
    # if matrix_available, two files should exist with names matr_dict_name and matr_list_name
    # if matrix_available = False, the function creates the matrix from scratch and saves the relevant two files with names:
    # 'fn'+matr_dict_name and 'fn'+matr_list_name
    #starttime = timeit.default_timer()
    s, result0 = compute_spoke_ij_conf_weighted(data, use_conf=use_conf, with_shrinkage=with_shrinkage,
                                                shrink_param=shrink_param)
    s = remove_self_loops(s, 'bait', 'prey')
    #print("The intermediate time difference (spoke terms ready) is: ", timeit.default_timer() - starttime)
    data1 = data[['prey', 'identifier']].drop_duplicates(ignore_index=True)

    if matrix_available:
        if matr_from_file:
            with open(matr_dict_name, 'rb') as file:
                mydict = pickle.load(file)
            with open(matr_list_name, 'rb') as file:
                matr_list = pickle.load(file)
        else:
            mydict = fn_matr_dict
            matr_list = fn_matr_list
    else:
        print('Calculating matrix dictionary...')
        mydict = create_prey_id_dict(data1)
        if save_data:
            with open(matr_dict_name, 'wb') as file:
                pickle.dump(mydict, file, pickle.HIGHEST_PROTOCOL)
        #print("The intermediate time difference (prey dictionary ready) is: ", timeit.default_timer() - starttime)
        print('Compiling matrix pairs list...')
        matr_list = create_matr_list(s, mydict)
        if save_data:
            with open(matr_list_name, 'wb') as file:
                pickle.dump(matr_list, file, pickle.HIGHEST_PROTOCOL)
        #print("The intermediate time difference (matrix pairs ready) is: ", timeit.default_timer() - starttime)

    ### II.2.1.ii. $n_{prey}$ = number of preys observed with a particular bait (excluding itself)
    print('Calculating n_prey...')
    result = result0[['bait', 'identifier', 'n_i_prey']].rename(
        columns={'identifier': 'interaction ID', 'n_i_prey': 'preys'})
    n_id_prey = result.groupby(['bait', 'interaction ID']).sum()
    n_id_prey.reset_index(level=[0, 1], inplace=True)

    n_prey = pd.DataFrame(
        {'bait': n_id_prey['bait'], 'term': (n_id_prey['preys'].mul(n_id_prey['preys'].subtract(1))).div(2)})
    n_prey = n_prey.groupby(['bait']).sum()
    n_prey.reset_index(level=0, inplace=True)

    total_sum = sum(n_prey['term'])

    total_purifications = np.shape(data.drop_duplicates(subset=['identifier', 'bait']))[0]

    ### II.2.1.iii. combining all matrix terms
    print('Combining matrix terms...')
    m = s[['bait', 'prey']].rename(columns={'bait': 'prey_x', 'prey': 'prey_y'}).copy()

    #### $n_{i,j}^{prey}$ = number of times that $i$ and $j$ are seen together in matrix
    m['interactors'] = m.apply(lambda row: ';'.join(sorted([row['prey_x'], row['prey_y']], reverse=True)), axis=1)
    m['O_matrix'] = m['interactors'].map(matr_list)
    ####

    f_j_prey = s[['prey', 'f_j_prey']].drop_duplicates(subset='prey', ignore_index=True)
    m = pd.merge(m, f_j_prey.rename(columns={'prey': 'prey_x', 'f_j_prey': 'f_x_prey'}), how="left", on="prey_x")
    m = pd.merge(m, f_j_prey.rename(columns={'prey': 'prey_y', 'f_j_prey': 'f_y_prey'}), how="left", on="prey_y")
    m = pd.merge(m, n_prey.rename(columns={'bait': 'prey_x'}), how="left", on="prey_x").rename(
        columns={'term': 'term_x'})
    m = pd.merge(m, n_prey.rename(columns={'bait': 'prey_y'}), how="left", on="prey_y").rename(
        columns={'term': 'term_y'})

    m[['term_x', 'term_y']] = m[['term_x', 'term_y']].fillna(0)
    m['binomial'] = total_sum - (m['term_x'] + m['term_y'])
    m['E_matrix'] = m[['f_x_prey', 'f_y_prey', 'binomial']].prod(axis=1)

    if with_shrinkage:
        m['m_ij'] = np.log((m['O_matrix'] + shrink_param).div(m['E_matrix'] + shrink_param))
    else:
        m['m_ij'] = np.log(m['O_matrix'].div(m['E_matrix']))
    m['m_ij'] = m['m_ij'].replace(np.nan, 0)

    m_weight = 1
    if scale_weight:
        E_Om = sum(matr_list.values()) / len(matr_list.values())  # np.mean(m['O_matrix']) #new
        E_Os = np.mean(s['O_spoke'])  # new
        m_weight = E_Os / E_Om  # new

    m['Lambda_m'] = (m_weight) * m['O_matrix'] * m['m_ij']  ### edited

    ## Combine all to get final SA score
    print('Combining all terms...')
    s_ij = pd.DataFrame(s[['bait', 'prey', 's_ij', 'Lambda_ij', 'O_spoke', 'E_spoke']]).rename(
        columns={'Lambda_ij': 's_ij_lrs'})
    s_ij['interactors'] = s_ij.apply(lambda row: ';'.join(sorted([row['bait'], row['prey']], reverse=True)), axis=1)
    s_ij['ij_tru'] = s_ij.apply(lambda row: ';'.join([row['bait'], row['prey']]), axis=1)
    s_ij['ij_rev'] = s_ij.apply(lambda row: ';'.join([row['prey'], row['bait']]), axis=1)

    s_ji = s_ij[['bait', 'prey', 's_ij', 's_ij_lrs', 'O_spoke', 'E_spoke', 'ij_rev', 'interactors']].rename(
        columns={'ij_rev': 'ij_tru', 's_ij': 's_ji', 's_ij_lrs': 's_ji_lrs'})
    m_ij = pd.DataFrame(
        m[['interactors', 'm_ij', 'Lambda_m', 'O_matrix', 'E_matrix']].rename(columns={'Lambda_m': 'm_ij_lrs'}))
    A = pd.merge(s_ij[['bait', 'prey', 's_ij', 's_ij_lrs', 'O_spoke', 'E_spoke', 'interactors', 'ij_tru']], s_ji,
                 how="outer", on="ij_tru")
    print('Number of bait-prey pairs for which a SA score is calculated = ',
          f'{np.shape(A)[0]:,}')  # Pairs from only the matrix are ignored
    A['interactors_x'] = A['interactors_x'].fillna(A['interactors_y'])
    A[['bait_x', 'prey_x']] = A[['bait_x', 'prey_x']].fillna(
        A[['bait_y', 'prey_y']].rename(columns={'bait_y': 'bait_x', 'prey_y': 'prey_x'}))

    A = pd.merge(A[['bait_x', 'prey_x', 'O_spoke_x', 'O_spoke_y', 'E_spoke_x', 'E_spoke_y', 'interactors_x', 's_ij',
                    's_ji', 's_ij_lrs', 's_ji_lrs']].rename(
        columns={'interactors_x': 'interactors', 'bait_x': 'bait', 'prey_x': 'prey'}), m_ij, how="left",
                 on="interactors")  # .drop_duplicates(subset = 'interactors', ignore_index=True)
    A = A.drop_duplicates(subset=['interactors'])
    A['a_ij'] = A[['s_ij', 's_ji', 'm_ij']].sum(axis=1)
    A['Lambda_ij'] = A[['s_ij_lrs', 's_ji_lrs', 'm_ij_lrs']].sum(axis=1)
    A = A.fillna(0)
    A = A.sort_values('a_ij', ascending=False, ignore_index=True)
    A = A[['bait', 'prey', 'a_ij', 'Lambda_ij'] + [c for c in A if c not in ['bait', 'prey', 'a_ij',
                                                                             'Lambda_ij']]]  # , 'O_spoke_x', 'E_spoke_x', 'O_spoke_y', 'E_spoke_y', 'O_matrix', 'E_matrix'

    A.replace([np.inf, -np.inf], np.nan, inplace=True)
    A = A.dropna(subset=["a_ij", "Lambda_ij"], how="all")
    A = A.fillna(0)

    #print("The intermediate time difference (after putting everything together) is: ", timeit.default_timer() - starttime)

    if save_data:
        print('Saving A...')
        A.sort_values(by='Lambda_ij', ascending=False).to_csv(fn_path_to_A, sep='\t', index=False)

    #print("Function finished. Total time: ", timeit.default_timer() - starttime)
    return A

def create_A_db(A):
    """Generate db."""
    for i in range(np.shape(A)[0]):
        item = Item(protein_x=A.protein_x[i],
                    protein_y=A.protein_y[i],
                    WeSA=A.WeSA[i],
                    SA=A.SA[i],
                    O_spoke_x=A.O_spoke_x[i],
                    O_spoke_y=A.O_spoke_y[i],
                    O_matrix=A.O_matrix[i])
        db.session.add(item)
    db.session.commit()

def replace_names(file, return_table):
    file = file.replace({'gene': r'\s.*'}, {'gene': ''}, regex=True).drop_duplicates()
    di = dict(zip(file['original'], file['gene']))
    return_table.loc[:, 'protein A'] = return_table['protein A'].map(di).fillna(return_table['protein A'])
    return_table.loc[:, 'protein B'] = return_table['protein B'].map(di).fillna(return_table['protein B'])
    return return_table


def remove_self_loops(dataframe, col1, col2):
    # col1 and col2 are strings
    self_baited = (dataframe[col1] == dataframe[col2])
    self_bait_ind = list(compress(range(len(self_baited)), self_baited))
    dataframe = dataframe.drop(dataframe.index[self_bait_ind])
    dataframe = dataframe.reset_index(drop=True)
    return dataframe


# if delete = False, imputing confidence scores; else - delete nan and 0 confidence
# input: data with column 'conf'
def impute_delete_na_conf(data, delete=True):
    if delete:
        data = data[data['conf'] > 0]
    else:
        l = len(data.loc[data['conf'] == 0, 'conf'])
        data.loc[data['conf'] == 0, 'conf'] = random.choices(list(data.loc[data['conf'] > 0, 'conf']), k=l)
    return data

# Compiling the matrix table from the spokes data
# Input should contain the columns: bait, prey and identifier
def compile_matrix_from_spoke(data, bool_self_baited):
    if bool_self_baited == True:
        self_baited = (data['bait'] == data['prey'])
        self_bait_ind = list(compress(range(len(self_baited)), self_baited))
        non_self_baits = data.drop(data.index[self_bait_ind])
        prey_and_id = non_self_baits[['bait', 'prey', 'identifier']]  # .drop_duplicates(ignore_index = True)
    else:
        prey_and_id = data[['bait', 'prey', 'identifier']]
    matrix = pd.merge(prey_and_id[['bait', 'interaction ID']].drop_duplicates(),
                      prey_and_id[['prey', 'interaction ID']].drop_duplicates(), how="inner",
                      on=['identifier', 'bait']).drop_duplicates(ignore_index=True)
    self_loops = (matrix['prey_x'] == matrix['prey_y'])
    self_loops_ind = list(compress(range(len(self_loops)), self_loops))
    matrix = matrix.drop(matrix.index[self_loops_ind])  # .iloc[:,[0,3]]
    matrix['interactors'] = matrix.apply(lambda row: ';'.join(sorted([row['prey_x'], row['prey_y']], reverse=True)),
                                         axis=1)
    matrix = matrix.drop_duplicates(subset=['bait', 'identifier', 'interactors'], ignore_index=True)
    return matrix


# Functions supporting matrix creation
def create_prey_id_dict(data):  # data should contain cols 'prey' and 'identifier'
    # records for each prey, its identifiers {prey:[identifiers]}
    mydict = {}
    step = 0
    for i in data['prey'].unique():  # len(data['prey'].unique()) = 12,186
        if step % 2000 == 0:
            print(step)
        mydict[i] = [data['identifier'][j] for j in data[data['prey'] == i].index]
        step += 1
    return mydict

def create_matr_list(data, prey_id_dict):  # data = s (with bait and prey columns to get the matrix from)
    # creates a dict {['prot1;prot2']:count} of matrix pairs (which have spoke observations) and their obs. numbers
    matr_list = {}
    for i in range(np.shape(data)[0]):
        if i % 300000 == 0:
            print(i)
        if data.loc[i, 'bait'] in prey_id_dict:
            matr_list[';'.join(sorted([data.loc[i, 'bait'], data.loc[i, 'prey']], reverse=True))] = len(
                list(set(prey_id_dict[data.loc[i, 'bait']]).intersection(prey_id_dict[data.loc[i, 'prey']])))
    return matr_list

# Plotting histogram of the scores. Lambda is always plotted. There are also the options for plotting double-Lambda & SA
# as well as curves of the chi-sq. distribution.
def sa_hist(bool_doubled, mydata, hist_range_min, hist_range_max, bool_chi_lines, chi_lines_df_list, bool_plot_sa,
            bool_density):
    plt.hist(mydata['Lambda_ij'], bins=range(hist_range_min, hist_range_max), log=False, density=bool_density,
             alpha=0.6, rwidth=1, label='$\Lambda$ / 2')
    if bool_doubled == True:
        plt.hist(2 * mydata['Lambda_ij'], bins=range(hist_range_min, hist_range_max), log=False, density=bool_density,
                 alpha=0.6, rwidth=1, label='$\Lambda$')
    x = np.arange(0.001, 100, 0.001)  # x-axis ranges from 0 to 100 with .001 steps
    if bool_chi_lines == True:
        for i in chi_lines_df_list:
            plt.plot(x, chi2.pdf(x, df=i), label=''.join([str(i), ' df']))
    if bool_plot_sa == True:
        plt.hist(mydata['s_ij'], bins=range(hist_range_min, hist_range_max), log=False, density=bool_density, alpha=0.4,
                 rwidth=1, label='SA')
    plt.legend(loc='upper right')
    plt.xlim((hist_range_min, hist_range_max))

# Input: score_str is 'Lambda_ij' or 'a_ij' (or any other column containing the score of interest)
#        gs_data contains a column 'interactors' and another (holder) column, which should be called 'ComplexName'
#        full_data should contain the columns 'bait', 'prey', 'interactors' and the score from score_str
# the default for full_data should be A but can't be defined if not loaded here
def compute_stat_rates_manual(score_str, gs_data, full_data):
    # roc_fun = pd.merge(A_of_interest[['bait','prey','a_ij','Lambda_low','Lambda_ij','Lambda_high','interactors']],gold_standard_data[['ComplexName','interactors']], on = 'interactors', how = 'left').fillna(0).rename(columns={'ComplexName':'group'})
    roc_fun = pd.merge(full_data[['bait', 'prey', score_str, 'interactors']], gs_data[['ComplexName', 'interactors']],
                       on='interactors', how='left').fillna(0).rename(columns={'ComplexName': 'group'})
    roc_fun = roc_fun[~(roc_fun['bait'] == roc_fun['prey'])]
    # print(roc_fun[:2])
    roc_fun.loc[roc_fun['group'] != 0, ['group']] = 1
    roc_fun = roc_fun.sort_values(by=score_str, ascending=False).reset_index(drop=True)

    temp = np.cumsum(roc_fun['group'])
    group1_tot = roc_fun['group'].sum()
    group0_tot = np.shape(roc_fun)[0] - group1_tot
    roc_fun['tpr'] = temp.div(group1_tot)
    roc_fun['fpr'] = np.subtract(roc_fun.index + 1, temp).div(group0_tot)
    roc_fun['precision'] = temp / (roc_fun.index + 1)
    return [roc_fun['fpr'], roc_fun['tpr'], roc_fun['precision'], roc_fun['tpr']]


# Recall seemed to not be equal to TPR for the function below (that is why I did the manual one)
def compute_stat_rates(score_str, gs_data, full_data):
    func_roc = pd.merge(full_data[['bait', 'prey', score_str, 'interactors']], gs_data[['ComplexName', 'interactors']],
                        on='interactors', how='left').fillna(0).rename(columns={'ComplexName': 'group'})
    func_roc = func_roc[~(func_roc['bait'] == func_roc['prey'])]
    func_roc.loc[func_roc['group'] != 0, ['group']] = 1
    func_fpr, func_tpr, thresholds = metrics.roc_curve(func_roc['group'].tolist(), func_roc[score_str].tolist(),
                                                       drop_intermediate=False)
    func_precision, func_recall, thresholds = metrics.precision_recall_curve(func_roc['group'].tolist(),
                                                                             func_roc[score_str].tolist())
    return [func_fpr, func_tpr, func_precision, func_recall]


# Calculates thresholds infered from the ROC curve
# if calc_all = True it calculates all 4 thresholds, otherwise - just the 0-1 closest point
def find_roc_thresholds(func_tpr, func_fpr, calc_all=False):
    roc_dists_to_0_1 = np.square(np.square(func_fpr) + np.square(1 - func_tpr))
    cut_off_ind_0_1_reworked = np.argmin(roc_dists_to_0_1)
    # print("At the cut-off minimising the distance between the ROC curve and (0,1):\n --> FPR is",
    #      round(func_fpr[cut_off_ind_0_1_reworked],3),"\t\t --> TPR is", round(func_tpr[cut_off_ind_0_1_reworked],3))
    l_cutoffs = cut_off_ind_0_1_reworked

    if calc_all:
        roc_dists_gmean = (1 - func_fpr) * func_tpr
        cut_off_ind_g_reworked = np.argmax(roc_dists_gmean)
        # print("At the cut-off maximising the geometric mean of sensitivity and specificity:\n --> FPR is",
        #      round(func_fpr[cut_off_ind_g_reworked],3), "\t\t --> TPR is", round(func_tpr[cut_off_ind_g_reworked],3))

        cut_off_ind_j_reworked = np.argmax(func_tpr - func_fpr)
        # print("At the cut-off maximising the statistic Youden's J statistic:\n --> FPR is",
        #      round(func_fpr[cut_off_ind_j_reworked],3),"\t\t --> TPR is", round(func_tpr[cut_off_ind_j_reworked],3))

        # This measure is from the paper cited in v2 from 12Aug
        auc_reworked = metrics.auc(func_fpr, func_tpr)
        roc_dists_iu = abs(func_tpr - auc_reworked) + abs(1 - func_tpr - auc_reworked)
        unique_dists_iu = (roc_dists_iu.drop_duplicates())
        cut_off_ind_iu_sup = np.where(roc_dists_iu <= sorted(unique_dists_iu)[3])[0].reshape(-1)
        se_sp_dist = abs(func_tpr - (1 - func_fpr))
        cut_off_ind_iu_reworked = cut_off_ind_iu_sup[0] + np.argmin([se_sp_dist[cut_off_ind_iu_sup]])
        # print("IUs considered:", sorted(unique_dists_iu)[:3], "\tThreshold IU:",sorted(unique_dists_iu)[3])
        l_cutoffs = [cut_off_ind_0_1_reworked, cut_off_ind_g_reworked, cut_off_ind_j_reworked, cut_off_ind_iu_reworked]
    return l_cutoffs


# Default that can't be set here: gold_standard_data = cx_gs
def make_roc_table_and_draw(A_of_interest, ax1, ax2, mycond, gold_standard_data):
    # roc_fun = pd.merge(A_of_interest[['bait','prey','a_ij','Lambda_low','Lambda_ij','Lambda_high','interactors']],gold_standard_data[['ComplexName','interactors']], on = 'interactors', how = 'left').fillna(0).rename(columns={'ComplexName':'group'})
    roc_fun = pd.merge(A_of_interest, gold_standard_data[['ComplexName', 'interactors']], on='interactors',
                       how='left').fillna(0).rename(columns={'ComplexName': 'group'})
    roc_fun = roc_fun[~(roc_fun['bait'] == roc_fun['prey'])]
    roc_fun.loc[roc_fun['group'] != 0, ['group']] = 1
    roc_fun = roc_fun.sort_values(by='Lambda_ij', ascending=False).reset_index(drop=True)

    temp = np.cumsum(roc_fun['group'])
    group1_tot = roc_fun['group'].sum()
    group0_tot = np.shape(roc_fun)[0] - group1_tot
    roc_fun['tpr'] = temp.div(group1_tot)
    roc_fun['fpr'] = np.subtract(roc_fun.index + 1, temp).div(group0_tot)
    roc_fun['precision'] = temp / (roc_fun.index + 1)

    ax1.plot(roc_fun['fpr'], roc_fun['tpr'], label=mycond)
    ax1.set_xlabel('FPR (1 - specificity)')
    ax1.set_ylabel('TPR (sensitivity)')

    ax2.plot(roc_fun['tpr'], roc_fun['precision'])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    fun_stats = [mycond, metrics.auc(roc_fun['fpr'], roc_fun['tpr']), metrics.auc(roc_fun['tpr'], roc_fun['precision'])]
    return fun_stats


def draw_roc(ax1, ax2, mycond, fpr, tpr, recall, precision, linetype='-', color_given=False, color_str='black'):
    # roc_fun = pd.merge(A_of_interest[['bait','prey','a_ij','Lambda_low','Lambda_ij','Lambda_high','interactors']],gold_standard_data[['ComplexName','interactors']], on = 'interactors', how = 'left').fillna(0).rename(columns={'ComplexName':'group'})
    if color_given:
        ax1.plot(fpr, tpr, linetype, linewidth=3, label=mycond, color=color_str)
        ax2.plot(recall, precision, linetype, linewidth=3, label=mycond, color=color_str)
    else:
        ax1.plot(fpr, tpr, linetype, linewidth=3, label=mycond)
        ax2.plot(recall, precision, linetype, linewidth=3, label=mycond)
    ax1.set_xlabel('FPR (1 - specificity)')
    ax1.set_ylabel('TPR (sensitivity)')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    fun_stats = [mycond, metrics.auc(fpr, tpr), metrics.auc(recall, precision)]
    return fun_stats
