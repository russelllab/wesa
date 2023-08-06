import flask
import pandas as pd
import numpy as np
from flask import request
import functions as f
import json
import pickle

path_to_data = 'C:/Users/magda/Work projects/SA/results-after-RECOMB/'

APP = flask.Flask(__name__)

columns = [
  {
    "field": "protein_x", # which is the field's name of data key
    "title": "protein X", # display as the table header's name
    "sortable": True,
  },
  {
    "field": "protein_y",
    "title": "protein Y",
    "sortable": True,
  },
  {
    "field": "WeSA",
    "title": "WeSA",
    "sortable": True,
  },
  {
    "field": "SA",
    "title": "SA",
    "sortable": True,
  },
    {
    "field": "O_spoke_x",
    "title": "Observed X -- Y",
    "sortable": True,
  },
    {
    "field": "O_spoke_y",
    "title": "Observed Y -- X",
    "sortable": True,
  },
    {
    "field": "O_matrix",
    "title": "Observed in matrix",
    "sortable": True,
  }
]

@APP.route('/')
def index():
    return flask.render_template('index.html')

@APP.route('/output', methods = ["POST","GET"])
def output():
    if request.method == "POST":
        input1 = request.form["prots_input"]
        source_str = request.form["DataSource"]
        input2 = request.files["file_prots"]
        if input2.filename == '':
            input = input1
        else:
            input = input2.read().decode("utf-8")
        ### IDENTIFY BACKGROUND DATA
        bg_name = request.form['DataSource']
        bg = pd.read_csv('C:/Users/magda/Work projects/webtool/data/'+bg_name+'.txt', sep = '\t')
        with open(path_to_data + 'article_mydict_matrix_' + bg_name + '.pkl', 'rb') as file:
            mydict_bg = pickle.load(file)
        with open(path_to_data + 'article_matr_list_' + bg_name + '.pkl', 'rb') as file:
            matr_list_bg = pickle.load(file)

        ### FORMAT INPUT
        input_list = ('{0}'.format('{0}'.format(input.replace(' ', ';')).replace('\r', ''))).split('\n')
        myinput = pd.DataFrame(input_list)[0].str.split(';', expand = True).rename(columns={0:'bait', 1:'prey'})
        if (np.shape(myinput)[1] > 1):
            if(np.shape(myinput)[1]==2):
                myinput['identifier'] = 'new'
            else:
                myinput = myinput.rename(columns={2:'identifier'})
            queried_interactions = myinput.apply(lambda row: ';'.join(sorted([str(row['bait']), str(row['prey'])], reverse = True)), axis=1)
            # Matrices for input
            s, result0 = f.compute_spoke_ij_conf_weighted(myinput, use_conf=False, with_shrinkage=False, shrink_param=1)
            s = f.remove_self_loops(s, 'bait', 'prey')
            data1 = myinput[['prey', 'identifier']].drop_duplicates(ignore_index=True)
            mydict1 = f.create_prey_id_dict(data1)
            matr_list1 = f.create_matr_list(s, mydict1)
            # Update matrix
            mydict = f.merge_matr_dicts(mydict_bg, mydict1)
            matr_list = f.merge_matr_dicts(matr_list_bg, matr_list1)
            # Calculate everything
            df = pd.concat([myinput, bg[['bait', 'prey', 'identifier']]])
            A = f.create_A_dict_v(df, path_to_data, fn_matr_dict = mydict, fn_matr_list = matr_list)
            A = A[A['interactors'].isin(queried_interactions)].rename(
                columns={'bait':'protein_x', 'prey':'protein_y', 'tot_pur_x':'tot_pur', 'Lambda_ij': 'WeSA',
                         'a_ij': 'SA','m_ij':'m', 'm_ij_lrs':'m_lrs'})
        else: # the case where we just request to see old records for protein of interest
            bg = pd.read_csv(path_to_data+'A_article_'+bg_name+'_'+'unweighted'+'.tsv', sep = '\t')
            A = bg[(bg.bait.isin(list(myinput.iloc[:,0]))) | (bg.prey.isin(list(myinput.iloc[:,0])))].rename(
                columns={'bait':'protein_x', 'prey':'protein_y', 'tot_pur_x':'tot_pur', 'Lambda_ij': 'WeSA',
                         'a_ij': 'SA','m_ij':'m', 'm_ij_lrs':'m_lrs'})#.drop(columns=['a_ij_not_obs','Lambda_ij_not_obs'])
        A = A[['protein_x', 'protein_y', 'WeSA', 'SA', 'O_spoke_x', 'O_spoke_y', 'O_matrix']] #
        output = json.loads(A.to_json(orient='split'))
        with open('C:/Users/magda/Work projects/webtool_v2/static/myresults.json', 'wt') as file_out:
            json.dump(output, file_out)
        data = output["data"]
        columns = output["columns"]
    return flask.render_template('results.html', ints_json= "myresults.json", data=data, columns=columns, title='Your scored results') #flask.render_template('output.html', out=output)

if __name__ == '__main__':
    APP.debug = True
    APP.run()