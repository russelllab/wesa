import sys
import os
import re
import json
import time
import logging
import pandas as pd
import numpy as np
import pickle5 as pickle
from flask import Flask, request, render_template, redirect, url_for, jsonify
from celery import Celery
from dotenv import dotenv_values

from wesa_app import this_app
from . import functions as f

HERE_PATH = os.path.dirname(os.path.abspath(__file__))
config = dotenv_values(dotenv_path=f"{HERE_PATH}/.env")
path_to_data = config["DATA_PATH"]
node_info_file = f'{HERE_PATH}/static/node_info_corum.pkl'
output_dir = f'{HERE_PATH}/static/jobs/'
graph_outfile = 'mygraph.json'
ints_outfile = 'myresults.json'
# Templates
template_input_error = "input_error.html"

# Configure logging
logging.basicConfig(
    filename=f'{HERE_PATH}/log.txt',
    filemode="a",
    level=logging.INFO,
    format="[%(process)d] %(asctime)s - [%(levelname)s] %(message)s",
)

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

mycolumns = ['Protein A', 'Protein B', 'WeSA', 'SA', 'Count: A retrieved B', 'Count: B retrieved A', 'Count:matrix']

def make_celery(this_app):
    celery = Celery(this_app.import_name, broker=this_app.config['broker_url'])
    celery.conf.update(this_app.config)
    return celery

celery = make_celery(this_app)

@celery.task
def process_data(input, bg_name, job_dir):
    ### IDENTIFY BACKGROUND DATA
    bg = pd.read_csv(path_to_data + bg_name + '.txt', sep='\t')
    print('Background file:', path_to_data + bg_name + '.txt')
    with open(path_to_data + 'article_mydict_matrix_' + bg_name + '.pkl', 'rb') as file:
        mydict_bg = pickle.load(file)
    with open(path_to_data + 'article_matr_list_' + bg_name + '.pkl', 'rb') as file:
        matr_list_bg = pickle.load(file)

    ### THRESHOLDS FOR THE DATASETS
    db_vector = ['biogrid', 'bioplex', 'intact', 'intact_bioGrid', 'intact_bioPlex', 'bioGrid_bioPlex', 'all']
    # threshold_vector = [18.78, 31.82, 11.43, 73.86, 24.73, 51.08, 21.94]
    # db_threshold = threshold_vector[[j for j in range(len(db_vector)) if db_vector[j] == bg_name][0]]
    # new (all lines in this block) + changes in results and graph_features2
    threshold_vector = [24.64, 7.33, 25.57, 31.47, 10.99, 12.7,
                        14.77]  # [18.78, 31.82, 11.43, 73.86, 24.73, 51.08, 21.94]
    fpr1_threshold_vector = [64.66, 45.66, 84.61, 81.74, 94.87, 62.73, 67.53]
    fpr5_threshold_vector = [36.61, 28.44, 46.21, 49.92, 53.65, 36.67, 40.58]
    fpr10_threshold_vector = [29.02, 15.57, 33.02, 36.06, 29.21, 20.84, 25.0]
    fpr20_threshold_vector = [19.69, 8.97, 23.15, 25.43, 14.67, 11.41, 12.26]
    all_thresholds_matrix = pd.DataFrame(
        [threshold_vector, fpr1_threshold_vector, fpr5_threshold_vector, fpr10_threshold_vector,
         fpr20_threshold_vector])

    # new (next three lines)
    # db_threshold = threshold_vector[[j for j in range(len(db_vector)) if db_vector[j] == bg_name][0]]
    db_thresholds = all_thresholds_matrix[[j for j in range(len(db_vector)) if db_vector[j] == bg_name][0]]
    db_threshold = db_thresholds[0]

    ### FORMAT INPUT
    input_list = ('{0}'.format('{0}'.format(input.replace(' ', ';')).replace('\r', ''))).split('\n')
    input_list = [re.sub(';+', ';', re.sub(';*$', '', i)) for i in input_list]
    input_list = [i for i in filter(None, input_list)]

    myinput = pd.DataFrame(input_list)[0].str.split(';', expand=True).rename(columns={0: 'bait', 1: 'prey'})
    if np.shape(myinput)[1] > 1:
        if np.shape(myinput)[1] == 2:
            myinput['identifier'] = 'new'
        else:
            myinput = myinput.rename(columns={2: 'identifier'})

        queried_interactions = myinput.apply(
            lambda row: ';'.join(sorted([str(row['bait']), str(row['prey'])], reverse=True)), axis=1)

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
        A = f.create_A_dict_v(df, path_to_data, fn_matr_dict=mydict, fn_matr_list=matr_list)
        A = A[A['interactors'].isin(queried_interactions)].rename(
            columns={'bait': 'protein_x', 'prey': 'protein_y', 'tot_pur_x': 'tot_pur', 'Lambda_ij': 'WeSA',
                     'a_ij': 'SA', 'm_ij': 'm', 'm_ij_lrs': 'm_lrs'})
        unknown_interactions = list(
            queried_interactions[queried_interactions.isin(bg['interactors'].tolist()) == False])
        # unknown_interactions = []
        # for i in queried_interactions:
        #    if any(remove_non_printable(str(i)) in remove_non_printable(str(j).strip()) for j in bg['interactors'].tolist()):
        #        continue
        #    else:
        #        unknown_interactions = unknown_interactions + [str(i)]
        # print(unknown_interactions)

    else:  # the case where we just request to see old records for protein of interest
        bg = pd.read_csv(path_to_data + 'A_article_' + bg_name + '_' + 'unweighted' + '.tsv', sep='\t')
        A = bg[(bg.bait.isin(list(myinput.iloc[:, 0]))) | (bg.prey.isin(list(myinput.iloc[:, 0])))].rename(
            columns={'bait': 'protein_x', 'prey': 'protein_y', 'tot_pur_x': 'tot_pur', 'Lambda_ij': 'WeSA',
                     'a_ij': 'SA', 'm_ij': 'm',
                     'm_ij_lrs': 'm_lrs'})  # .drop(columns=['a_ij_not_obs','Lambda_ij_not_obs'])
        unknown_interactions = []
    A = A[['protein_x', 'protein_y', 'WeSA', 'SA', 'O_spoke_x', 'O_spoke_y', 'O_matrix', 'interactors']]  #
    A = A.round({'WeSA': 1, 'SA': 1})
    A = A.rename(columns={'protein_x': 'Protein A', 'protein_y': 'Protein B', 'O_spoke_x': 'Count: A retrieved B',
                          'O_spoke_y': 'Count: B retrieved A', 'O_matrix': 'Count:matrix'})
    A = A.sort_values('WeSA').reset_index(drop=True)

    mynodes = list(set((A['Protein A'])).union(set(A['Protein B'])))
    with open(node_info_file, 'rb') as file:
        corum_info = pickle.load(file)
    temp_dict = {}
    A_only_interactions_above_threshold = A.loc[A.iloc[:, 2] > db_threshold, :]
    has_green_neighbours = list(set(A_only_interactions_above_threshold['Protein A']).union(
        set(A_only_interactions_above_threshold['Protein B'])))
    node_color = []
    for i in mynodes:
        hex_color = '#F60E0E'
        if i in has_green_neighbours:
            hex_color = '#39BF58'
        node_color.append(hex_color)
        if i in corum_info.keys():
            temp_dict[i] = {'Complexes': corum_info[i]['Complexes'],
                            "number-of-complexes": corum_info[i]['No. complexes']}
        else:
            temp_dict[i] = {'Complexes': 'None', 'number-of-complexes': "0"}

    logging.info("Number of proteins in graph: %s", str(len(mynodes)))

    mygraph = [
        {
            "group": "nodes",
            "data": {
                        "id": mynodes[i],
                        "node_color": node_color[i],
                        "number-of-complexes": str(temp_dict[mynodes[i]]['number-of-complexes']),
                        "complexes": str(temp_dict[mynodes[i]]['Complexes'])
                     }
        } for i in range(len(mynodes))
    ]

    nodes_with_solid_connections = set()
    A = A.sort_values('WeSA').reset_index(drop=True)
    A['linewidth'] = np.logspace(np.log10(0.1), np.log10(6), num=np.shape(A)[0])
    for i in range(np.shape(A)[0]):
        if A.iloc[i, 2] > db_threshold:
            edge_color = '#39BF58'
        else:
            edge_color = '#F60E0E'

        temp_thresholds = [0, 0, 0, 0]
        for j in range(1, 5):
            if A.iloc[i, 2] > db_thresholds[j]:
                temp_thresholds[j - 1] = 'keep'
            else:
                temp_thresholds[j - 1] = 'discard'

        if (A.iloc[i, np.shape(A)[1] - 1] in unknown_interactions):
            edge_linestyle = 'dashed'
            print('I am in dashed if because:', A.iloc[i, np.shape(A)[1] - 1])
        else:
            edge_linestyle = 'solid'
            nodes_with_solid_connections = nodes_with_solid_connections.union(set(A.iloc[i, :2]))

        mygraph.append(
            {
                "group": "edges",
                "data": {
                    "id": i,
                    "source": A.iloc[i, 0],
                    "target": A.iloc[i, 1],
                    "WeSA": A.iloc[i, 2],
                    "SA": A.iloc[i, 3],
                    "Observed: bait = source": A.iloc[i, 4],
                    "Observed: bait = target": A.iloc[i, 5],
                    "Observed matrix": A.iloc[i, 6],
                    "color": edge_color,
                    "linestyle": edge_linestyle,
                    "linewidth": A.iloc[i, np.shape(A)[1] - 1],
                    "fpr1": temp_thresholds[0],
                    "fpr5": temp_thresholds[1],
                    "fpr10": temp_thresholds[2],
                    "fpr20": temp_thresholds[3]
            }}
        )

    for i in mygraph:
        if i['group'] == 'nodes':
            if i['data']['id'] in nodes_with_solid_connections:
                i['data']['nodeLinestyles'] = 'include_solid'
            else:
                i['data']['nodeLinestyles'] = 'only_dashed'

    with open(f'{job_dir}{graph_outfile}', 'wt') as file_out:
        json.dump(mygraph, file_out)
    logging.info("Graph created in %s", f'{job_dir}{graph_outfile}')

    A = A[mycolumns]
    output = json.loads(A.to_json(orient='split'))
    with open(f'{job_dir}{ints_outfile}', 'wt') as file_out:
        json.dump(output, file_out)
    logging.info("Results created in %s", f'{job_dir}{ints_outfile}')

    return


@this_app.route('/')
@this_app.route("/index")
def index():
    return render_template('index.html')

@this_app.route("/help")
def help():
        return render_template('help.html')

@this_app.route('/submit', methods=['POST'])
def submit():
    # Receive data
    input1 = request.form["prots_input"]
    input2 = request.files["file_prots"]
    bg_name = request.form['DataSource']
    if input2.filename == '':
        input = input1
    else:
        input = input2.read().decode("utf-8")

    if not len(input.strip()):
        return render_template(template_input_error)

    # Generate job_id
    job_id = f.generate_job_id()

    # Create job directory
    job_dir = f'{output_dir}job_{job_id}/'
    try:
        os.mkdir(job_dir)
    except OSError:
        logging.error(f"Could not create {job_dir}")

    logging.info("Initiated job %s", job_id)

    # Process the data and queue the task with Celery
    task = process_data.delay(input, bg_name, job_dir)

    logging.info("Task is running with task_id: %s", task.id)

    return redirect(url_for('results', task_id=task.id, job_id=job_id))


@this_app.route('/status/<task_id>')
def task_status(task_id):
    task = process_data.AsyncResult(task_id)
    if task.state == 'PENDING':
        # job did not start yet
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        # job succeeded
        response = {
            'state': 'Completed',
            'result': task.result
        }
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'status': str(task.info)  # this is the exception raised
        }
    return jsonify(response)


@this_app.route('/results/<task_id>_<job_id>')
def results(task_id, job_id):
    # Check the status of the task using task_id
    # If the task is completed, return the results
    # Otherwise, inform the user that the job is still processing
    return render_template('results.html',
                           task_id=task_id,
                           graph_json=f'jobs/job_{job_id}/mygraph.json',
                           ints_json=f'jobs/job_{job_id}/myresults.json',
                           columns=mycolumns,
                           title='Your scored results')


if __name__ == '__main__':
    APP = Flask(__name__)
    APP.debug = True
    APP.run()
