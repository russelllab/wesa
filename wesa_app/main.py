import sys
import os
import re
import json
import logging
import time
import pandas as pd
import numpy as np
import pickle5 as pickle
from flask import Flask, request, render_template, redirect, url_for, jsonify, make_response, session
from celery import Celery
from dotenv import dotenv_values

from wesa_app import this_app
from . import functions as f


# Set file paths and names
HERE_PATH = os.path.dirname(os.path.abspath(__file__))
config = dotenv_values(dotenv_path=f"{HERE_PATH}/.env")
path_to_data = config["DATA_PATH"]
node_info_file = f'{path_to_data}/node_info_corum.pkl'
output_dir = f'{HERE_PATH}/static/jobs/'
graph_outfile = 'mygraph.json'
ints_outfile = 'myresults.json'
summary_outfile = "summary.txt"
template = {
    "index": "index.html",
    "results": "results.html",
    "help": "help.html",
    "input_error": "input_error.html",
    "internal_error": "internal_error.html"
}


# Configure logging
logging.basicConfig(
    filename=f'{HERE_PATH}/wesa.log',
    filemode="a",
    level=logging.INFO,
    format="[%(process)d] %(asctime)s - [%(levelname)s] %(message)s",
)

positive_color = '#005AB5'
negative_color = '#F60E0E'
network_node_limit = 100
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

databases = {
    "intact": "IntAct",
    "biogrid": "BioGRID",
    "bioplex": "BioPlex",
    "intact_bioGrid": "BioGRID + IntAct",
    "intact_bioPlex": "BioPlex + IntAct",
    "bioGrid_bioPlex": "BioPlex + BioGRID",
    "all": "BioPlex + BioGRID + Intact"
}


def make_celery(app):
    celery = Celery(app.import_name, broker=app.config['broker_url'])
    celery.conf.update(app.config)
    celery.conf.update(worker_hijack_root_logger=False)
    return celery

celery = make_celery(this_app)


def get_db_thresholds(bg_name):
    db_vector = ['biogrid', 'bioplex', 'intact', 'intact_bioGrid', 'intact_bioPlex', 'bioGrid_bioPlex', 'all']
    # threshold_vector = [18.78, 31.82, 11.43, 73.86, 24.73, 51.08, 21.94]
    threshold_vector = [24.64, 7.33, 25.57, 31.47, 10.99, 12.7, 14.77]
    fpr1_threshold_vector = [64.66, 45.66, 84.61, 81.74, 94.87, 62.73, 67.53]
    fpr5_threshold_vector = [36.61, 28.44, 46.21, 49.92, 53.65, 36.67, 40.58]
    fpr10_threshold_vector = [29.02, 15.57, 33.02, 36.06, 29.21, 20.84, 25.0]
    fpr20_threshold_vector = [19.69, 8.97, 23.15, 25.43, 14.67, 11.41, 12.26]

    all_thresholds_matrix = pd.DataFrame(
        [threshold_vector, fpr1_threshold_vector, fpr5_threshold_vector, fpr10_threshold_vector, fpr20_threshold_vector]
    )
    db_thresholds = all_thresholds_matrix[[j for j in range(len(db_vector)) if db_vector[j] == bg_name][0]]
    db_threshold = db_thresholds[0]

    return db_thresholds, db_threshold


def custom_serializer(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def create_graph_nodes(protein_list, has_green_neighbours, corum_info,
                       node_limit=100, node_list=[], graph=[]):
    node_list = set(node_list)
    for gene in protein_list:
        if gene not in node_list:
            if len(graph) >= node_limit:
                break

            node_color = negative_color
            if gene in has_green_neighbours:
                node_color = positive_color

            if gene in corum_info.keys():
                complex_info = {'Complexes': corum_info[gene]['Complexes'],
                                   'number-of-complexes': corum_info[gene]['No. complexes']}
            else:
                complex_info = {'Complexes': 'None', 'number-of-complexes': "0"}

            graph.append(
                {
                    "group": "nodes",
                    "data": {
                                "id": gene,
                                "node_color": node_color,
                                "number-of-complexes": str(complex_info['number-of-complexes']),
                                "complexes": str(complex_info['Complexes'])
                             }
                }
            )
            node_list.add(gene)
    return graph, list(node_list)


@celery.task
def process_data(query, database, job_dir, job_id):
    # Identify background data
    df_bg = pd.read_csv(path_to_data + database + '.txt', sep='\t')
    logging.info("%s Using background file: '%s'", job_id, path_to_data + database + '.txt')
    with open(path_to_data + 'article_mydict_matrix_' + database + '.pkl', 'rb') as file:
        mydict_bg = pickle.load(file)
    with open(path_to_data + 'article_matr_list_' + database + '.pkl', 'rb') as file:
        matr_list_bg = pickle.load(file)

    # Get corresponding thresholds
    db_thresholds, db_threshold = get_db_thresholds(database)

    # Format Input
    input_list = ('{0}'.format('{0}'.format(query.replace(' ', ';')).replace('\r', ''))).split('\n')
    input_list = [re.sub(';+', ';', re.sub(';*$', '', i)) for i in input_list]
    input_list = list(set([i for i in filter(None, input_list)]))
    df_input = pd.DataFrame(input_list)[0].str.split(';', expand=True).rename(
        columns={0: 'bait', 1: 'prey', 2: "identifier"}
    )

    if np.shape(df_input)[1] > 1:
        input_proteins = list(set(df_input['bait']).union(set(df_input['prey'])))
        nr_pairs = str(len(input_list))
        nr_proteins = str(len(input_proteins))
        logging.info("%s Input contains: %s protein pairs & %s unique proteins",
                     job_id,
                     nr_pairs,
                     nr_proteins)
        if np.shape(df_input)[1] == 2:
            df_input['identifier'] = 'new'
        queried_interactions = df_input.apply(
            lambda row: ';'.join(sorted([str(row['bait']), str(row['prey'])], reverse=True)), axis=1
        )

        # Matrices for input
        s, result0 = f.compute_spoke_ij_conf_weighted(df_input, use_conf=False, with_shrinkage=False, shrink_param=1)
        s = f.remove_self_loops(s, 'bait', 'prey')
        prey_id_dict = f.create_prey_id_dict(df_input)
        matr_list1 = f.create_matr_list(s, prey_id_dict)

        # Update matrix
        mydict = f.merge_matr_dicts(mydict_bg, prey_id_dict)
        matr_list = f.merge_matr_dicts(matr_list_bg, matr_list1)

        # Calculate everything
        df_all = pd.concat([df_input, df_bg[['bait', 'prey', 'identifier']]])
        A = f.create_A_dict_v(df_all, path_to_data, fn_matr_dict=mydict, fn_matr_list=matr_list)
        A = A[A['interactors'].isin(queried_interactions)].rename(
            columns={'bait': 'protein_x', 'prey': 'protein_y', 'tot_pur_x': 'tot_pur', 'Lambda_ij': 'WeSA',
                     'a_ij': 'SA', 'm_ij': 'm', 'm_ij_lrs': 'm_lrs'})

        # Determine previously unknown interactions
        interactors_set = set(map(str, df_bg['interactors']))
        unknown_interactions = list(set([str(i) for i in queried_interactions if
                                not any(str(i) in interactor for interactor in interactors_set)]))
        logging.info(
            "%s Input contains %s protein interactions not previously observed in the selected database",
            job_id, str(len(unknown_interactions))
        )


    else:  # the case where we just request to see old records for protein of interest
        input_proteins = list(set(df_input["bait"]))
        nr_pairs = "none"
        nr_proteins = str(len(input_proteins))
        logging.info("%s Input contains %s unique proteins", job_id, str(len(input_proteins)))
        df_bg = pd.read_csv(path_to_data + 'A_article_' + database + '_' + 'unweighted' + '.tsv', sep='\t')
        A = df_bg[(df_bg.bait.isin(list(df_input.iloc[:, 0]))) | (df_bg.prey.isin(list(df_input.iloc[:, 0])))].rename(
            columns={'bait': 'protein_x', 'prey': 'protein_y', 'tot_pur_x': 'tot_pur', 'Lambda_ij': 'WeSA',
                     'a_ij': 'SA', 'm_ij': 'm',
                     'm_ij_lrs': 'm_lrs'})  # .drop(columns=['a_ij_not_obs','Lambda_ij_not_obs'])
        unknown_interactions = []

    A = A[['protein_x', 'protein_y', 'WeSA', 'SA', 'O_spoke_x', 'O_spoke_y', 'O_matrix', 'interactors']]  #
    A = A.round({'WeSA': 1, 'SA': 1})
    A = A.rename(columns={'protein_x': 'Protein A', 'protein_y': 'Protein B', 'O_spoke_x': 'Count: A retrieved B',
                          'O_spoke_y': 'Count: B retrieved A', 'O_matrix': 'Count:matrix'})
    A = A.sort_values('WeSA').reset_index(drop=True)

    added_proteins = set((A['Protein A'])).union(set(A['Protein B']))
    added_proteins = list(added_proteins.difference(set(input_proteins)))
    logging.info("%s Added %s proteins from selected database", job_id, str(len(added_proteins)))

    # Create network elements
    # Create nodes
    with open(node_info_file, 'rb') as file:
        corum_info = pickle.load(file)

    A_only_interactions_above_threshold = A.loc[A.iloc[:, 2] > db_threshold, :]
    has_green_neighbours = list(set(A_only_interactions_above_threshold['Protein A']).union(
        set(A_only_interactions_above_threshold['Protein B'])))

    mygraph, mynodes = create_graph_nodes(
        input_proteins, has_green_neighbours, corum_info,
        node_limit=network_node_limit, node_list=[], graph=[]
    )

    mygraph, mynodes = create_graph_nodes(
        added_proteins, has_green_neighbours, corum_info,
        node_limit=network_node_limit, node_list=mynodes, graph=mygraph
    )

    logging.info("%s Number of nodes in graph: %s", job_id, str(len(mygraph)))

    # Create edges
    nodes_with_solid_connections = set()
    A = A.sort_values('WeSA').reset_index(drop=True)
    A['linewidth'] = np.logspace(np.log10(0.1), np.log10(6), num=np.shape(A)[0])
    for i in range(np.shape(A)[0]):
        node_a = A.loc[i, "Protein A"]
        node_b = A.loc[i, "Protein B"]
        if node_a in mynodes and node_b in mynodes:
            if A.loc[i, "WeSA"] > db_threshold:
                edge_color = positive_color
            else:
                edge_color = negative_color

            if A.loc[i, "interactors"] in unknown_interactions:
                edge_linestyle = 'dashed'
            else:
                edge_linestyle = 'solid'
                nodes_with_solid_connections = nodes_with_solid_connections.union(set(A.iloc[i, :2]))

            temp_thresholds = []
            for j in range(1, 5):
                if A.loc[i, "WeSA"] > db_thresholds[j]:
                    temp_thresholds.append('keep')
                else:
                    temp_thresholds.append('discard')

            mygraph.append(
                {
                    "group": "edges",
                    "data": {
                        "id": i,
                        "source": node_a,
                        "target": node_b,
                        "WeSA": A.loc[i, "WeSA"],
                        "SA": A.loc[i, "SA"],
                        "Observed: bait = source": A.loc[i, "Count: A retrieved B"],
                        "Observed: bait = target": A.loc[i, "Count: B retrieved A"],
                        "Observed matrix": A.loc[i, "Count:matrix"],
                        "color": edge_color,
                        "linestyle": edge_linestyle,
                        "linewidth": A.loc[i, "linewidth"],
                        "fpr1": temp_thresholds[0],
                        "fpr5": temp_thresholds[1],
                        "fpr10": temp_thresholds[2],
                        "fpr20": temp_thresholds[3]
                    }
                }
            )
    logging.info("%s Number of edges in graph: %s", job_id, str(np.shape(A)[0]))

    for ele in mygraph:
        if ele['group'] == 'nodes':
            if ele['data']['id'] in nodes_with_solid_connections:
                ele['data']['nodeLinestyles'] = 'include_solid'
            else:
                ele['data']['nodeLinestyles'] = 'only_dashed'

    with open(f'{job_dir}{graph_outfile}', 'wt') as file_out:
        json.dump(mygraph, file_out, default=custom_serializer)
    logging.info("%s Graph created in '%s'", job_id, f'{job_dir}{graph_outfile}')

    A = A[mycolumns]
    output = json.loads(A.to_json(orient='split'))
    with open(f'{job_dir}{ints_outfile}', 'wt') as file_out:
        json.dump(output, file_out, default=custom_serializer)
    logging.info("%s Results created in '%s'", job_id, f'{job_dir}{ints_outfile}')

    with open(f'{job_dir}{summary_outfile}', 'wt') as file_out:
        file_out.write(f"databases={databases[database]}\n")
        file_out.write(f"nr_pairs={nr_pairs}\n")
        file_out.write(f"nr_proteins={nr_proteins}\n")
    logging.info("%s Summary file created in '%s'", job_id, f'{job_dir}{summary_outfile}')

    return

@this_app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            # Append a timestamp to the query string of the URL
            values['q'] = int(time.time())
    return url_for(endpoint, **values)

@this_app.route('/')
@this_app.route("/index")
def index():
    return render_template(template["index"])


@this_app.route("/help")
def help():
    return render_template(template["help"])


@this_app.route('/submit', methods=['POST'])
def submit():
    # Receive data
    input_prot = request.form["prots_input"]
    input_file = request.files["file_prots"]
    bg_name = request.form['DataSource']
    if input_file.filename == '':
        job_input = input_prot
    else:
        job_input = input_file.read().decode("utf-8")

    # Generate job_id
    job_id = f.generate_job_id()
    short_job_id = f"[{job_id[:8]}...]"
    logging.info("%s STARTING job with job_id: %s", short_job_id, job_id)

    # Check if the input is empty
    if not len(job_input.strip()):
        logging.error("%s Input box is empty. TERMINATING", short_job_id)
        return render_template(template["input_error"])

    # Create job directory
    job_dir = f'{output_dir}job_{job_id}/'
    try:
        os.mkdir(job_dir)
    except OSError:
        logging.error("%s Could not create %s", short_job_id, job_dir)
        return render_template(template["internal_error"], job_id=job_id)

    # Process the data and queue the task with Celery
    task = process_data.delay(query=job_input, database=bg_name,
                              job_dir=job_dir, job_id=short_job_id)
    logging.info("%s Processing job as Celery task running with task_id: %s", short_job_id, task.id)

    return redirect(url_for('results', job_id=job_id, task_id=task.id))


@this_app.route('/status/<job_id>_<task_id>')
def task_status(job_id, task_id):
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
        logging.info("%s Task %s completed", job_id, task_id)
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'status': str(task.info)  # this is the exception raised
        }
    return jsonify(response)


@this_app.route('/results/<job_id>_<task_id>')
def results(job_id, task_id):

    response = make_response(
        render_template(
            template["results"],
            job_id=job_id,
            task_id=task_id,
            graph_json=f'jobs/job_{job_id}/{graph_outfile}',
            ints_json=f'jobs/job_{job_id}/{ints_outfile}',
            columns=mycolumns,
            title='Your scored results')
    )

    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@this_app.route('/params/<job_id>')
def params(job_id):
    try:
        with open(f'{output_dir}/job_{job_id}/{summary_outfile}', 'r') as file:
            params = file.read().split('\n')
            params_dict = {param.split('=')[0]: param.split('=')[1] for param in params if param}
            return jsonify(params_dict)
    except FileNotFoundError:
        logging.error(f'"jobs/job_{job_id}/{summary_outfile}" not found.')
        return jsonify({'error': 'Parameter file not found'}), 404
    except Exception as e:
        logging.error(f'Error reading "jobs/job_{job_id}/{summary_outfile}": {str(e)}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    APP = Flask(__name__)
    APP.debug = True
    APP.run()
