from pathlib import Path
import omnifig as fig
# from ..graphs.revised import GraphProperty,Estimand, LinearGraph,PropGraph,BernGraph
from .. import util
# from causalnlp import ci_relation_check
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
# from  ..verbal import Verbalizer

from ..graphs.builders import NiceSCMBuilder
from ..queries import create_query
from ..generator import generate_questions, extract_summary_data
from ..verbal.anticommonsense import AnticommonsenseTransformation
from ..verbal import load_story
from ..graphs import create_graph
from ._test_util import example_labels


_temp_data_file_path = Path('temp-test-data.json')
_temp_data_summary_path = Path('temp-test-data-summary.csv')


def test_anticommonsense():

	original = load_story('simpson_drug')

	anti = AnticommonsenseTransformation()

	changed = anti.transform(original)

	print()
	print(changed)



def test_generate_questions():

	builder = NiceSCMBuilder()

	story_id = 'simpson_drug'

	query = create_query('ate',
	                     ask_treatments=False,
	                     ask_polarities=False,
	                     ask_outcomes=False)

	entries = [q for q in generate_questions(story_id, builder, None, query, spec_limit=10, pbar=False)]

	assert len(entries) == 10



def test_full_generator():
	if _temp_data_file_path.exists():
		_temp_data_file_path.unlink()

	outpath = fig.quick_run('generate',
		path=str(_temp_data_file_path),

		stories=['simpson_drug'],
		queries=['ate'],

		seed=1001,
		spec_limit=5,
	)

	assert _temp_data_file_path.exists()



def test_extract_summary():

	builder = NiceSCMBuilder()

	story_id = 'simpson_drug'

	query = create_query('ate',
	                     ask_treatments=False,
	                     ask_polarities=False,
	                     ask_outcomes=False)

	entries = [q for q in generate_questions(story_id, builder, None, query, spec_limit=10, pbar=False)]

	summary = extract_summary_data(entries)

	assert len(summary) == 10



def test_summary():

	outpath = fig.quick_run('summary',
		path=str(_temp_data_file_path),
		out_path=str(_temp_data_summary_path),
	)

	assert outpath.exists()



def test_cleanup_temp_data():

	assert _temp_data_file_path.exists()
	assert _temp_data_summary_path.exists()

	_temp_data_file_path.unlink()
	_temp_data_summary_path.unlink()

























