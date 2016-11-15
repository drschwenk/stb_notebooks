
# coding: utf-8

# # Table of Contents
# * &nbsp;
# 	* [imports](#imports)
# 	* [simple functions](#simple-functions)
# * [Load data](#Load-data)
# 	* [setting paths](#setting-paths)
# 	* [parsed content and raw questions and descriptions](#parsed-content-and-raw-questions-and-descriptions)
# 	* [localization and recognition](#localization-and-recognition)
# 	* [building spellings and grammar](#building-spellings-and-grammar)
# * [Clean and prepare data](#Clean-and-prepare-data)
# 	* [extract media links](#extract-media-links)
# 	* [remove non-conforming content](#remove-non-conforming-content)
# 		* [code](#code)
# 		* [run](#run)
# 	* [remove recognition and localization errors](#remove-recognition-and-localization-errors)
# * [Add image annotations](#Add-image-annotations)
# 	* [localization](#localization)
# 	* [recognition](#recognition)
# 		* [code](#code)
# 		* [run](#run)
# 		* [hide](#hide)
# * [Integrate diagram questions and descriptions](#Integrate-diagram-questions-and-descriptions)
# 	* [match diagram topics to lessons](#match-diagram-topics-to-lessons)
# 		* [code](#code)
# 		* [run](#run)
# 		* [hide](#hide)
# 	* [merge questions](#merge-questions)
# 		* [code](#code)
# 		* [run](#run)
# 		* [hide](#hide)
# 	* [merge descriptions](#merge-descriptions)
# 		* [hide](#hide)
# 	* [Apply spelling and grammar fixes](#Apply-spelling-and-grammar-fixes)
# 		* [code](#code)
# 		* [run](#run)
# 		* [hide](#hide)
# * [Topic key collisions](#Topic-key-collisions)
# * [Refinements to make](#Refinements-to-make)
# * [End](#End)
# 

# ## imports

# In[378]:

get_ipython().run_cell_magic('capture', '', 'import matplotlib as mpl\nmpl.use("Agg")\nimport matplotlib.pylab as plt\n#%matplotlib notebook\n%matplotlib inline\n%load_ext base16_mplrc\n# %base16_mplrc light so  larized\n%base16_mplrc dark solarized\nplt.rcParams[\'grid.linewidth\'] = 0\nplt.rcParams[\'figure.figsize\'] = (16.0, 10.0)\n\nimport numpy as np\nimport pandas as pd\nimport scipy.stats as st\nfrom scipy.stats.mstats import mode\n\nimport itertools\nimport math\nfrom collections import Counter, defaultdict, OrderedDict\n%load_ext autoreload\n%autoreload 2\n\nimport cv2\nimport pprint\nimport pickle\nimport json\nimport requests\nimport io\nimport sys\nimport os\nfrom binascii import b2a_hex\nimport base64\nfrom wand.image import Image as WImage\nfrom IPython.display import display\nfrom IPython.core.display import HTML\nimport PIL.Image as Image\nfrom copy import deepcopy\nimport glob\nimport random\n\nimport nltk\nimport string\nfrom nltk.corpus import stopwords\nfrom nltk.tokenize import word_tokenize\nfrom nltk.tokenize import wordpunct_tokenize\nfrom nltk.stem.wordnet import WordNetLemmatizer\nimport language_check\nimport enchant\nimport difflib\nimport diff_match_patch\nimport fuzzywuzzy.fuzz as fuzz\nimport re\nimport jsonschema\nfrom pdfextraction.ck12_schema import ck12_schema as schema\nfrom pdfextraction.ck12_new_schema import ck12_schema as new_schema')


# ## simple functions

# In[99]:

def write_file(filename, data_dict, output_dir='output_data_from_nbs'):
    with open(os.path.join(output_dir, filename), 'w') as f:
        json.dump(data_dict, f, indent=4, sort_keys=True)
        
def get_img_n(image_name):
    return [re.findall("[0-9]+", image_name)][0][0]

def clean_list(dir_path):
    hidden_removed = filter(lambda f: not f.startswith('.'), os.listdir(dir_path))
    return [topic.replace('_diagram', '') for topic in hidden_removed]


# # Load data

# load last_checkpoint

# In[323]:

with open('build_v6.pkl', 'rb') as f:
    latest_checkpoint = pickle.load(f)


# In[325]:

latest_checkpoint.keys()


# ## setting paths

# In[242]:

output_dir = 'output_data_from_nbs/'
raw_data_dir = '../spare5_produced_data/data/'
raw_dq_file = 'ai2_testquestions_20161005.csv'
s5_raw_decriptions = 'ai2_diagramdescriptions_20161018.csv'
ai2_raw_decriptions = 'our_description.csv'
glossary_path = os.path.join(output_dir, 'flexbook_glossary.pkl')

turk_proc_dir = '/Users/schwenk/wrk/stb/diagram_questions/turk_processing/'
metadata_dir = turk_proc_dir + 'store_hit_results_metadata/'
lc_results_dir = 'loc_group_3'
box_loc_joined = 'loc_annotations'
recog_results_dir = 'group_latest_combined'

box_choices_1_dir = 'final_text_boxes_fixed'
box_choices_2_dir = 'final_text_boxes_pass_2'
none_agree = 'no_turkers_agree_lookup.pkl'
two_agree_lookup = 'two_turkers_agree_lookup.pkl'
all_agree_lookup = 'user_diag_loopkup.pkl'

recog_performed = '/Users/schwenk/wrk/stb/diagram_questions/turk_processing/final_diagrams/'
all_dir = '/Users/schwenk/wrk/stb/ai2-vision-textbook-dataset/diagrams/tqa_diagrams_v0.9/'
pruned_dir = '/Users/schwenk/wrk/stb/ai2-vision-textbook-dataset/diagrams/dataset_Sep_27/tqa_diagrams_v0.9_question_images/'
description_dir = '/Users/schwenk/wrk/stb/spare5_produced_data/tqa_diagrams_v0.9_inbook/'


# ## parsed content and raw questions and descriptions

# In[379]:

get_ipython().run_cell_magic('capture', '', "# load complete text v 3.5, raw diagram questions and descriptions\n# now ck12_dataset_text_only_beta_v5_build.json\nwith open(output_dir + 'ck12_dataset_text_only_beta_v5_build.json', 'r') as f:\n    ck12_combined_dataset_raw = json.load(f)\nwith open(output_dir + 'ck12_flexbook_only_beta_v3.json', 'r') as f:\n    flexbook_ds = json.load(f)\nwith open(output_dir + 'ck12_lessons_only_beta_v3.json', 'r') as f:\n    lessons_ds = json.load(f)\n\n# loading questions\ndesc_df = pd.read_csv(raw_data_dir + s5_raw_decriptions, encoding='latin-1')\ndesc_df['diagram'] = desc_df['reference_id'].apply(lambda x: x.split('/')[-1])\n\nai2_raw_decriptions_df = pd.read_csv(raw_data_dir + ai2_raw_decriptions, encoding='latin-1')\nai2_written_df_completed = ai2_raw_decriptions_df[['Topic', 'Image Path', 'Description']]\nai2_written_df_completed['diagram'] = ai2_written_df_completed['Image Path'].apply(lambda x: x.split('/')[-1])\nai2_written_df_completed['topic'] = ai2_written_df_completed['Topic']\ndel  ai2_written_df_completed['Topic']\n\n#loading questions\n# q_col = '03_write_question'\n# r_ans_col = '04_write_right_answer'\n# w_ans_col = '05_write_wrong_answers'\n# data_cols = [q_col, r_ans_col, w_ans_col]\n# raw_dq_df = pd.read_csv(raw_data_dir + raw_dq_file, encoding='latin-1')\n# dr_proc_df = raw_dq_df.copy()\n# dr_proc_df['wac_list'] = dr_proc_df[w_ans_col].apply(lambda x: json.loads(x))\n# dr_proc_df['diagram'] = dr_proc_df['reference_id'].apply(lambda x: x.split('/')[-1])\n# dr_proc_df['topic'] = dr_proc_df['reference_id'].apply(lambda x: x.split('/')[-1].rsplit('_', maxsplit=1)[0])\n\ndr_proc_df = pd.read_pickle('complete_s5_dq.pkl')\ndr_proc_df['diagram'] = dr_proc_df['reference_id'].apply(lambda x: x.split('/')[-1])\n\n\nwith open('../diagram_questions/topic_match_terms.json', 'r') as f:\n    topic_term_match = json.load(f)    \n\nwith open(glossary_path, 'rb') as f:\n    flexbook_glossary = pickle.load(f)")


# ### localization and recognition

# In[244]:

loc_res_df = pd.read_pickle(os.path.join(metadata_dir, lc_results_dir, 'complete_df.pkl'))
recog_res_df = pd.read_pickle(os.path.join(metadata_dir, recog_results_dir, 'recog_df.pkl'))


# ## building spellings and grammar

# In[106]:

# loading spelling defs
with open(output_dir + 'ck_12_vocab_words.pkl', 'rb') as f:
    ck_12_vocab = set(pickle.load(f))
with open(output_dir + 'ck_12_all_words.pkl', 'rb') as f:
    ck_12_corp = set(pickle.load(f))
    
with open(output_dir + 'spellings_to_rev.txt', 'r') as f:
    whitelisted_words = f.read().split('\n')[:-1]    
with open(output_dir + './desc_spellings_to_rev.txt', 'r') as f:
    whitelisted_words += f.read().split('\n')[:-1]
with open(output_dir + './ck_12_spelling_rev.txt', 'r') as f:
    whitelisted_words += f.read().split('\n')[:-1]
with open('diagram_rec_corp.pkl', 'rb') as f:
    diagram_rec_corpus = pickle.load(f)
    
ck_12_corp.update(ck_12_vocab)
ck_12_corp.update(whitelisted_words)
ck_12_corp.update(diagram_rec_corpus)

# build spelling dict updated with words from science corpus
edict = enchant.Dict("en_US")
anglo_edict = enchant.Dict("en_UK")
cached_sw = stopwords.words("english") + list(string.punctuation)
for word in ck_12_corp:
    if word.isalpha() and len(word) > 3:
        edict.add(word)
        
# grammaer checker
gram_checker = language_check.LanguageTool('en-US')
gram_checker.disabled = set(['SENT_START_CONJUNCTIVE_LINKING_ADVERB_COMMA', 'POSSESSIVE_APOSTROPHE', 'A_PLURAL'])
gram_checker.disable_spellchecking()


# # Clean and prepare data

# ## extract media links

# In[380]:

ck12_combined_dataset = deepcopy(ck12_combined_dataset_raw)


# In[381]:

pat_str = "(?:https?:\/\/(?:www\.).*?\s)"
web_link_patern=re.compile(pat_str)

def clean_content_text(content_str, web_link_patern):
    removed_links = web_link_patern.findall(content_str)
    if not removed_links:
        return '', ''
    split_txt = web_link_patern.split(content_str)
    cleaned_text = ' '.join([txt for txt in split_txt if txt])
    return cleaned_text, [link.strip() for link in removed_links]

def extract_links(complete_ds):
    for subject, lessons in complete_ds.items():
        for lesson_title, lesson in lessons.items():
            for topic, content in lesson['topics'].items():
                content_str = content['content']['text']
                new_text, links = clean_content_text(content_str, web_link_patern)
                content['content']['mediaLinks'] = []
                if links:
                    content['content']['text'] = new_text
                    content['content']['mediaLinks'].extend(links)


# In[382]:

extract_links(ck12_combined_dataset)


# ## remove non-conforming content

# ### old code (nested)

# In[383]:

def validate_schema(dataset_json, ):
    errors = []
    try:
        validator = jsonschema.Draft4Validator(schema)
        for error in sorted(validator.iter_errors(dataset_json), key=lambda x: x.absolute_path[0]):
            errors.append([error.message, list(error.absolute_path)[:4]])
    except jsonschema.ValidationError as e:
        errors.append("Error in schema --%s-", e.message)
    return errors

def validate_dataset(dataset_json, scheme=schema):
    for subject, flexbook in dataset_json.items():
        schema_errors = validate_schema(flexbook, schema)
        for lesson_name, lesson in flexbook.items():
            ac_errors = check_ac_counts(lesson, subject, lesson_name)
        all_errors = schema_errors + ac_errors
        if not all_errors:
            return 'all validation test passed'
        else:
            return all_errors

def check_ac_counts(lesson_content, subject, lesson_name):
    errors = []
    for qid, question in lesson_content['questions']['nonDiagramQuestions'].items():
        if question['type'] == 'Multiple Choice':
            if len(question['answerChoices']) != 4:
                errors.append([subject, lesson_name, qid + ' mc error'])
        if question['type'] == 'True or False':
            if len(question['answerChoices']) != 2:
                errors.append([subject, lesson_name, qid + ' tf error'])
    return errors

def record_validation_errors(dataset):
    qs_removed = []
    for subject, flexbook in dataset.items():
        validator = jsonschema.Draft4Validator(schema)
        for error in sorted(validator.iter_errors(flexbook), key=lambda x: x.absolute_path[0]):
            lesson, quest, question_class, q_number = list(error.absolute_path)[:4]
            problem_q_section = dataset[subject][lesson][quest][question_class]
            if q_number in problem_q_section.keys():
#                 print(dataset[subject][lesson][quest][question_class].pop(q_number))
                qs_removed.append(dataset[subject][lesson][quest][question_class].pop(q_number))
    return qs_removed


# In[384]:

val_errors = record_validation_errors(ck12_combined_dataset)


# In[385]:

val_errors


# ### new code (flat)

# In[421]:

def validate_schema(dataset_json, schema):
    errors = []
    try:
        validator = jsonschema.Draft4Validator(schema)
        for error in sorted(list(validator.iter_errors(dataset_json)), key=lambda x: x.absolute_schema_path[0]):
            errors.append([error.message, list(error.absolute_path)[:4]])
    except jsonschema.ValidationError as e:
        errors.append("Error in schema --%s-", e.message)
    return errors

def validate_dataset(dataset_json, val_scheme):
    ac_errors = []
    schema_errors = validate_schema(dataset_json, val_scheme)

    if not all_errors:
        return 'all validation test passed'
    else:
        return all_errors    

def record_validation_errors(dataset, schema):
    qs_removed = []
    validator = jsonschema.Draft4Validator(schema)
    for error in sorted(validator.iter_errors(dataset), key=lambda x: x.absolute_path[0]):
        print(error)
        lesson, quest, question_class, q_number = list(error.absolute_schema_path)[:4]
        problem_q_section = dataset[subject][lesson][quest][question_class]
        if q_number in problem_q_section.keys():
            print(dataset[subject][lesson][quest][question_class].pop(q_number))
            qs_removed.append(dataset[subject][lesson][quest][question_class].pop(q_number))
    return qs_removed


# ### run

# In[438]:

from pdfextraction.ck12_new_schema import ck12_schema as new_schema


# In[439]:

len(complete_flat_ds)


# In[440]:

NDQ_018942


# In[471]:

validate_schema(complete_flat_ds, new_schema)


# In[306]:

len(errors_returned)


# In[296]:

len(errors_returned[0][0])


# In[293]:

errors_returned[0][0][1]


# In[259]:

len(errors_returned)


# In[244]:

from pdfextraction.ck12_new_schema import ck12_schema as new_schema
validator = jsonschema.Draft4Validator(new_schema)
for error in sorted(validator.iter_errors(complete_flat_ds), key=lambda x: x.absolute_path[0]):
    print(error.message)
    print(error.absolute_path)
    print()


# In[238]:

record_validation_errors(complete_flat_ds, new_schema)


# ### hide

# In[535]:

test_flat_ds = complete_flat_ds


# In[309]:

validate_schema(complete_flat_ds, new_schema)


# In[432]:


test_flat_ds[0]['questions']['diagramQuestions']['DQ_000031'].keys()


# In[ ]:




# In[177]:

qs_rem = record_validation_errors(ck12_combined_dataset)


# In[178]:

len(qs_rem)


# ## remove recognition and localization errors

# In[18]:

diagram_image_names = clean_list(recog_performed)

rec_files = glob.glob(all_dir + '*/*')
more_paths = glob.glob(all_dir + '*/*')
pruned_paths = glob.glob(pruned_dir + '*/*')
more_files = [fp.split('/')[-1] for fp in more_paths]
pruned_files = [fp.split('/')[-1] for fp in pruned_paths]
desc_paths = glob.glob(description_dir + '*/*')
desc_files = [fp.split('/')[-1] for fp in desc_paths]


pruned_nums = set([get_img_n(name) for name in pruned_files])
all_nums = set([get_img_n(name) for name in more_files])
rec_nums = set([get_img_n(name) for name in diagram_image_names])
desc_nums = set([get_img_n(name) for name in desc_files])

removed_images = all_nums.difference(pruned_nums.union(desc_nums))

removed_image_names = []
for img_n in removed_images:
    for image_name in more_files:
        if img_n == get_img_n(image_name):
            removed_image_names.append(image_name)

name_change_lookup = {}
for image_name in more_files:
    img_n = get_img_n(image_name)
    for newer_name in pruned_files:
        if img_n == get_img_n(newer_name) and newer_name != image_name:
            name_change_lookup[image_name] = newer_name

removed_image_names = sorted(removed_image_names)


# In[19]:

blacklisted_topics = ['periodic_table', 'em_spectrum', 'hydrocarbons', 'geologic_time'] + ['lewis_dot_idapgrams', 'circuits']  # correct this mispelling in future round


# In[20]:

len(removed_image_names)


# # Add image annotations

# ## localization

# In[360]:

loc_res_df.head(1)


# In[361]:

loc_anno = clean_list(os.path.join(turk_proc_dir, box_loc_joined))
loc_anno_images = [fig.split('.json')[0]  for fig in loc_anno]
keep_figures = [fig for fig in loc_anno_images if fig not in removed_image_names]

loc_box_path = os.path.join(turk_proc_dir, box_loc_joined)

diag_loc_annotations = {}
for diagram_name in keep_figures:
    anno_file_path = os.path.join(loc_box_path, diagram_name + '.json')
    if not os.path.exists(anno_file_path):
        diagram_name = diagram_name.replace('optics_rays', 'optics_ray_diagrams')
        anno_file_path = os.path.join(loc_box_path, diagram_name  + '.json')
    with open(anno_file_path, 'r') as f:
        diag_loc_annotations[diagram_name] = json.load(f)

combined_master_file_list = pruned_files + desc_files
combined_master_file_list_whitelisted = [file for file in combined_master_file_list if file in keep_figures]
files_still_needing_localisation = sorted(list(set(combined_master_file_list).difference(set(diag_loc_annotations))))
len(files_still_needing_localisation)


# ## recognition

# ### code

# In[362]:

def most_common_strict(image_response):
    """
    returns the consensus response of the three raw response strings for a given image
    """
    most_common = image_response[1]['raw_text'].mode()
    if most_common.empty:
        most_common = 'nonconsensus'
        noncon.append(image_response[1]['raw_text'])
    else:
        most_common = most_common.values[0]
    return most_common

def most_common_lax(image_response, strings_denoting_missing_image=[]):
    """
    returns the consensus response after stripping white space and converting the reponses to lower case
    """
    simple_sanitizer = lambda x : x.lower().strip().lstrip()
    ind_responses = image_response[1]['raw_text'].values
    probobly_blanks = [response for response in ind_responses if response in strings_denoting_missing_image]
    if probobly_blanks:
        return 'skip'
    most_common = image_response[1]['raw_text'].apply(simple_sanitizer).mode()
    if most_common.empty:
        most_common = 'no consensus'
        noncon[image_response[0][0]].extend(image_response[1]['raw_text'])
    else:
        most_common = most_common.values[0]
    return most_common

def find_transcriptions_matches(batch_results_df, response_matcher):
    """
    returns a pandas series with the consunsus response for each image
    """
    agreed_responses = pd.DataFrame()
    for image_response in batch_results_df.groupby(['diagram', 'box_diag_idx']):
        diagram_and_idx = image_response[0]
        most_common = response_matcher(image_response, strings_denoting_missing_image=[])
        if most_common == 'skip':
            continue
        this_row = pd.DataFrame(list(diagram_and_idx) + [most_common, image_response[1]['rectangle'].iloc[0], image_response[1]['assignment_id'].iloc[0]]).T
        agreed_responses = pd.concat([agreed_responses, this_row])
        # The reindex below is needed to match the original df index after the groupby operation
    agreed_responses.columns = ['diagram', 'box_diag_idx', 'consensus_res', 'rectangle', 'assignment_id']
    return agreed_responses


# ### run

# In[363]:

recog_performed_on = set(pd.unique(recog_res_df['diagram']).tolist())
len(recog_performed_on)


# In[364]:

files_still_needing_recognition = sorted(list(set(pruned_files).difference(set(recog_performed_on))))
print(len(files_still_needing_recognition))
file_with_loc_no_recog = set(files_still_needing_recognition).difference(files_still_needing_localisation)
print(len(file_with_loc_no_recog))


# In[365]:

noncon = defaultdict(list)
transcription_results_lax = find_transcriptions_matches(recog_res_df, most_common_lax)


# In[366]:

noncon_entries = [entries for entries in noncon.values()]
flattened_noncon = [item for sublist in noncon_entries for item in sublist]


# In[367]:

curated_no_image_strings = set(['*no image showing*', '', ' ', 'NA', '?', 'na', '0', 'No image found', 'blank', 'Nothing showing', "where is the images , i can't see anything", 'NO IMAGE', ''])


# In[368]:

non_blank_no_consensus = {d_name: rec_res for d_name, rec_res in noncon.items() if not curated_no_image_strings.intersection(set(rec_res))}
blank_no_consensus = {d_name: rec_res for d_name, rec_res in noncon.items() if curated_no_image_strings.intersection(set(rec_res))}
print(len(non_blank_no_consensus))
print(len(blank_no_consensus))


# In[369]:

flattened_noncon_no_blank = [item for sublist in non_blank_no_consensus.values() for item in sublist]
build_diagram_rec_corpus  = [words.split() for words in transcription_results_lax['consensus_res'].values.tolist()]
diagram_rec_corpus = set([item.lower().strip() for sublist in build_diagram_rec_corpus for item in sublist if item.isalpha() and len(item) > 3])


# In[451]:

# transcription_results_lax.to_pickle('recog_proc_checkpoint.pkl')


# ### hide

# In[370]:

# with open('diagram_rec_corp.pkl', 'wb') as f:
#     pickle.dump(diagram_rec_corpus, f)


# In[371]:

# strings_denoting_missing_image = list(pd.Series(flattened_noncon).value_counts()[:20].index)
# Image.open('../ai2-vision-textbook-dataset/diagrams/turk_data/optics_ray_diagrams_9170.png')


# In[372]:

len(diagram_rec_corpus)


# # Integrate diagram questions and descriptions

# ## match diagram topics to lessons

# first need to match diagram topics to flexbook lessons

# ### code

# In[32]:

def make_topic_matches(topic_list, combined_topics):
    topic_matches = {}
    for diagram_topic in topic_list:
        topic_matches[diagram_topic] = []
        for terms in topic_term_match[diagram_topic]:
            lev_dist_threshed = [topic for topic in combined_topics.keys() if fuzz.ratio(topic, terms) > 85]
            topic_matches[diagram_topic] += lev_dist_threshed
        if not topic_matches[diagram_topic]:
                for terms in topic_term_match[diagram_topic]:
                    lev_dist_threshed = [topic for topic in combined_topics.keys() if fuzz.token_set_ratio(topic, terms) > 80]
                    topic_matches[diagram_topic] += lev_dist_threshed
    return topic_matches

def make_lesson_matches(ck12_dataset, diagram_topic_name, topic_matches):
    lesson_matches = defaultdict(list)
    lessons_seen = set()
    content_topics =  topic_matches[diagram_topic_name]
    for topic in sorted(content_topics):
        associated_lesson =combined_topics[topic]['lesson']
        if associated_lesson not in lessons_seen:
            lessons_seen.add(associated_lesson)
            lesson_matches[diagram_topic_name].append(associated_lesson)
    return dict(lesson_matches)


# ### run

# The pruned directory is the tqa 0.91 set assmbled by Ani on Sept 27th. It should be treated as definitive

# In[43]:

diagram_topic_list = clean_list(pruned_dir)


# In[54]:

es_lesson_names = [item for sublist in [val['topics'].keys() for val in ck12_combined_dataset['earth-science'].values()] for item in sublist]
ps_lesson_names = [item for sublist in [val['topics'].keys() for val in ck12_combined_dataset['physical-science'].values()] for item in sublist]
ls_lesson_names = [item for sublist in [val['topics'].keys() for val in ck12_combined_dataset['life-science'].values()] for item in sublist]

combined_lessons = es_lesson_names + ps_lesson_names + ls_lesson_names
topic_series = pd.Series(combined_lessons).value_counts()
# the 17 here found by inspection- any "topic" appearing many times is something general like review, vocab, etc
topics_to_remove = list(topic_series[:16].index)


# In[55]:

topics_to_remove


# In[56]:

combined_topics = defaultdict(dict)
for subject, book in ck12_combined_dataset.items():
    for lesson, material in book.items():
        for topic, text in material['topics'].items():
            if topic in topics_to_remove:
                continue
            combined_topics[topic.lower()]['lesson'] = lesson


# In[57]:

topic_matches = make_topic_matches(diagram_topic_list, combined_topics)
missing= []
for k, v in topic_matches.items():
    if not v:
        missing.append(k)


# In[58]:

matching_lessons = {}
for topic in diagram_topic_list:
    matched_lessons = make_lesson_matches(ck12_combined_dataset, topic, topic_matches)
    matching_lessons.update(matched_lessons)


# In[59]:

diagram_lesson_lookup = {}
for d_topic, lessons in matching_lessons.items():
    diagram_lesson_lookup[d_topic] = sorted(lessons)[0]


# In[60]:

#manually correct name changes made since diagrams were assembled
diagram_lesson_lookup['lewis_dot_diagrams'] = diagram_lesson_lookup['lewis_dots']
diagram_lesson_lookup['optics_ray_diagrams'] = diagram_lesson_lookup['optics_rays']


# In[62]:

get_ipython().magic('page diagram_lesson_lookup')


# In[64]:

len(set(diagram_lesson_lookup.values()))


# In[931]:

diagram_lesson_lookup


# In[935]:

get_ipython().magic('page meta_lesson_id_lookup')


# In[932]:

meta_lesson_assignments


# ### hide

# In[381]:

lessons_seen = []
dupe_lessons = []
for k, v in diagram_lesson_lookup.items():
    if v not in lessons_seen:
        lessons_seen.append(v)
    else:
        dupe_lessons.append(v)


# In[382]:

dupe_topics = defaultdict(list)
for k, v in diagram_lesson_lookup.items():
    if v in dupe_lessons:
        dupe_topics[v].append(k)
# dupe_topics


# In[383]:

missing


# In[384]:

len(diagram_lesson_lookup.keys())

len(set(diagram_lesson_lookup.values()))

for k, v in sorted(matching_lessons.items()):
    print(k)
    print(sorted(v))
    print()


# In[385]:

# pprint.pprint(dict(dupe_topics))


# ## merge questions

# ### code

# In[250]:

dq_image_folder = 'diagram-question-images/'
td_image_folder = 'diagram-teaching-images/'

def make_question_entry(qdf_row):
    ask = qdf_row[qdf_row.index == '03_write_question'].values[0]
    answer = qdf_row[qdf_row.index == '04_write_right_answer'].values[0]
    wrong_answers = qdf_row[qdf_row.index == 'wac_list'].values[0]
    q_topic = qdf_row[qdf_row.index == 'lesson_assigned_to'].values[0]
    image_uri = qdf_row[qdf_row.index == 's3_uri'].values[0]
    image_name = qdf_row[qdf_row.index == 'diagram'].values[0]
    
    def make_answer_choices(answer_choices):
        build_answer_choices = {}
        letter_options = list('abcd')
        random.shuffle(answer_choices)
        for idx, answer_choice in enumerate(answer_choices):
            answer_choice_dict = {
                "idStructural": letter_options[idx] + '.',
                "rawText": answer_choice,
                "processedText": answer_choice
            }
            build_answer_choices[letter_options[idx]] = answer_choice_dict
        return build_answer_choices
    a_choices = make_answer_choices(wrong_answers + [answer])
    single_q_dict = {
        "id": 'q',
        "type": 
            "Diagram Multiple Choice",
        "beingAsked": {
            "rawText": ask,
            "processedText": ask.encode('ascii', 'ignore').decode('utf-8')
        },
        "correctAnswer": {
            "rawText": answer,
            "processedText": answer.encode('ascii', 'ignore').decode('utf-8')
        },
        "answerChoices": a_choices,
        "imageUri": image_uri,
        "imageName": image_name
    }
    build_questions[q_topic].append(single_q_dict)
    
    
def refine_question_formats(raw_questions):
    reformatted_dq_ds = {}
    for topic, topic_questions in raw_questions.items():
        reformatted_topic = {topic: {'questions': {'diagramQuestions': {}}}}
        reformatted_questions = {}
        for idx, question in enumerate(topic_questions):
            question = deepcopy(question)
            question['id'] += str(idx + 1).zfill(4)
            reformatted_questions[question['id']] = question
        reformatted_topic[topic]['questions']['diagramQuestions'] = reformatted_questions
        reformatted_dq_ds.update(reformatted_topic)
    return reformatted_dq_ds

s3_base = 'https://s3.amazonaws.com/ai2-vision-textbook-dataset/diagrams/' + dq_image_folder
s3_base_descriptions = 'https://s3.amazonaws.com/ai2-vision-textbook-dataset/diagrams/' + td_image_folder

def make_image_link(old_url, s3_base=s3_base):
    image_name = old_url.split('/')[-1]
    new_url = s3_base + image_name
    return new_url


# ### run

# In[386]:

dr_proc_df['s3_uri'] = dr_proc_df['reference_id'].apply(make_image_link)
dr_proc_df['lesson_assigned_to'] = dr_proc_df['topic'].apply(lambda x: diagram_lesson_lookup[x])


# In[387]:

build_questions = defaultdict(list)
_ = dr_proc_df.apply(make_question_entry, axis=1)


# In[388]:

refined_questions = refine_question_formats(build_questions)


# In[389]:

for subject, lessons in ck12_combined_dataset.items():
    for l_name, lesson in lessons.items():
        if l_name in refined_questions.keys():        
            lesson['questions']['diagramQuestions'] = refined_questions[l_name]['questions']['diagramQuestions']
        else:
            lesson['questions']['diagramQuestions']  = {}


# ### hide

# In[391]:

refined_questions = dict(refine_question_formats(build_questions))

refined_questions['10.4 Erosion and Deposition by Glaciers']['questions'].keys()


# In[392]:

refined_questions['10.4 Erosion and Deposition by Glaciers']

len(ck12_combined_dataset['earth-science']['10.4 Erosion and Deposition by Glaciers']['questions']['diagramQuestions'])

len(ck12_combined_dataset['earth-science']['10.4 Erosion and Deposition by Glaciers']['questions']['nonDiagramQuestions'])

val_counts=dr_proc_df['lesson_assigned_to'].value_counts()


# In[393]:

val_counts


# ## merge descriptions

# In[390]:

def make_description_entry(qdf_row):
    description = qdf_row[qdf_row.index == 'Description'].values[0]
    q_topic = qdf_row[qdf_row.index == 'lesson_assigned_to'].values[0]
    image_uri = qdf_row[qdf_row.index == 's3_uri'].values[0]
    image_name = qdf_row[qdf_row.index == 'diagram'].values[0]
    image_key = image_name.replace('.png', '')
    single_desc_dict = {
        "imageUri": image_uri,
        "imageName": image_name,
        "rawText": description,
        "processedText": description.encode('ascii', 'ignore').decode('utf-8')
        }
    if image_key not in build_descriptions[q_topic].keys():
        build_descriptions[q_topic].update({image_key: single_desc_dict})
    # I've found the longest description is usually best
    elif len(single_desc_dict['processedText']) > len(build_descriptions[q_topic][image_key]['processedText']):
        build_descriptions[q_topic].update({image_key: single_desc_dict})


# In[391]:

get_ipython().run_cell_magic('capture', '', "ai2_written_df_completed['lesson_assigned_to'] = ai2_written_df_completed['topic'].apply(lambda x: diagram_lesson_lookup[x])\nai2_written_df_completed['s3_uri'] = ai2_written_df_completed['Image Path'].apply(make_image_link)\nai2_written_df_completed = ai2_written_df_completed.dropna()\n\ndesc_df['topic'] = desc_df['diagram'].apply(lambda x: x.rsplit('_', maxsplit=1)[0])\ndesc_df['lesson_assigned_to'] = desc_df['topic'].apply(lambda x: diagram_lesson_lookup[x])\ndesc_df['s3_uri'] = desc_df['reference_id'].apply(make_image_link)\ndesc_df['Description'] = desc_df['01_write_description']             ")


# In[392]:

build_descriptions = defaultdict(dict)
_ = desc_df.apply(make_description_entry, axis=1)
_ = ai2_written_df_completed.apply(make_description_entry, axis=1)


# In[393]:

len(build_descriptions.keys())


# In[394]:

# this adds the descriptions to the combined dataset
for subject, lessons in ck12_combined_dataset.items():
    for l_name, lesson in lessons.items():
        if l_name in build_descriptions.keys():
            lesson['instructionalDiagrams'] = build_descriptions[l_name]
        else:
            lesson['instructionalDiagrams'] = {}


# In[ ]:




# In[130]:

dimages = []
for vals in build_descriptions.values():
    for img in vals.values():
        dimages.append(img['imageName'])


# In[133]:

len(set(dimages))


# In[114]:

len(dimages)


# In[311]:

desc_corpus = ''
for lesson, descriptions in build_descriptions.items():
    for dd in descriptions.values():
        desc_corpus += ' ' + dd['processedText']


# In[316]:

'the' in cached_sw


# In[335]:

d_corp = [lmtizer.lemmatize(word.lower()) for word in desc_corpus.split() if len(word) > 2 and word.lower() not in cached_sw]


# In[336]:

tb_freq_d = nltk.FreqDist(d_corp)
most_common_fb_words = tb_freq_d.most_common()


# ### hide

# In[398]:

pd.unique(desc_df['lesson_assigned_to']).shape


# In[399]:

build_descriptions.keys()


# In[400]:

# with open(output_dir + 'ck12_dataset_beta_v4.json', 'w') as f:
#     json.dump(ck12_combined_dataset, f, indent=4, sort_keys=True)


# In[401]:

# with open(output_dir + 'ck12_dataset_beta_v4.json', 'r') as f:
#     ck12_combined_dataset = json.load(f)


# ## Test spelling and grammar fixes

# ### code

# In[94]:

def check_mispelled(word):
    return word and word.isalpha() and not (edict.check(word) or anglo_edict.check(word) or edict.check(word[0].upper() + word[1:]))

def correct_spelling_error(misspelled_word, suggested_spellings):
    highest_ratio = 0
    closest_match = None
    for word in suggested_spellings:
        match_r = fuzz.ratio(misspelled_word, word)
        if match_r >= highest_ratio and (word[0] == misspelled_word[0] or not check_mispelled(word[0] + misspelled_word)) and len(misspelled_word) <= len(word):
            highest_ratio = match_r
            closest_match = word
            break
#     spell_changes[misspelled_word] = closest_match
    return closest_match

def apply_spelling_fix(orig_text):
    orig_text_tokens = orig_text.split()
    processed_tokens = []
    for token in orig_text_tokens:
        norm_token = token.lower()
        if len(norm_token) < 4:
            processed_tokens.append(token)
            continue
        if check_mispelled(norm_token):
            suggested_replacements = edict.suggest(token)
            replacement_text = correct_spelling_error(norm_token, suggested_replacements)
            if replacement_text:
                if norm_token[0].isupper():
                    replacement_text = upper(replacement_text[0]) + replaced_text[1:]
                processed_tokens.append(replacement_text)
            else:
                processed_tokens.append(token)
        else:
            processed_tokens.append(token)
    return ' '.join(processed_tokens)

def diff_corrected_text(orig_text, corrected_text):
    diff = dmp.diff_main(orig_text, corrected_text)
    return HTML(dmp.diff_prettyHtml(diff))

def specify_lesson_q_path(lesson):
    pass


# ### run

# In[95]:

dmp = diff_match_patch.diff_match_patch()


# In[96]:

ck12_spell_gramm_fix_test = deepcopy(ck12_combined_dataset)


# In[97]:

gram_checker = language_check.LanguageTool('en-US')
gram_checker.disabled = set(['SENT_START_CONJUNCTIVE_LINKING_ADVERB_COMMA', 'POSSESSIVE_APOSTROPHE', 'A_PLURAL'])
gram_checker.disable_spellchecking()

punc_set_space = set([',', ':', ';', '/"'])
punc_set_nospace = set(['-', '\'', '-', '?', '.', '!'])
question_enders = set(['.', '?', ':'])


# In[406]:

#check descriptions
spell_changes = {}
unaltered_text = []
replaced_text = []
for lesson in list(ck12_spell_gramm_fix_test['life-science'].values()):
    if lesson['instructionalDiagrams']:
        for diagram, description in lesson['instructionalDiagrams'].items():
            orig_text = description['processedText']
            spell_fixed_text = apply_spelling_fix(orig_text)
            for punc_char in punc_set_nospace:
                spell_fixed_text = spell_fixed_text.replace(' ' + punc_char + ' ' , punc_char)
            for punc_char in punc_set_space:
                spell_fixed_text = spell_fixed_text.replace(' ' + punc_char + ' ' , punc_char + ' ')
            gram_fixed = gram_checker.correct(spell_fixed_text)
            if gram_fixed != orig_text:
                unaltered_text.append(orig_text)
                replaced_text.append(gram_fixed)


# In[64]:

spell_changes = {}
unaltered_text = []
replaced_text = []
for lesson in list(ck12_spell_gramm_fix_test['life-science'].values()):
    if lesson['questions']['nonDiagramQuestions']:
        for diagram, description in lesson['questions']['diagramQuestions'].items():
            orig_text = description['beingAsked']['processedText']
            spell_fixed_text = apply_spelling_fix(orig_text)
            gram_fixed = gram_checker.correct(spell_fixed_text)
            for punc_char in punc_set_nospace:
                gram_fixed = gram_fixed.replace(' ' + punc_char + ' ' , punc_char)
                gram_fixed = gram_fixed.replace(' ' + punc_char, punc_char)
            for punc_char in punc_set_space:
                gram_fixed = gram_fixed.replace(' ' + punc_char + ' ' , punc_char + ' ')
            if gram_fixed[-1] not in question_enders:
                if gram_fixed.split()[0] in ['Identify', 'Name'] or '__' in gram_fixed:
                    gram_fixed += '.'
                else:
                    gram_fixed += '?'
            if gram_fixed != orig_text:
                unaltered_text.append(orig_text)
                replaced_text.append(gram_fixed)


# In[65]:

comp_text = list(zip(unaltered_text, replaced_text))

print(len(spell_changes))
print(len(comp_text))
# spell_changes


# In[66]:

rand_idx = np.random.randint(len(comp_text))
print(unaltered_text[rand_idx])
print()
print(replaced_text[rand_idx])
diff_corrected_text(*comp_text[rand_idx])


# ### hide

# In[410]:

# with open(output_dir + 'ck12_dataset_beta_v4.json', 'r') as f:
#     ck12_combined_dataset = json.load(f)


# # Topic key collisions

# ### old nested structure

# In[411]:

flexbook_ds.keys()


# In[412]:

build_website_lessons = [list(lesson.keys()) for lesson in lessons_ds.values()]
website_lessons= sorted([item for sublist in build_website_lessons for item in sublist])

build_flexbook_lessons = [list(lesson.keys()) for lesson in flexbook_ds.values()]
flexbook_lessons= [item for sublist in build_flexbook_lessons for item in sublist]
flexbook_lessons = sorted([lesson.split(maxsplit=1)[1].strip().lower() for lesson in flexbook_lessons])
fbls = set(flexbook_lessons)
wsls = set(website_lessons)


# In[413]:

n_dd_lessons = len(fbls.union(wsls))


# In[414]:

n_all_lessons = len(website_lessons + flexbook_lessons)


# In[415]:

n_all_lessons - n_dd_lessons


# In[416]:

len(flexbook_lessons)


# In[417]:

print(len(flexbook_lessons))
print(len(set(flexbook_lessons)))


# In[418]:

# pd.Series(flexbook_lessons).value_counts()


# In[419]:

combined_lesson_series = pd.Series(flexbook_lessons + website_lessons)
lessons_with_name_dupes = combined_lesson_series.value_counts()[combined_lesson_series.value_counts() > 1].index.tolist()


# In[420]:

# lessons_with_name_dupes


# In[421]:

print(len(website_lessons))
print(len(set(website_lessons)))


# In[424]:

len(set(website_lessons).union(set(flexbook_lessons)))


# In[ ]:

build_website_topics = [list(lesson.keys()) for lesson in lessons_ds.values()]
website_topics= sorted([item for sublist in build_website_lessons for item in sublist])

# build_flexbook_topics = [list(lesson.keys()) for lesson in flexbook_ds.values()]
# flexbook_topics= [item for sublist in build_flexbook_lessons for item in sublist]
# flexbook_lessons = sorted([lesson.split(maxsplit=1)[1].strip().lower() for lesson in flexbook_lessons])
# fbls = set(flexbook_lessons)
# wsls = set(website_lessons)


# ### with new flat dataset

# ### hide

# In[ ]:

topic_list = []
for lesson in complete_flat_ds:
#     for t_id, topic in lesson['topics'].items():
    topic_list.append(lesson['lessonName'].lower())

topic_tokens = {idx: [word for word in val.split() if word not in cached_sw] for idx, val in enumerate(topic_list)}

token_to_topic_index = defaultdict(list)
for k, vals in topic_tokens.items():
    for v in vals:
        token_to_topic_index[v].append(k)


# In[ ]:

collision_dict = {token: topic_idxs for token, topic_idxs in token_to_topic_index.items() if len(topic_idxs) < 50 and len(topic_idxs) > 10}
build_topics_collision_idxs = collision_dict.values()
topics_collision_idxs = [item for sublist in build_topics_collision_idxs for item in sublist]
topics_w_collisions_set = set(topics_collision_idxs)


# In[ ]:

for k, v in collision_dict.items():
    for topic_idx in v:
        print(topic_list[topic_idx])
    print()


# In[ ]:




# In[ ]:

lmtizer.stem('plants')


# ### code

# In[15]:

with open('./output_data_from_nbs/build_v5_5f.json', 'r') as f:
    complete_flat_ds = json.load(f)


# In[ ]:




# In[6]:

lesson_names = [lesson['lessonName'].lower() for lesson in complete_flat_ds]
lesson_idx_lookup = {v: k for k, v in enumerate(lesson_names)}


# In[7]:

lmtizer = WordNetLemmatizer()


# In[24]:

# ignore_tokens = ['introduction', 'form', 'component', 'destruction', 'destruction', 'components', 'characteristic', 'classification']
ignore_tokens = ['introduction', 'form', 'component', 'destruction', 'destruction', 'components', 'characteristic', 
                 'classification', 'design', 'effect', 'first', 'form', 'introduction', 'model', 'modern', 'non', 
                 'nature', 'process', 'science', 'thing', 'use']


# In[8]:

cached_sw = stopwords.words("english") + list(string.punctuation)
def find_meta_lessons(complete_ds):
    meta_lesson_id = 0
    meta_lesson_prefix = 'MT_'

    lesson_names = [lesson['lessonName'].lower() for lesson in complete_ds]
    lesson_tokens = {idx: filter_tokens(val) for idx, val in enumerate(lesson_names)}
    token_to_lesson_index = defaultdict(list)
    
    for k, vals in lesson_tokens.items():
        for v in vals:
            token_to_lesson_index[v].append(k)
            
    collision_dict = {token: topic_idxs for token, topic_idxs in token_to_lesson_index.items() if len(topic_idxs) < 20 and len(topic_idxs) > 1}
    return collision_dict

def pick_rarest_metalesson(metalessons):
    return sorted({k: meta_topic_sizes[k] for k in metalessons}.items(), key=lambda x: x[1])[0][0]

def filter_pos(token_list):
    tagged_tokens = nltk.pos_tag(token_list)
    filtered_tokes = [toke[0] for toke in tagged_tokens if toke[1] in ['NN', 'RB', 'NNS', 'JJ']]
    if filtered_tokes:
        return filtered_tokes
    else:
        return ['aaa_none']

def filter_tokens(tokens):
    return filter_pos([lmtizer.lemmatize(toke) for toke in tokens.split() if toke not in cached_sw])


# In[9]:

col_dict = find_meta_lessons(complete_flat_ds)

meta_topic_sizes = {v[0]: len(v[1])for v in sorted(col_dict.items(), key=lambda x: len(x[1]), reverse=True)}

token_to_lesson_index = defaultdict(list)
for k, vals in col_dict.items():
    for v in vals:
        token_to_lesson_index[v].append(k)


# In[10]:

meta_lesson_assignments = {}
for lesson in lesson_names:
    associated_metalessons = token_to_lesson_index[lesson_idx_lookup[lesson]]
    if associated_metalessons:
        meta_lesson_assignments[lesson] = pick_rarest_metalesson(associated_metalessons)
    else:
        meta_lesson_assignments[lesson] = ' '.join(filter_tokens(lesson))

meta_lesson_string_ids = set(meta_lesson_assignments.values())
meta_lesson_id_lookup =  {val: idx for idx, val in enumerate(meta_lesson_string_ids)}
meta_lesson_id_assignments = {k: meta_lesson_id_lookup[v] for k, v in meta_lesson_assignments.items()}


# In[11]:

# print(len(token_to_lesson_index.keys()))

# match_seq = pd.Series([len(tokens) for tokens in token_to_lesson_index.values()])
# match_seq.value_counts()


# In[941]:

linked_lessons = defaultdict(list)
for k, v in meta_lesson_assignments.items():
    linked_lessons[v].append(k)
linked_lessons = dict(linked_lessons)


# In[929]:

meta_lesson_assignments


# In[934]:

meta_lesson_id_assignments


# In[943]:

len(set(meta_lesson_assignments.values()))


# In[434]:

# %page linked_lessons


# ### hide

# In[ ]:

ttl = ['respiration', 'reproductive', 'animal']


# In[ ]:

filter_pos(ttl)


# In[ ]:

nltk.pos_tag([' current', 'test'])


# # Refinements to make

# ### todo

# get viz working
# return annotations to ds
# fix lessons missing topics

# ### file i/o

# In[472]:

write_file('build_v6.json', complete_flat_ds)


# In[445]:

# write_file('build_v5_prior_to_refinement.json', ck12_combined_dataset)


# In[77]:

with open('../dataset_releases/data_release_beta5/inter/tqa_dataset_beta5.json', 'r') as f:
    prev_v5 = json.load(f)


# In[ ]:

# with open('build_v5.pkl', 'wb') as f:
#     pickle.dump(ck12_combined_dataset ,f)


# In[55]:

with open('build_v5.pkl', 'rb') as f:
    ck12_combined_dataset = pickle.load(f)


# In[395]:

# with open('output_data_from_nbs/build_v5_prior_to_refinement.json', 'w') as f:
#       json.dump(ck12_combined_dataset, f)


# In[299]:

with open('output_data_from_nbs/build_v5_prior_to_refinement.json', 'r') as f:
     ck12_latest = json.load(f)


# ### code

# In[396]:

#print(topics_to_remove) #specicied in match topics section above, explictly set here

structural_topics = ['Summary', 'Review', 'References', 'Explore More', 'Lesson Summary', 'Lesson Objectives', 'Points to Consider', 'Introduction',
                    'Recall', 'Apply Concepts', 'Think Critically', 'Resources', 'Explore More II', 'Explore More I', 'Explore More III']

vocab_topics = ['Lesson Vocabulary', 'Vocabulary']


# In[913]:

def iterate_over_all_material(complete_ds, apply_function):
    lesson_returns = []
    for subject, lessons in sorted(complete_ds.items(), key= lambda x: x[0]):
        for lesson_name, lesson_content in sorted(lessons.items(), key= lambda x: x[0]):
            response = apply_function(lesson_name, lesson_content)
            if response:
                lesson_returns.append(response)
    return lesson_returns
        

def apply_fixes(lesson_name, lesson_content):
    struct_content, vocab_content = iterate_over_text(lesson_content['topics'], lesson_name)
    lesson_content['adjunctTopics'] = struct_content
    lesson_content['adjunctTopics']['Vocabulary'] = vocab_content
    iterate_over_text_questions(lesson_content['questions']['nonDiagramQuestions'])
    if lesson_content['instructionalDiagrams']:
        if not lesson_content['questions']['diagramQuestions']:
            print(lesson_name + ' missing questions')
        iterate_over_diagram_questions(lesson_content['questions']['diagramQuestions'])
        iterate_over_diagram_descriptions(lesson_content['instructionalDiagrams'])
        
def iterate_over_text(topic_sections, lesson_name=None):
    structural_content = {}
    vocab_section = {}
    topics_to_remove = []
    for topic, content in sorted(topic_sections.items(), key= lambda x: x[1]['orderID']):
        if content['content']['figures']:
            iterate_over_textbook_figs(content['content']['figures'], lesson_name)
        if topic in vocab_topics:
            vocab_section.update(add_defintions_to_vocab(content))
            topics_to_remove.append(topic)
        elif topic in structural_topics:
            structural_content[topic] = content
            topics_to_remove.append(topic)
    for topic in topics_to_remove:
        topic_sections.pop(topic)
    return structural_content, vocab_section
    

def iterate_over_text_questions(text_questions):
    qs_to_del = []
    replace_local_ids_w_global(text_questions, 'text', True, 'id')
    for question_id, question_content in text_questions.items():
        if not question_content['correctAnswer']:
            print(question_content['globalID'])
            qs_to_del.append(question_content['globalID'])
            continue
        check_type_assignments(question_content)
        change_q_type_key(question_content)
        check_raw_text_present(question_content)
    for qid in qs_to_del:
        del text_questions[qid]
        
def iterate_over_diagram_questions(diagram_questions):
    for qid, question in diagram_questions.items():
        replace_uri_with_path(question, 'question_images')
        orig_question= question['beingAsked']['processedText']
        fixed_question = apply_spelling_and_grammar_fixes(orig_question)
        if fixed_question:
            question['beingAsked']['processedText'] = fixed_question
        if detect_abc_question(question):
            standardize_abc_question(question)
        change_q_type_key(question)
    replace_local_ids_w_global(diagram_questions, 'diagram', True, 'id')

def iterate_over_diagram_descriptions(diagram_descriptions, description_path_prefix=None):
    for diagram_name, diagram_content in diagram_descriptions.items():
        replace_uri_with_path(diagram_content, 'teaching_images')
        orig_description = diagram_content['processedText']
        fixed_description = apply_spelling_and_grammar_fixes(orig_description)
        if fixed_description:
            diagram_content['processedText'] = fixed_description
    replace_local_ids_w_global(diagram_descriptions, 'description', True, z_pad=4)

            
def add_defintions_to_vocab(vocab_section): #done
    lesson_vocab = {}
    for word in vocab_section['content']['text'].split('\n'):
        if word in flexbook_glossary.keys():
            lesson_vocab[word] = flexbook_glossary[word]
        elif word:
            lesson_vocab[word] = ''
    return lesson_vocab

def add_global_ids(data_object, object_type, zero_padding=6): #done
    id_prefix = {'text': 'NDQ_', 'diagram': 'DQ_', 'lesson': 'L_', 'description': 'DD_', 'topics': 'T_'}
    global_ids_counters[object_type] += 1
    data_object['globalID'] = id_prefix[object_type] + str(global_ids_counters[object_type]).zfill(zero_padding)

def detect_abc_question(question):
    image_name = question['imageName']
    image_number = int(image_name.split('_')[-1].replace('.png', ''))
    return image_number >= 10000

def standardize_abc_question(question): #done
    question['imagePath'] = question['imagePath'].replace('question_images', 'abc_question_images')
    question['correctAnswer']['processedText'] = question['correctAnswer']['processedText'].upper()
    for ac in question['answerChoices'].values():
        ac['processedText'] = ac['processedText'].upper()
    
def iterate_over_textbook_figs(figure_content, lesson_n): # done
    for figure in sorted(figure_content, key= lambda x: x['image_uri'].split('/')[-1]):
        replace_uri_with_path(figure, 'textbook_images', lesson_n)
        
def replace_uri_with_path(image_content, path_prefix, lesson_name=None): # done
    image_key_str = 'imageUri'
    if 'image_uri' in image_content.copy().keys():
        image_key_str = 'image_uri'
    image_name = image_content[image_key_str].split('/')[-1]
    image_number = image_name.split('_')[-1].split('.')[0]
    del image_content[image_key_str]
    if lesson_name:
        global_ids_counters['image'] += 1
        prev_image_name = deepcopy(image_name)
        image_name =  rename_lesson(lesson_name).replace(' ', '_') + '_' + str(global_ids_counters['image']) + '.png'
        record_image_name_changes[os.path.join(path_prefix, prev_image_name)] = os.path.join(path_prefix, image_name)
    image_content['imagePath'] = os.path.join(path_prefix, image_name)
      
def apply_spelling_and_grammar_fixes(orig_text): #done
    spell_fixed_text = apply_spelling_fix(orig_text)
    for punc_char in punc_set_nospace:
        spell_fixed_text = spell_fixed_text.replace(' ' + punc_char + ' ' , punc_char)
    for punc_char in punc_set_space:
        spell_fixed_text = spell_fixed_text.replace(' ' + punc_char + ' ' , punc_char + ' ')
    gram_fixed = gram_checker.correct(spell_fixed_text)
    if gram_fixed != orig_text:
        return gram_fixed
    else:
        return None
    
def standardize_true_false_question(question):
    replacement_text = {'true': 'a', 'false': 'b'}
    if question['correctAnswer']['processedText'] not in (['true', 'false']):
        print('p', question['correctAnswer']['processedText'])
    if question['correctAnswer']['rawText'] not in (['true', 'false']):
        print('r', question['correctAnswer']['rawText'])
    question['correctAnswer']['processedText'] = replacement_text[question['correctAnswer']['processedText']]

def check_raw_text_present(question):
    if 'rawText' not in question['correctAnswer'].keys():
        question['correctAnswer']['rawText'] = question['correctAnswer']['processedText']

def check_type_assignments(question):
    if question['type'] == "True or False":
        if question['correctAnswer']['processedText'] not in ['true', 'false']:
            print(question['correctAnswer']['processedText'])
            if '__' in question['beingAsked']['processedText']:
                question['type'] = 'Fill in the Blank'
                del question['answerChoices']

def replace_local_ids_w_global(data_unit, unit_type, keys_too=False, id_key='', z_pad=6):
    if id_key:
        objs_to_iterate = sorted(data_unit.items(), key=lambda x: x[1][id_key])
    else:
        objs_to_iterate = data_unit.items()
    for k, v in objs_to_iterate:
        if(unit_type == 'description'):
            global_ids_counters['dds_count'] += 1 
        add_global_ids(v, unit_type, zero_padding=z_pad)
        if id_key:
            v.pop(id_key)
    for k, v in objs_to_iterate:
        if keys_too:
            data_unit[v['globalID']] = data_unit.pop(k)
            if unit_type == 'topics':
                name_field = unit_type[:-1] + 'Name'
                v[name_field] = k
    pass

def rename_lesson(lesson_name):
    if lesson_name[0].isdigit():
        return lesson_name.split(maxsplit=1)[1].strip().lower()
    else:
        return lesson_name

def flatten_complete_ds(lesson_name, lesson_content):
    lesson_content['lessonName'] =lesson_name
    add_global_ids(lesson_content, 'lesson', 4)
    lesson_content['lessonName'] = rename_lesson(lesson_name)
    obj_key = 'topics'
    replace_local_ids_w_global(lesson_content[obj_key], obj_key, True, 'orderID', 4)
    return lesson_content

def add_metalesson_ids(lesson, meta_lesson_assignments):
    lesson['metaLessonID'] = 'ML_' + str(meta_lesson_assignments[lesson['lessonName']]).zfill(4)
    
def change_q_type_key(question):
    multi_choice_types = ["True or False", "Multiple Choice", "Matching"]
    if question['type'] == 'Diagram Multiple Choice':
        question['questionType'] = 'Diagram Multiple Choice'
    elif question['type'] in multi_choice_types:
        question['questionSubType'] = question['type']
        question['questionType'] = "Multiple Choice"
    else:
        question['questionSubType'] = question['type']
        question['questionType'] = "Direct Answer"
    question.pop('type')


# In[1017]:

def standardize_answer_choices(question):
    if not question['answerChoices']:
        return None
    correct_answer_text = question['correctAnswer']['processedText']
    if question['questionSubType'] == 'True or False':
        if 'true' in correct_answer_text:
            question['correctAnswer']['processedText'] = 'a'
            return question
        elif 'false' in correct_answer_text:
            question['correctAnswer']['processedText'] = 'b'
            return question
        else:
            return None
    elif correct_answer_text.replace('.', '') in list(string.ascii_letters)[:9]:
        question['correctAnswer']['processedText'] = correct_answer_text.replace('.', '')
        return question
    else:
        for ac_id, ac in question['answerChoices'].items():
            ac_text = ac['processedText']
            if ac_text in correct_answer_text:  
                question['correctAnswer']['processedText'] = ac_id
                return question
        else:
            print(correct_answer_text)
            print(question['answerChoices'])
            
def standardize_answer_choices_diagram(question):
    correct_answer_text = question['correctAnswer']['processedText']
    if len(correct_answer_text) > 300:
        return
    if correct_answer_text.replace('.', '') in list(string.ascii_letters)[:3]:
        question['correctAnswer']['processedText'] = correct_answer_text.replace('.', '')
        return question
    else:
        for ac_id, ac in question['answerChoices'].items():
            ac_text = ac['processedText']
            if ac_text in correct_answer_text:  
                question['correctAnswer']['processedText'] = ac_id
                return question
        else:
            return
            print(question['correctAnswer'])
            print()
            print(question['answerChoices'])


# In[1018]:

complete_flat_ds_kc_fix = deepcopy(complete_flat_ds)


# In[1019]:

new_complete_flat_ds_kc_fix = []
for lesson in complete_flat_ds_kc_fix:
    new_lesson = deepcopy(lesson)
    for qid, question in new_lesson['questions']['nonDiagramQuestions'].copy().items():
        if question['questionType'] == 'Multiple Choice':
            new_question = standardize_answer_choices(question)
            if new_question:
                new_lesson['questions']['nonDiagramQuestions'][qid] = new_question
            else:
                del new_lesson['questions']['nonDiagramQuestions'][qid]
    new_complete_flat_ds_kc_fix.append(new_lesson)


# In[1020]:

new_w_dw_complete_flat_ds_kc_all_fix = []
for lesson in new_complete_flat_ds_kc_fix:
    new_lesson = deepcopy(lesson)
    if new_lesson['questions']['diagramQuestions']:
        for qid, question in new_lesson['questions']['diagramQuestions'].copy().items():
            new_question = standardize_answer_choices_diagram(question)
            if new_question:
                new_lesson['questions']['diagramQuestions'][qid] = new_question
            else:
                del new_lesson['questions']['diagramQuestions'][qid]
        new_w_dw_complete_flat_ds_kc_all_fix.append(new_lesson)
    else:
        new_w_dw_complete_flat_ds_kc_all_fix.append(new_lesson)


# In[1021]:

ca_series = []
for lesson in new_w_dw_complete_flat_ds_kc_all_fix:
    for qid, question in lesson['questions']['nonDiagramQuestions'].items():
        if question['questionType'] == 'Multiple Choice':
            ca_series.append(question['correctAnswer']['processedText'])


# In[1022]:

dq_ca_series = []
for lesson in new_w_dw_complete_flat_ds_kc_all_fix:
    for qid, question in lesson['questions']['diagramQuestions'].items():
        dq_ca_series.append(question['correctAnswer']['processedText'])


# In[1023]:

pd.Series(dq_ca_series).value_counts()


# In[1024]:

pd.Series(ca_series).value_counts()


# In[1025]:

non_diagram_questions = [list(dict_key_extract('nonDiagramQuestions', lesson)) for lesson in new_w_dw_complete_flat_ds_kc_all_fix]

nd_questions = {}
for lesson in non_diagram_questions:
    for lesson_questions in lesson:
        for question_id, question in lesson_questions.items():
            nd_questions[question_id] = question


# In[1026]:

len(nd_questions)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[1028]:

write_file('beta7_5_final_fixed.json', new_w_dw_complete_flat_ds_kc_all_fix)


# ### run

# In[ ]:

ck12_combined_dataset


# In[914]:

# ck12_combined_dataset = 
full_test_ds = deepcopy(ck12_combined_dataset)
test_vals= {k: v  for k, v in list(full_test_ds['earth-science'].items())[:20]}
buid_test_cds = {}
buid_test_cds['earth-science'] = test_vals
test_cds = deepcopy(buid_test_cds)


# In[915]:

get_ipython().run_cell_magic('time', '', "dq_gid = 0\nndq_gid = 0\nl_id = 0\nt_id = 0\ndd_id = 0\npd_image = 20000\nn_ddds_seen = 0\n\n\nrecord_image_name_changes = {}\nglobal_ids_counters = {'text': ndq_gid, 'diagram': ndq_gid, 'lesson': l_id, 'topics': t_id, 'description': dd_id, 'image':pd_image , 'dds_count': n_ddds_seen}\n\ndataset_build_3p5 = full_test_ds\n_ = iterate_over_all_material(dataset_build_3p5, apply_fixes)\ncomplete_flat_ds = iterate_over_all_material(dataset_build_3p5, flatten_complete_ds)\n\nfor lesson in complete_flat_ds:\n    add_metalesson_ids(lesson, meta_lesson_id_assignments)")


# In[918]:

complete_flat_ds[2]['instructionalDiagrams']


# In[945]:

def dict_key_extract(key, var):
    if hasattr(var, 'items'):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in dict_key_extract(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in dict_key_extract(key, d):
                        yield result


# In[946]:

non_diagram_questions = [list(dict_key_extract('nonDiagramQuestions', lesson)) for lesson in complete_flat_ds]

nd_questions = {}
for lesson in non_diagram_questions:
    for lesson_questions in lesson:
        for question_id, question in lesson_questions.items():
            nd_questions[question_id] = question


# In[948]:

len(nd_questions)


# In[550]:

len(record_image_name_changes)


# In[ ]:

# [lesson for lesson in complete_flat_ds if lesson['lessonName'] == 'soils'][0]['questions']['diagramQuestions']


# In[473]:

global_ids_counters['dds_count']


# In[181]:

complete_flat_ds[3]['questions']


# ### hide

# In[ ]:




# In[370]:

for lessons in list(ck12_combined_dataset.values())[:20]:
    for lid, lesson in lessons.items():
        for qid, question in lesson['questions']['nonDiagramQuestions'].items():
            if question['type'] == 'True or False':
                if 'processedText' not in question['correctAnswer'].keys():
                    pass
                elif 'rawText' not in question['correctAnswer'].keys():
                    pass
                elif question['correctAnswer']['rawText'].split('.')[-1].strip() not in (['true', 'false']):
                    print('p', question['correctAnswer'])


# In[373]:

for lessons in list(ck12_combined_dataset.values())[:20]:
    for lid, lesson in lessons.items():
        for qid, question in lesson['questions']['nonDiagramQuestions'].items():
            if '___' in question['beingAsked']['processedText']:
                if question['type'] == 'True or False':
                    if question['correctAnswer']['processedText'] not in ['true', 'false']:
                        print(question['correctAnswer'])


# In[560]:

original_image_uris = []
for lessons in list(ck12_combined_dataset.values()):
    for lid, lesson in lessons.items():
        for topic_name, topic in lesson['topics'].items():
            for figure in topic['content']['figures']:
                original_image_uris.append(figure['image_uri'])


# In[562]:

len(record_image_name_changes_name_only)


# In[561]:

len(original_image_uris)


# In[431]:

for lessons in complete_flat_ds:
    for qid, question in lesson['questions']['nonDiagramQuestions'].items():
        if 'processedText' not in question['correctAnswer'].keys():
            print('question')
        elif 'rawText' not in question['correctAnswer'].keys():
            pass


# In[ ]:




# In[207]:

complete_flat_ds[3]['instructionalDiagrams'].keys()

# _ = iterate_over_all_material(full_test_ds, iterate_over_diagram_questions)

prob_image = 'tectonic_plates_9275.png'

prob_lesson = []
for lesson in complete_flat_ds:
    if lesson['lessonName'] == 'theory of plate tectonics':
        prob_lesson.append(lesson)

prob_nest_lesson = {'es': {}}
for subject, lessons in sorted(ck12_combined_dataset.items(), key= lambda x: x[0]):
        for lesson_name, lesson_content in sorted(lessons.items(), key= lambda x: x[0]):
            if lesson_name == "6.4 Theory of Plate Tectonics":
                prob_nest_lesson['es'][lesson_name] = lesson_content

prob_nest_lesson

prob_lesson[0]['instructionalDiagrams']

id_seq = []
for lesson in complete_flat_ds:
    for val in lesson['instructionalDiagrams'].values():
        if 'globalID' in val.keys():
            id_seq.append(val['globalID'])
#         print(val['imageName'])
#         if val['imageName'] == 'optics_reflection_9179.png':
#             print(val.keys())

sorted_ids = sorted([int(idn.split('_')[-1]) for idn in id_seq])

sorted_ids[-1]

complete_flat_ds = iterate_over_all_material(full_test_ds, flatten_complete_ds)

new_l_set = [len(lesson['questions']['nonDiagramQuestions']) for lesson in complete_flat_ds]
# old_l_set = [len(lesson['questions']['nonDiagramQuestions'])  for lesson in prev_v5]

pd.Series(new_l_set).sum()


# In[ ]:

# complete_flat_ds[0].keys()
# complete_flat_ds[0]['questions']
# sorted(complete_flat_ds[9]['instructionalDiagrams'].items(), key=lambda x: x[1]['globalID'])
# sorted(complete_flat_ds[0]['questions']['nonDiagramQuestions'].items(), key=lambda x: x[1]['globalID'])
# complete_flat_ds[0]['adjunctTopics']
# list(complete_flat_ds[0]['topics'].items())[5]
# complete_flat_ds[9]


# In[ ]:

t_ids_seen = []
q_ids_seen = []
for lesson in list(list(test_cds.values())[0].values()):
    for que in lesson['questions']['nonDiagramQuestions'].values():
        t_ids_seen.append(que['globalID'])  
for lesson in list(list(test_cds.values())[0].values()):
    for que in lesson['questions']['diagramQuestions'].values():
        q_ids_seen.append(que['globalID'])  


# In[548]:

# list(test_cds['earth-science'].values())[0]['topics']


# In[ ]:

# list(test_cds['earth-science'].values())[0]['adjunctTopics']


# In[ ]:

#             print(lesson_content['topics'].keys())


# # renaming diagram image files

# In[482]:

image_base = '/Users/schwenk/wrk/stb/dataset_releases/data_release_beta6/'


# In[124]:

len(missing)


# In[474]:

len(record_image_name_changes)


# In[162]:

d_image = 'textbook_images/biotechnology_in_agriculture_21732.png'


# In[136]:

# list(record_image_name_changes.values())


# In[477]:

image_base


# In[483]:

missing = []
for old_name, new_name in list(record_image_name_changes.items()):
    try:
        old_path = os.path.join(image_base, old_name)
        new_path = os.path.join(image_base, new_name)
        if not os.path.exists(new_path):
            print(new_path)
#         os.rename(old_path, new_path)
    except FileNotFoundError:
        missing.append(old_path)


# In[521]:

with open('/Users/schwenk/Downloads/textbook_image_old_file_names.json', 'r') as f:
    list_images_ani = json.load(f)


# In[526]:

len(list_images_ani)


# In[ ]:




# In[538]:

missing_image_example = '1.1_Scientific_Ways_of_Thinking_Scientific_Theories_and_Scientific_Laws_0013_fig_1.2.png'


# In[539]:

record_image_name_changes_name_only[missing_image_example]


# In[537]:

sorted(record_image_name_changes_name_only.keys())


# In[564]:

text_only_starting_names = [name.split('/')[-1] for name in original_image_uris]


# In[565]:

text_only_starting_names


# In[1032]:

missing_image_lookup


# In[634]:

lookup_for_ani = {}
missing_image_list = []
for image_n in list_images_ani:
    if image_n not in record_image_name_changes_name_only:
        missing_image_list.append(image_n)
        new_name = missing_image_lookup[image_n]
        if new_name == 'MISSING':
            lookup_for_ani[image_n] = new_name
        else:
            lookup_for_ani[image_n] = record_image_name_changes_name_only[new_name]
    else:
        lookup_for_ani[image_n] = record_image_name_changes_name_only[image_n]


# In[1029]:

len(lookup_for_ani)


# In[636]:

lookup_for_ani


# In[642]:

missing = []
for old_name, new_name in list(lookup_for_ani.items()):
    try:
        old_path = os.path.join(image_base, old_name)
        new_path = os.path.join(image_base, 'textbook_images', new_name)
        if not os.path.exists(new_path):
            print(new_path)
#         os.rename(old_path, new_path)
    except FileNotFoundError:
        missing.append(old_path)


# In[631]:

len(lookup_for_ani)


# In[625]:

len(missing_image_list)


# In[ ]:




# In[581]:

len(missing_image_list)


# In[598]:

missing_image_list[0]


# In[595]:

words_to_check = missing_image_list[4].split('_')
words_to_check


# In[615]:

len(missing_image_list)


# In[617]:

missing_image_lookup = {}


# In[620]:

for missing_image in missing_image_list:
    for image in record_image_name_changes_name_only.keys(): 
        if missing_image in image:
            missing_image_lookup[missing_image] = image
            break
        else:
            missing_image_lookup[missing_image] = 'MISSING'


# In[639]:

with open('textbook_image_lookup.pkl' , 'wb')as f:
    pickle.dump(lookup_for_ani, f)


# In[619]:

len(missing_image_lookup)


# In[599]:

'Atmosphere_Air_Pressure_Zones_0004_fig_1.1.png' in 'Circulation_in_the_Atmosphere_Air_Pressure_Zones_0004_fig_1.1.png'


# In[609]:

[lesson['lessonName'] for lesson in complete_flat_ds if 'atmosphere' in lesson['lessonName']]


# In[601]:

sorted(list([k for k in record_image_name_changes_name_only.keys() if k[0] == 'C']))


# In[594]:

missing_image_list[4]


# In[568]:

len(missing_image_list)


# In[504]:

old_original_image_names = get_ipython().getoutput('ls /Users/schwenk/wrk/stb/flexbook_image_extraction/pdf_pages/single_page_pdfs/figures_first_run')
new_original_image_names = get_ipython().getoutput('ls /Users/schwenk/wrk/stb/flexbook_image_extraction/pdf_pages/single_page_pdfs/figures')


# In[505]:

new_original_image_names


# In[532]:

record_image_name_changes_name_only[ '9.3_Carbon_and_Living_Things_Structure_of_Nucleic_Acids_0215_fig_9.23.png']


# In[527]:

len(record_image_name_changes)


# In[506]:

old_original_image_names


# In[949]:

# set(new_original_image_names).difference(set(old_original_image_names))


# In[513]:

len(set(old_original_image_name).intersection(set(new_original_image_name)))


# In[488]:

images_present = get_ipython().getoutput('ls /Users/schwenk/wrk/stb/dataset_releases/data_release_beta6/textbook_images')


# In[492]:

renamed_images = [new_name.split('/')[-1] for new_name in record_image_name_changes.values()]
for image_n in images_present:
    if image_n not in renamed_images:
        print(image_n)


# In[950]:

# for old_name, new_name in list(record_image_name_changes.items()):
#     new_name.split('/')[-1]


# In[499]:

with open('textbook_image_renaming_lookup.pkl', 'wb') as f:
    pickle.dump(record_image_name_changes_name_only, f)


# In[232]:

missing


# In[496]:

record_image_name_changes_name_only = {k.split('/')[-1]:v.split('/')[-1] for k,v in record_image_name_changes.items()}


# In[164]:

d_image in record_image_name_changes.keys()


# In[ ]:




# # End

# In[ ]:

# flexbook_glossary.keys()


# In[ ]:

# print(list(diagram_lesson_lookup.values()))


# In[ ]:

test_lesson = ck12_combined_dataset['earth-science']['24.1 Planet Earth']


# In[ ]:

# pprint.pprint(test_lesson['instructionalDiagrams'])


# In[ ]:

list(test_lesson['instructionalDiagrams'].values())[0]['processedText']


# In[ ]:

test_lesson['topics'].keys()


# In[ ]:

test_vocab = test_lesson['topics']['Vocabulary']['content']['text'].split('\n')


# In[ ]:

for word in test_vocab:
    if word in flexbook_glossary:
        print(flexbook_glossary[word])
        print()


# In[ ]:

# write_file('ck12_v4_5.json', ck12_combined_dataset, 'experimental_output')


# In[ ]:



