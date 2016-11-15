
# coding: utf-8

# # Table of Contents
# * [Load data](#Load-data)
# 	* [paths and reads](#paths-and-reads)
# 	* [extracting specifics](#extracting-specifics)
# * [Basic question types](#Basic-question-types)
# * [Non-Diagram Questions](#Non-Diagram-Questions)
# 	* [bigram/trigram question phrases](#bigram/trigram-question-phrases)
# 	* [question length](#question-length)
# 	* [looking at all/some/none of the above questions](#looking-at-all/some/none-of-the-above-questions)
# 	* [Looking for answer text in associated lessons](#Looking-for-answer-text-in-associated-lessons)
# 	* [hide](#hide)
# * [Templates](#Templates)
# 

# In[103]:

get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\nimport numpy as np\nimport pandas as pd\nimport scipy.stats as st\nimport itertools\nimport math\nfrom collections import Counter, defaultdict, OrderedDict\n\nimport cv2\nimport pprint\nimport pickle\nimport json\nimport requests\nimport io\nimport sys\nimport os\nfrom binascii import b2a_hex\nimport base64\nfrom wand.image import Image as WImage\nfrom IPython.display import display\nimport PIL.Image as Image\nfrom copy import deepcopy\nimport glob\n\nimport json\nimport enchant\nimport pickle\nimport glob\n\nimport nltk\nimport string\nfrom nltk.corpus import stopwords\nfrom nltk.tokenize import wordpunct_tokenize\nfrom nltk.tokenize import sent_tokenize\nfrom nltk.collocations import BigramCollocationFinder\nfrom nltk.collocations import TrigramCollocationFinder\nfrom nltk.stem.wordnet import WordNetLemmatizer\n\nimport language_check\n\nimport jsonschema\n# from pdfextraction.ck12_new_schema import ck12_schema\nfrom tqa_utils import Evaluator\nfrom tqa_utils.render_html import render_sample_question_and_lesson\nfrom analysis_utils import *')


# In[2]:

get_ipython().run_cell_magic('capture', '', 'import matplotlib as mpl\nmpl.use("Agg")\nimport matplotlib.pylab as plt\n%matplotlib inline\n%load_ext base16_mplrc\n%base16_mplrc light bright\nplt.rcParams[\'grid.linewidth\'] = 0\nplt.rcParams[\'figure.figsize\'] = (16.0, 10.0)\n# import seaborn as sns\nfrom IPython.display import set_matplotlib_formats\nset_matplotlib_formats(\'png\', \'pdf\')')


# # Load data

# ## paths and reads

# In[3]:

dataset_root_dir = '/Users/schwenk/wrk/stb/dataset_releases/data_release_beta6/'
file_name = 'tqa_dataset_beta7_5.json'
data_file =  dataset_root_dir + file_name

quest_evaluator = Evaluator(data_file)


# In[4]:

with open(os.path.join(dataset_root_dir, file_name), 'r') as f:
    ck12_combined_dataset_raw = json.load(f)
ck12_combined_dataset = deepcopy(ck12_combined_dataset_raw)


# ## extracting specifics

# In[5]:

lesson_text = collect_filtered_lesson_text(ck12_combined_dataset, True, True)


# In[6]:

sum([len(sent_tokenize(text)) for text in lesson_text.values()])


# In[7]:

count_textbook_images(ck12_combined_dataset)


# In[8]:

all_ndqs = quest_evaluator.build_question_lookup(by_type=True)['nonDiagramQuestions']
mc_questions = quest_evaluator.build_questions_by_subtype(all_ndqs)['Multiple Choice']
all_diagram_qs = quest_evaluator.build_question_lookup(by_type=True)['diagramQuestions']


# # Basic question types

# In[452]:

combined_topics = [lesson['lessonName'] for lesson in ck12_combined_dataset]
topic_series = pd.Series(combined_topics).value_counts()


# In[453]:

# topic_series[:20]
len(combined_topics)


# In[454]:

q_types = []
for lesson in ck12_combined_dataset:
    for question in lesson['questions']['nonDiagramQuestions'].values():
        q_types.append(question['questionSubType'])
question_counts = pd.Series(q_types).value_counts()
print('total number of questions = ' + str(question_counts.sum()))
question_counts


# In[455]:

_ = question_counts.plot(kind="barh")
plt.title('Question Format Distribution', fontsize=50, verticalalignment='bottom', color = b16_colors.b)
plt.ylabel("Question type", fontsize=30, labelpad=10, color = b16_colors.b)
plt.xlabel("Number of unique questions", fontsize=30, labelpad=10, color = b16_colors.b)
plt.tick_params(axis='x', which='major', labelsize=20)
plt.tick_params(axis='y', which='major', labelsize=20)


# In[935]:

13693 + 12567


# In[933]:

question_counts


# In[10]:

usable_questions = question_counts[:2].append(question_counts[3:4])


# In[ ]:




# In[11]:

sum(usable_questions)


# # Non-Diagram Questions

# ## bigram/trigram question phrases

# In[555]:

cached_sw = ['?', 'the', 'a', 's', "'", 'this', '.', ',', '_____', '__________', 'an']


# In[556]:

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()


# ### first pass

# In[557]:

ndq_corpus = ''
for lesson in ck12_combined_dataset:
    for qid, quest in lesson['questions']['nonDiagramQuestions'].items():
        ndq_corpus += ' ' + quest['beingAsked']['processedText'].replace('(', ' ').replace(')', ' ')
        
q_tokens = wordpunct_tokenize(ndq_corpus)
normalized_tokens = [toke.strip().lower().encode('ascii', 'ignore').decode() for toke in q_tokens if toke.strip().lower().encode('ascii', 'ignore').decode() not in cached_sw]
    
qw_freq_d = nltk.FreqDist(normalized_tokens)
most_common_qw = qw_freq_d.most_common(20)


# In[558]:

most_common_qw


# In[10]:

print([word[0] for word in most_common_qw])


# In[465]:

phrase_bi_finder = BigramCollocationFinder.from_words(normalized_tokens)
phrase_bi_finder.apply_freq_filter(75) 
# phrase_bi_finder.nbest(bigram_measures.pmi, 10)
most_common_bigrams = []

for k,v in sorted(phrase_bi_finder.ngram_fd.items(), key=lambda x:x[1], reverse=True)[:15]:
    most_common_bigrams.append((' '.join(k), v))


# In[466]:

sorted(phrase_bi_finder.ngram_fd.items(), key=lambda x:x[1], reverse=True)[:15]


# In[467]:

most_common_bigrams


# In[470]:

phrase_tri_finder = TrigramCollocationFinder.from_words(normalized_tokens)
phrase_tri_finder.apply_freq_filter(30) 
# phrase_tri_finder.nbest(trigram_measures.pmi, 20)  

most_common_tri = []
three_word_qtypes = []
for k,v in sorted(phrase_tri_finder.ngram_fd.items(), key=lambda x:x[1], reverse=True)[:9]:
    most_common_tri.append((' '.join(k), v))
    three_word_qtypes.append(' '.join(k))


# In[45]:

high_values_lessons = q_type_df[q_type_df.apply(lambda x: sum(x), axis=1) > 42] ## three word
high_values_lessons = q_type_df[q_type_df.apply(lambda x: sum(x), axis=1) > 375] ## two word


# In[47]:

fig, ax = plt.subplots(figsize=(8, 20))
_ = sns.heatmap(high_values_lessons, cmap='viridis', robust=False)
_ = plt.xticks(rotation=75) 


# In[468]:

sorted_common = sorted(most_common_bigrams, key=lambda x: x[1], reverse=True) 

qtype = list(zip(*sorted_common))[0]
q_num = list(zip(*sorted_common))[1]
x_pos = np.arange(len(qtype))

plot_to_save = plt.bar(x_pos, q_num,align='center')
plt.xticks(x_pos, qtype, rotation=40) 

fig_labels = {
    'fig_title': 'Common Question Bigrams', 
    'x_label': '',
    'y_label': 'Number of Questions'
}

# weights = np.ones_like(array_to_hist)/len(array_to_hist)
# tick_labels = plot_to_save.get_yticks()
# plot_to_save.set_yticklabels(['{:2.2f}%'.format(i*100) for i in tick_labels])

image_filename = fig_dir +  fig_labels['fig_title'].lower().replace(' ', '_') + '.pdf'
make_and_save_standard_fig(plot_to_save, fig_labels,  image_filename)


# In[572]:

all_ndqs = quest_evaluator.build_question_lookup(by_type=True)['nonDiagramQuestions']
question_text = make_prop_series(all_ndqs, lambda x: x['beingAsked']['processedText'])
questions_sw_removed = {}
for qid, quest in all_ndqs.items():
    quest_text = quest['beingAsked']['processedText']
    quest['beingAsked']['processedText'] = ' '.join(tokenize_text(quest_text))
    questions_sw_removed[qid] = quest


# In[505]:

most_common_tri


# In[504]:

most_common_bigrams


# In[512]:

common_answers_to_common_questions = defaultdict(list)
for qid, question in questions_sw_removed.items():
    for phrase in most_common_bigrams:
        if phrase[0] in question['beingAsked']['processedText']:
            answer_id = question['correctAnswer']['processedText']
            if answer_id not in question['answerChoices'].keys():
                continue
            answer_text = question['answerChoices'][answer_id]['processedText']
            answer_text_cleaned = ' '.join(tokenize_question(answer_text))
            if 'the above' not in answer_text:
                common_answers_to_common_questions[phrase[0]].append(answer_text)


# In[513]:

for phrase, answer_series in common_answers_to_common_questions.items():
    answer_series = pd.Series(answer_series)
    print(phrase)
    print(answer_series.value_counts()[:5])
    print()


# ### second pass look only at start of last sentence

# In[706]:

seven_ww = ['what', 'where', 'when', 'who', 'why', 'how', 'which']


# In[ ]:




# In[ ]:




# ## question length

# In[547]:

question_lengths = make_prop_series(all_ndqs, lambda x: len(x['beingAsked']['processedText'].split()))
question_lengths = question_lengths[question_lengths < 20]

fig_labels = {
    'fig_title': 'Question Length Distribution', 
    'x_label': '# Words in Question',
    'y_label': 'Percentage of Questions'
}
weights = np.ones_like(question_lengths)/len(question_lengths)
plot_to_save = question_lengths.hist(bins= 19, log=False, weights=weights)
tick_labels = plot_to_save.get_yticks()
plot_to_save.set_yticklabels(['{:2.2f}%'.format(i*100) for i in tick_labels])

image_filename = fig_dir +  fig_labels['fig_title'].lower().replace(' ', '_') + '.pdf'
make_and_save_standard_fig(plot_to_save, fig_labels,  image_filename)


# In[716]:

sum([len(sent_tokenize(text)) for text in lesson_text.values()])


# In[906]:

len(all_diagram_qs)


# ## looking at all/some/none of the above questions

# In[253]:

reasoning_q_phrases = [' if ', 'what would ', 'what happens']
reasoning_questions = collect_filtered_question_text(all_ndqs, reasoning_q_phrases)


# In[125]:

question_asks = {}
answer_choices_with_above = []
correct_answer_choices_with_above = []
for qid, question in mc_questions.items(): 
    question_text = question['beingAsked']['processedText']
    ans_choices = {acid: ac['processedText'] for acid, ac in question['answerChoices'].items() if 'the above' in ac['processedText'].lower()}
    if ans_choices:
        above_choice_correct = 'd' == question['correctAnswer']['processedText']
        correct_answer_choices_with_above.append(above_choice_correct)
    answer_choices_with_above += list(ans_choices.values())
    question_asks[qid] = question_text

sum(correct_answer_choices_with_above) / len(answer_choices_with_above)

len(answer_choices_with_above)


# In[ ]:




# ## Looking for answer text in associated lessons

# In[78]:

punct_stop = list(string.punctuation)
cached_sw = stopwords.words("english") + punct_stop + list(['_____', '__________'])
lmtizer = WordNetLemmatizer()
tokenizer = wordpunct_tokenize

def filter_pos(token_list):
    tagged_tokens = nltk.pos_tag(token_list)
    filtered_tokes = [toke[0] for toke in tagged_tokens if toke[1] in ['NN', 'RB', 'NNS', 'JJ']]
    if filtered_tokes:
        return filtered_tokes
    else:
        return []

def lemmatize_and_filter_sw(tokens):
    return filter_pos([lmtizer.lemmatize(toke) for toke in wordpunct_tokenize(tokens) if toke not in cached_sw])


# In[79]:

vocab_topics = ['Lesson Vocabulary', 'Vocabulary']
structural_topics = ['Summary', 'Review', 'References', 'Explore More', 'Lesson Summary', 'Lesson Objectives',
                     'Points to Consider', 'Introduction', 'Recall', 'Apply Concepts', 'Think Critically', 'Resources',
                     'Explore More II', 'Explore More I', 'Explore More III']
ignore_topics = vocab_topics + structural_topics


# In[80]:

answer_choice_idxs = make_prop_series(all_ndqs, lambda x:x['correctAnswer']['processedText'])


# In[81]:

lesson_text_lookup = {}
for lesson in ck12_combined_dataset:
    lesson_text = ' '.join([topic['content']['text'] for topic_name, topic in sorted(lesson['topics'].items(), key=lambda x:x[1]['globalID']) if topic_name not in ignore_topics])
    lesson_text_lookup[lesson['globalID']] = [lmtizer.lemmatize(toke).lower() for toke in wordpunct_tokenize(lesson_text) if toke not in punct_stop]


# In[82]:

list(lesson_text_lookup.items())[50]


# In[83]:

question_text_lookup = {}
for lesson in ck12_combined_dataset:
    question_text_lookup[lesson['globalID']] = defaultdict(dict)
    for qid, question in lesson['questions']['nonDiagramQuestions'].items():
        if question['questionSubType'] != 'Multiple Choice':
            continue
        correct_answer_idx = question['correctAnswer']['processedText']
        if correct_answer_idx not in question['answerChoices'].keys():
            continue
        ans_choices_text = ''
        for ac_text in question['answerChoices'].values():
            ans_choices_text += ac_text['processedText'] + ' '
#         correct_answer_text = question['answerChoices'][correct_answer_idx]['processedText']
        question_asked = question['beingAsked']['processedText']
        question_text_lookup[lesson['globalID']][qid]= ' '.join([question_asked, ans_choices_text])


# In[84]:

# list(question_text_lookup.items())[531]


# In[ ]:




# In[85]:

get_ipython().run_cell_magic('time', '', 'question_tokens = {lid: {qid: set(lemmatize_and_filter_sw(q_text)) for qid, q_text in questions.items()} for lid, questions in question_text_lookup.items()}')


# In[86]:

tlid, tqt = list(question_tokens.items())[531]
tlt = lesson_text_lookup[tlid]

stqt = list(tqt.values())[0]


# In[87]:

def compute_question_spread(lesson_text_array, question_tokens):
    oo_lesson_words = []
    q_toke_positions = []
    for token in question_tokens:
        if token not in lesson_text_array:
            pass
#             print(token)
        else:
            q_toke_positions.append(lesson_text_array.index(token))
    if q_toke_positions:
        return max(q_toke_positions) - min(q_toke_positions)


# In[88]:

compute_question_spread(tlt, stqt)


# In[89]:

question_dist_measure = {}
for l_id, lesson_text in lesson_text_lookup.items():
    for qid, question_tokes in question_tokens[l_id].items():
        q_dist = compute_question_spread(lesson_text, question_tokes)
        if q_dist:
            question_dist_measure[qid] = q_dist


# In[90]:

len(mc_questions)


# In[91]:

len(q_dist_series)


# In[ ]:

q_dist_series = pd.Series(list(question_dist_measure.values()))


# In[ ]:

fig_labels = {
    'fig_title': 'Simple Question Spread Metric', 
    'x_label': 'Number of words in lesson question spans',
    'y_label': 'Percentage of Questions'
}

array_to_hist = q_dist_series
weights = np.ones_like(array_to_hist)/len(array_to_hist)
plot_to_save = array_to_hist.hist(bins= 20, log=False, weights=weights)
tick_labels = plot_to_save.get_yticks()
plot_to_save.set_yticklabels(['{:2.2f}%'.format(i*100) for i in tick_labels])

image_filename = fig_dir +  fig_labels['fig_title'].lower().replace(' ', '_') + '.pdf'
make_and_save_standard_fig(plot_to_save, fig_labels,  image_filename)


# In[92]:

get_ipython().run_cell_magic('javascript', '', 'require([\'d3\'], function(d3){\n    \nvar diameter = 960,\n    radius = diameter / 2,\n    innerRadius = radius - 120;\n\nvar cluster = d3.layout.cluster()\n    .size([360, innerRadius])\n    .sort(null)\n    .value(function(d) { return d.size; });\n\nvar bundle = d3.layout.bundle();\n\nvar line = d3.svg.line.radial()\n    .interpolate("bundle")\n    .tension(.85)\n    .radius(function(d) { return d.y; })\n    .angle(function(d) { return d.x / 180 * Math.PI; });\n\nvar svg = d3.select("body").append("svg")\n    .attr("width", diameter)\n    .attr("height", diameter)\n  .append("g")\n    .attr("transform", "translate(" + radius + "," + radius + ")");\n\nvar link = svg.append("g").selectAll(".link"),\n    node = svg.append("g").selectAll(".node");\n\nd3.json("files/readme-flare-imports.json", function(error, classes) {\n  if (error) throw error;\n\n  var nodes = cluster.nodes(packageHierarchy(classes)),\n      links = packageImports(nodes);\n\n  link = link\n      .data(bundle(links))\n    .enter().append("path")\n      .each(function(d) { d.source = d[0], d.target = d[d.length - 1]; })\n      .attr("class", "link")\n      .attr("d", line);\n\n  node = node\n      .data(nodes.filter(function(n) { return !n.children; }))\n    .enter().append("text")\n      .attr("class", "node")\n      .attr("dy", ".31em")\n      .attr("transform", function(d) { return "rotate(" + (d.x - 90) + ")translate(" + (d.y + 8) + ",0)" + (d.x < 180 ? "" : "rotate(180)"); })\n      .style("text-anchor", function(d) { return d.x < 180 ? "start" : "end"; })\n      .text(function(d) { return d.key; })\n      .on("mouseover", mouseovered)\n      .on("mouseout", mouseouted);\n});\n\nfunction mouseovered(d) {\n  node\n      .each(function(n) { n.target = n.source = false; });\n\n  link\n      .classed("link--target", function(l) { if (l.target === d) return l.source.source = true; })\n      .classed("link--source", function(l) { if (l.source === d) return l.target.target = true; })\n    .filter(function(l) { return l.target === d || l.source === d; })\n      .each(function() { this.parentNode.appendChild(this); });\n\n  node\n      .classed("node--target", function(n) { return n.target; })\n      .classed("node--source", function(n) { return n.source; });\n}\n\nfunction mouseouted(d) {\n  link\n      .classed("link--target", false)\n      .classed("link--source", false);\n\n  node\n      .classed("node--target", false)\n      .classed("node--source", false);\n}\n\nd3.select(self.frameElement).style("height", diameter + "px");\n\n// Lazily construct the package hierarchy from class names.\nfunction packageHierarchy(classes) {\n  var map = {};\n\n  function find(name, data) {\n    var node = map[name], i;\n    if (!node) {\n      node = map[name] = data || {name: name, children: []};\n      if (name.length) {\n        node.parent = find(name.substring(0, i = name.lastIndexOf(".")));\n        node.parent.children.push(node);\n        node.key = name.substring(i + 1);\n      }\n    }\n    return node;\n  }\n\n  classes.forEach(function(d) {\n    find(d.name, d);\n  });\n\n  return map[""];\n}\n\n// Return a list of imports for the given array of nodes.\nfunction packageImports(nodes) {\n  var map = {},\n      imports = [];\n\n  // Compute a map from name to node.\n  nodes.forEach(function(d) {\n    map[d.name] = d;\n  });\n\n  // For each import, construct a link from the source to target node.\n  nodes.forEach(function(d) {\n    if (d.imports) d.imports.forEach(function(i) {\n      imports.push({source: map[d.name], target: map[i]});\n    });\n  });\n\n  return imports;\n}\n});')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[702]:




# In[ ]:




# In[ ]:




# ## hide

# In[232]:

# for qid, q in sorted(question_asks.items(), key=lambda x: len(x[1]), reverse=True)[:20]:
#     print(q)
# #     print(mc_questions[qid]['answerChoices'])
#     print()


# In[161]:

# plt.scatter(xns, yb, color=b16_colors.g, marker = 'o', s=45)
# plt.plot(xns, lin_mod.predict(np.array(xns).reshape(-1,1)), color=b16_colors.b ,alpha=0.8, linewidth=5.0)
# plt.xlim((0,100))
# plt.ylim((0, 1))
# label_color =  b16_colors.b
# plt.title('Ingredient Cooccurrences', fontsize=50, verticalalignment='bottom', color = label_color)
# plt.ylabel("Fraction of pairs extant", fontsize=30, labelpad=15, color = label_color)
# plt.xlabel("Number of shared flavor compounds", fontsize=30, labelpad=10, color = label_color)
# plt.tick_params(axis='x', which='major', labelsize=20)
# plt.tick_params(axis='y', which='major', labelsize=20)


# In[519]:

# def tokenize_question(question_text):
#     q_tokens = wordpunct_tokenize(question_text)
#     normalized_tokens = [toke.strip().lower().encode('ascii', 'ignore').decode() for toke in q_tokens if toke.strip().lower().encode('ascii', 'ignore').decode() not in cached_sw]
#     return normalized_tokens

# def make_and_save_standard_fig(fig_plt, fig_labels=None, outfile='latest_fig.pdf', main_color=b16_colors.b, label_color = '0.25'):
#     if fig_labels:
#         if 'fig_title' in fig_labels:
#             plt.title(fig_labels['fig_title'], fontsize=30, verticalalignment='bottom', color = label_color)
#         if 'y_label' in fig_labels:
#             plt.ylabel(fig_labels['y_label'], fontsize=25, labelpad=10, color = label_color)
#         if 'x_label' in fig_labels:
#             plt.xlabel(fig_labels['x_label'], fontsize=25, labelpad=10, color = label_color)
#     plt.tick_params(axis='x', which='major', labelsize=15)
#     plt.tick_params(axis='y', which='major', labelsize=15)
#     plt.savefig(outfile, bbox_inches='tight')

# def collect_filtered_question_text(question_dict, filter_phrases=None):
#     filtered_questions = {}
#     for phrase in filter_phrases:
#         for qid, question in question_dict.items():
#             question_text = question['beingAsked']['processedText']
#             if phrase in question_text:
#                 filtered_questions[qid] = question_text
#     return filtered_questions

# def make_prop_series(ds_items, property_collector):
#     return(pd.Series([property_collector(item) for item in ds_items.values()]))

# def collect_filtered_lesson_text(complete_ds):
#     filtered_lesson_text = defaultdict(str)
#     for lesson in complete_ds:
#         lesson_key = lesson['lessonName'] + '_' + lesson['globalID']
#         for topic_name, topic in sorted(lesson['topics'].items(), key=lambda x: x[1]['globalID']):
#             if not topic['topicName'] in structural_topics + vocab_topics:
#                 filtered_lesson_text[lesson_key] += topic['content']['text'] 
#     return filtered_lesson_text


# # lesson length

# In[236]:

lesson_text = collect_filtered_lesson_text(ck12_combined_dataset, True, True)
topic_lengths = pd.Series([len(sent_tokenize(text)) for text in lesson_text.values() if len(sent_tokenize(text)) < 300])


# In[238]:

fig_labels = {
    'fig_title': 'Lesson Length Distribution', 
    'x_label': '# Sentences in Lesson',
    'y_label': 'Percentage of Lessons'
}
weights = np.ones_like(topic_lengths)/len(topic_lengths)
plot_to_save = topic_lengths.hist(bins= 20, log=False, weights=weights)
tick_labels = plot_to_save.get_yticks()
plot_to_save.set_yticklabels(['{:2.2f}%'.format(i*100) for i in tick_labels])

image_filename = fig_dir +  fig_labels['fig_title'].lower().replace(' ', '_') + '.pdf'
make_and_save_standard_fig(plot_to_save, fig_labels,  image_filename)


# # generate question annoations hists

# In[298]:

anno_data_dir = './q_anno_results/'


# In[299]:

anno_files = get_ipython().getoutput("ls './q_anno_results/'")


# In[300]:

anno_files 


# In[301]:

print(list(qa_df.columns))


# In[311]:

cols_new_names = ['multiple paragraphs', 'single paragraph', 'single sentence', 'external knowledge', 'multiple contexts', 'n1', 'requires diagram', 'lesson_name', 'qid']
qa_df.columns = cols_new_names


# In[303]:

qa_df_build = pd.read_csv(anno_data_dir + anno_files[3])
qa_df_build = qa_df_build.append(pd.read_csv(anno_data_dir + anno_files[4]))
qa_df_build = qa_df_build.append(pd.read_csv(anno_data_dir + anno_files[5]))
qa_df = qa_df_build.fillna(0)
del qa_df['Weird']
del qa_df['Junk']


# In[318]:

dqa_df_build = pd.read_csv(anno_data_dir + anno_files[0])
dqa_df_build = dqa_df_build.append(pd.read_csv(anno_data_dir + anno_files[1]))
dqa_df_build = dqa_df_build.append(pd.read_csv(anno_data_dir + anno_files[2]))
dqa_df = dqa_df_build.fillna(0)


# In[319]:

print(list(dqa_df.columns))


# In[323]:

dqa_df.columns = dq_new_cols_to_hist_1 + dq_new_cols_to_hist_2 + ['Is OCR enough', 'Nothing to do with D', 'Only D'] + ['Unnamed: 13', 'lesson_name', 'qid']


# In[312]:

# cols_to_hist = ['1 lesson', '1 para', '1 sentence', 'External Knowledge', 'Multi-lesson', 'Uses Diagram']
cols_to_hist = ['multiple paragraphs', 'single paragraph', 'single sentence', 'external knowledge', 'multiple contexts', 'requires diagram']


# In[320]:

dq_cols_to_hist_1 = ['D + 1 image', 'D + 1 image + Lesson (Multi paragraphs)', 'D + 1 para', 'D + External Knowledge', 'D + Lesson (Multi paragraphs)', 'D + Multiple Lessons']
dq_cols_to_hist_2 = ['D-Structure: 1 part', 'D-Structure: Multiple parts', 'D: Global']
dq_new_cols_to_hist_1 = ['one image', 'D + one image + multiple paragraphs', 'one paragraph', 'external knowledge', 'D + multiple paragraphs', 'D + multiple contexts']
dq_new_cols_to_hist_2 = ['single component', 'multiple components', 'all components']


# In[314]:

fig_labels = {
    'fig_title': '', 
    'x_label': '',
    'y_label': 'Number of Questions'
}

# plt.rcParams['figure.figsize'] = (16.0, 10.0)

plot_to_save  = qa_df[cols_to_hist].apply(sum).sort_values(ascending=False).plot(kind='bar')
tick_labels = plot_to_save.get_yticks()
plt.xticks(rotation=40) 

image_filename = fig_dir + 'question_manual_annotation_hist'  + '.pdf'
make_and_save_standard_fig(plot_to_save, fig_labels,  image_filename)


# In[339]:

fig_labels = {
    'fig_title': ' ', 
    'x_label': '',
    'y_label': 'Number of Questions'
}

plt.rcParams['figure.figsize'] = (10.0, 10.0)

plot_to_save  = dqa_df[dq_new_cols_to_hist_2].apply(sum).sort_values(ascending=False).plot(kind='bar')
tick_labels = plot_to_save.get_yticks()
plt.xticks(rotation=0) 
image_filename = fig_dir +  'Diagram Question Annotation Hist 2'.replace(' ', '_') + '.pdf'
make_and_save_standard_fig(plot_to_save, fig_labels,  image_filename)


# In[338]:

fig_labels = {
    'fig_title': '', 
    'x_label': '',
    'y_label': 'Number of Questions'
}

plot_to_save = qa_df[cols_to_hist].apply(sum).sort_values(ascending=False).plot(kind='bar')
tick_labels = plot_to_save.get_yticks()
plt.xticks(rotation=20) 

image_filename = fig_dir +  'Question Annotation Hist'.lower().replace(' ', '_') + '.pdf'
make_and_save_standard_fig(plot_to_save, fig_labels,  image_filename)


# In[ ]:

fig_labels = {
    'fig_title': 'Question Annotation Hist', 
    'x_label': '',
    'y_label': 'Number of Questions'
}

plot_to_save = qa_hist = qa_df[cols_to_hist].apply(sum).sort_values(ascending=False).plot(kind='bar')
tick_labels = plot_to_save.get_yticks()
plt.xticks(rotation=40) 

image_filename = fig_dir +  fig_labels['fig_title'].lower().replace(' ', '_') + '.pdf'
make_and_save_standard_fig(plot_to_save, fig_labels,  image_filename)


# # hierarchical edge bundling

# In[115]:

import random


# In[105]:

util_eval = Evaluator(data_file)


# In[95]:

with open('./readme-flare-imports.json') as f:
    sample_data =json.load(f)


# In[102]:

sample_data[40]


# In[151]:

random_lesson_connections[40]


# In[126]:

rc = [random.choice(['test']*20 + ['train']*80) for i in range(100)]
sum([c == 'train' for c in rc])


# In[129]:

all_lessons = [{'lname': lesson['lessonName'], 'tta': random.choice(['test']*20 + ['train']*80)} for lesson in ck12_combined_dataset]


# In[185]:

for lesson in all_lessons:
    lesson['importName'] = lesson['tta'] + '.' + lesson['lname'].replace(' ', '_')
random_lesson_sample = [lesson for lesson in all_lessons if random.choice(range(10)) > 8]


# In[221]:

random_lesson_connections = []
lessons_to_show = random_lesson_sample
for i in range(len(lessons_to_show)):
    this_lesson = lessons_to_show[i]

    this_entry = {
        'name': this_lesson['importName'],
        'imports': [lesson['importName'] for lesson in random.sample(lessons_to_show, random.randint(1 ,5)) if lesson['importName'].split('.')[0] == this_lesson['importName'].split('.')[0]],
        'size': '300'
    }
    random_lesson_connections.append(this_entry)


# In[228]:

len(random_lesson_connections)


# In[223]:

with open('lesson_connections.json', 'w') as f:
    json.dump(random_lesson_connections, f)


# In[230]:

get_ipython().run_cell_magic('HTML', '', '<iframe width="100%" height="500" src="index.html?inline=false"></iframe>')


# In[ ]:




# In[ ]:




# # Generate sample HTML to review

# In[967]:

from tqa_utils.render_html import render_sample_question_and_lesson
import random
import gzip


# In[834]:

def assign_questions(questions, n):
    for i in rangfe(0, len(questions), n):
        yield questions[i:i + n]


# In[968]:

with open('./fn_to_ocrs.pklz', 'rb') as f:
    ocr_results = pickle.load(gzip.open(f))


# In[971]:

ocr_results.keys()


# In[ ]:

result = pickle.load(gzip.open(some_pklz_file))


# non diagrams

# In[805]:

# random_question_sample = random.sample(list(all_ndqs.values()), 300)
# random_question_dict = {v['globalID']: v for v in random_question_sample}


# diagrams

# In[893]:

# random_question_sample_diagram = random.sample(list(all_diagram_qs.values()), 300)
# random_diagram_question_dict = {v['globalID']: v for v in random_question_sample_diagram}


# In[972]:

# random_assignments = list(assign_questions(list(random_question_dict.keys()), 100))
random_assignments = list(assign_questions(list(random_diagram_question_dict.keys()), 100))


# In[973]:

reviewers = ['dustin', 'ani', 'choi']
reviewer_assignments = {reviewers[i]: random_assignments[i] for i in range(len(random_assignments))}
question_reviewer_lookup = {}
for reviewer, qid_list in reviewer_assignments.items():
    for qid in qid_list:
        question_reviewer_lookup[qid] = reviewer


# In[985]:

question_sample_with_lessons = {} 
lessons_involved = []
# id_ocr_lookup = {}
#         id_ocr_lookup[qid] = quest_dict[qid]['imageName']

q_type = 'diagramQuestions'
q_not_type = 'nonDiagramQuestions'
# q_type = 'nonDiagramQuestions'
# q_not_type = 'diagramaQuestions'
quest_dict = random_diagram_question_dict
for lesson in ck12_combined_dataset:
    question_sample_in_lesson = [qid for qid in lesson['questions'][q_type].keys() if qid in quest_dict.keys()]
    for qid in question_sample_in_lesson:
        lessons_involved.append(lesson['lessonName'])
        sample_lesson = deepcopy(lesson)
        sample_lesson['questions'][q_not_type] = {}
        question_ocr = ocr_results[id_ocr_lookup[qid]]
        quest_dict[qid]['ocrResults'] = question_ocr
        sample_lesson['questions'][q_type] = {qid: quest_dict[qid]}
        sample_lesson['reviewer'] = question_reviewer_lookup[qid]
        question_sample_with_lessons[qid] = sample_lesson


# In[1002]:

bad_qs = []
for q in all_diagram_qs.values():
    if q['correctAnswer']['rawText'].lower() == q['beingAsked']['processedText'].lower():
        bad_qs.append(q['correctAnswer'])


# In[1004]:

bad_qs


# In[955]:

len(set(lessons_involved))


# In[900]:

pd.Series(lessons_involved).value_counts()


# In[869]:

# pd.Series(lessons_involved).value_counts()[:10]


# In[988]:

for qid, lesson_to_render in question_sample_with_lessons.items():
        out_path = lesson_to_render['reviewer'] + '_sample_diagram_questions' 
        render_sample_question_and_lesson(lesson_to_render, qid, out_path, question_ocr)


# In[871]:

# test_html = render_sample_question_and_lesson(test_sample_out[1], 'somepath', 'qid')


# In[903]:

qid_df = pd.DataFrame(list(quest_dict.keys()))
qid_df.columns = ['qid']
qid_df['lesson_name'] = qid_df['qid'].apply(lambda x: question_sample_with_lessons[x]['lessonName'])

qid_df['reviwer_file'] = qid_df['qid'].apply(lambda x: question_reviewer_lookup[x] + '_sample_diagram_questions.csv')
# qid_df['reviwer_file'] = qid_df['qid'].apply(lambda x: question_reviewer_lookup[x] + '_sample_questions.csv')


# In[905]:

for rev, gdf in qid_df.groupby('reviwer_file'):
     gdf[['qid', 'lesson_name']].to_csv(rev, index=False)


# # Templates

# In[ ]:

fig_labels = {
    'fig_title': '', 
    'x_label': '',
    'y_label': 'Percentage of Questions'
}

array_to_hist = 
weights = np.ones_like(array_to_hist)/len(array_to_hist)
plot_to_save = array_to_hist.hist(bins= 20, log=False, weights=weights)
tick_labels = plot_to_save.get_yticks()
plot_to_save.set_yticklabels(['{:2.2f}%'.format(i*100) for i in tick_labels])

image_filename = fig_dir +  fig_labels['fig_title'].lower().replace(' ', '_') + '.pdf'
make_and_save_standard_fig(plot_to_save, fig_labels,  image_filename)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



