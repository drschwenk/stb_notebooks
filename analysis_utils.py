import pandas as pd
import numpy as np
import json
from collections import defaultdict
from collections import Counter
import string


def tokenize_question(question_text):
    q_tokens = wordpunct_tokenize(question_text)
    normalized_tokens = [toke.strip().lower().encode('ascii', 'ignore').decode() for toke in q_tokens if toke.strip().lower().encode('ascii', 'ignore').decode() not in cached_sw]
    return normalized_tokens


def make_and_save_standard_fig(fig_plt, fig_labels=None, outfile='latest_fig.pdf', main_color=None, label_color = '0.25'):
    if fig_labels:
        if 'fig_title' in fig_labels:
            plt.title(fig_labels['fig_title'], fontsize=30, verticalalignment='bottom', color = label_color)
        if 'y_label' in fig_labels:
            plt.ylabel(fig_labels['y_label'], fontsize=25, labelpad=10, color = label_color)
        if 'x_label' in fig_labels:
            plt.xlabel(fig_labels['x_label'], fontsize=25, labelpad=10, color = label_color)
    plt.tick_params(axis='x', which='major', labelsize=15)
    plt.tick_params(axis='y', which='major', labelsize=15)
    plt.savefig(outfile, bbox_inches='tight')


def collect_filtered_question_text(question_dict, filter_phrases=None):
    filtered_questions = {}
    for phrase in filter_phrases:
        for qid, question in question_dict.items():
            question_text = question['beingAsked']['processedText']
            if phrase in question_text:
                filtered_questions[qid] = question_text
    return filtered_questions


def make_prop_series(ds_items, property_collector):
    return pd.Series([property_collector(item) for item in ds_items.values()])


def collect_filtered_lesson_text(complete_ds):
    filtered_lesson_text = defaultdict(str)
    for lesson in complete_ds:
        lesson_key = lesson['lessonName'] + '_' + lesson['globalID']
        for topic_name, topic in sorted(lesson['topics'].items(), key=lambda x: x[1]['globalID']):
            if not topic['topicName'] in structural_topics + vocab_topics:
                filtered_lesson_text[lesson_key] += topic['content']['text'] 
    return filtered_lesson_text
