import pandas as pd
import numpy as np
import json
from collections import defaultdict
from collections import Counter
import string
import matplotlib as mpl
import matplotlib.pylab as plt
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder

fig_dir = './figs_to_save/'
vocab_topics = ['Lesson Vocabulary', 'Vocabulary']
structural_topics = ['Summary', 'Review', 'References', 'Explore More', 'Lesson Summary', 'Lesson Objectives',
                     'Points to Consider', 'Introduction', 'Recall', 'Apply Concepts', 'Think Critically', 'Resources',
                     'Explore More II', 'Explore More I', 'Explore More III']

cached_sw = ['?', 'the', 'a', 's', "'", 'this', '.', ',', '_____', '__________', 'an']


def tokenize_text(text, cached_sw):
    tokens = wordpunct_tokenize(text)
    normalized_tokens = [toke.strip().lower().encode('ascii', 'ignore').decode() for toke in tokens if toke.strip().lower().encode('ascii', 'ignore').decode() not in cached_sw]
    return normalized_tokens


def make_and_save_standard_fig(fig_plt, fig_labels=None, outfile='latest_fig.pdf', main_color=None, label_color = '0.25'):
    if fig_labels:
        if 'fig_title' in fig_labels:
            plt.title(fig_labels['fig_title'], fontsize=30, verticalalignment='bottom', color = label_color)
        if 'y_label' in fig_labels:
            plt.ylabel(fig_labels['y_label'], fontsize=25, labelpad=10, color=label_color)
        if 'x_label' in fig_labels:
            plt.xlabel(fig_labels['x_label'], fontsize=25, labelpad=10, color=label_color)
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


def extract_topics_and_adjunct_topics(lesson):
    values_to_return = [topic for topic in lesson['topics'].values()]
    values_to_return += [topic for topic_name, topic in lesson['adjunctTopics'].items() if topic_name not in vocab_topics]
    return values_to_return


def collect_filtered_lesson_text(complete_ds, include_adjunct=False, include_descriptions=False):
    filtered_lesson_text = defaultdict(str)
    for lesson in complete_ds:
        lesson_key = lesson['lessonName'] + '_' + lesson['globalID']
        for topic_name, topic in sorted(lesson['topics'].items(), key=lambda x: x[1]['globalID']):
                    filtered_lesson_text[lesson_key] += topic['content']['text'] + '\n'
        if include_adjunct:
            for topic_name, topic in lesson['adjunctTopics'].items():
                if topic_name not in vocab_topics:
                    filtered_lesson_text[lesson_key] += topic['content']['text'] + '\n'
        if include_descriptions:
            for d_description in lesson['instructionalDiagrams'].values():
                filtered_lesson_text[lesson_key] += d_description['processedText'] + '\n'
    return filtered_lesson_text


def count_textbook_images(complete_ds):
    images_encountered = 0
    for lesson in complete_ds:
        for topic_name, topic in sorted(lesson['topics'].items(), key=lambda x: x[1]['globalID']):
            images_encountered += len(topic['content']['mediaLinks'])
            len(topic['content']['mediaLinks'])
        for topic_name, topic in lesson['adjunctTopics'].items():
            if topic_name not in vocab_topics:
                images_encountered += len(topic['content']['mediaLinks'])
    return images_encountered