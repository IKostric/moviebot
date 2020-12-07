"""This file contains a class which can be used to annotate slot values
in the user utterance based on rules and keyword matching."""

import json
import re
import string
from copy import deepcopy

from nltk import ngrams
from nltk.corpus import stopwords

from moviebot.nlu.annotation.slot_annotator import SlotAnnotator
from moviebot.nlu.annotation.item_constraint import ItemConstraint
from moviebot.nlu.annotation.semantic_annotation import SemanticAnnotation
from moviebot.nlu.annotation.semantic_annotation import AnnotationType
from moviebot.nlu.annotation.semantic_annotation import EntityType
from moviebot.nlu.annotation.operator import Operator
from moviebot.nlu.annotation.slots import Slots

from REL.mention_detection import MentionDetection
from REL.utils import process_results
from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import Cmns, load_flair_ner


class NeuralAnnotator(SlotAnnotator):
    """This is a rule based annotator. It uses regex and kayword matching
    for annotation."""

    def __init__(self, process_value, lemmatize_value, slot_values):
        self.base_url = './third_party/REL/data'
        self.wiki_version = 'wiki_2019'

        # self.annotations = None
        self.mention_detection = MentionDetection(self.base_url,
                                                  self.wiki_version)
        self.taggers = {
            1: Cmns(self.base_url, self.wiki_version, n=5),
            2: load_flair_ner('ner-ontonotes-fast'),
            3: load_flair_ner('ner-fast')
        }

        config = {
            'mode': 'eval',
            'model_path': './third_party/REL/data/ed-wiki-2019/model',
        }

        self.model = EntityDisambiguation(self.base_url, self.wiki_version,
                                          config)

        with open('data/slot_values_dbpedia.json', 'r') as f:
            self.classes = json.load(f)

    def preprocess_for_rel(self, user_utterance):
        return {
            'query': [user_utterance.get_text(), []],
        }

    def slot_annotation(self, slot, user_utterance):
        if slot in [Slots.GENRES.value, Slots.TITLE.value]:
            return []
            tagger = self.taggers[1]

        elif slot in [Slots.ACTORS.value, Slots.DIRECTORS.value]:
            tagger = self.taggers[2]

        else:
            return []

        input_text = self.preprocess_for_rel(user_utterance)

        mentions_dataset, n_mentions = self.mention_detection.find_mentions(
            input_text, tagger)

        predictions, timing = self.model.predict(mentions_dataset)

        # print(predictions)
        result = process_results(mentions_dataset, predictions, input_text)
        # print('Final result\n', result)

        for res in result.get('query', []):
            link = res[3]
            for k, v in self.classes.items():
                if link in v:
                    print(res[:3], 'is', k)
        return []
