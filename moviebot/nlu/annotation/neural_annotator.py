"""This file contains a class which can be used to annotate slot values
in the user utterance based on rules and keyword matching."""

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

    def __init__(self, _process_value, _lemmatize_value, slot_values):
        self.base_url = './third_party/REL/data'
        self.wiki_version = 'wiki_2019'

        self.annotations = None

    def preprocess_for_rel(self, user_utterance):
        return {
            'query': [user_utterance.get_text(), []],
        }

    def slot_annotation(self, slot, user_utterance):
        input_text = self.preprocess_for_rel(user_utterance)

        mention_detection = MentionDetection(self.base_url, self.wiki_version)
        # tagger = load_flair_ner("ner-fast")
        tagger = Cmns(self.base_url, self.wiki_version, n=5)
        mentions_dataset, n_mentions = mention_detection.find_mentions(
            input_text, tagger)

        print(mentions_dataset, n_mentions)
        config = {
            "mode": "eval",
            "model_path": "./third_party/REL/data/ed-wiki-2019/model",
        }

        model = EntityDisambiguation(self.base_url, self.wiki_version, config)
        predictions, timing = model.predict(mentions_dataset)

        print(predictions)
        result = process_results(mentions_dataset, predictions, input_text)
        print("Final result\n", result)
