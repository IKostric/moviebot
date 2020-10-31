"""This file contains a class which can be used to annotate slot values
in the user utterance based on rules and keyword matching."""

__author__ = 'Javeria Habib'

import re
import string
from copy import deepcopy

from nltk import ngrams
from nltk.corpus import stopwords

from moviebot.nlu.annotation.slot_annotator import SlotAnnotator
from moviebot.nlu.annotation.item_constraint import ItemConstraint
from moviebot.nlu.annotation.operator import Operator
from moviebot.nlu.annotation.slots import Slots


class RBAnnotator(SlotAnnotator):
    """This is a rule based annotator. It uses regex and keyword matching
    for annotation."""

    def __init__(self, process_value, lemmatize_value, slot_values):
        self._process_value = process_value
        self._lemmatize_value = lemmatize_value
        self.slot_values = slot_values
        self.ngram_size = {
            Slots.TITLE.value: 8,
            Slots.KEYWORDS.value: 8,
            'person': 3
        }
        self.stop_words = [
            'give', 'need', 'good', 'tell', 'movie', 'yes', 'please', 'would',
            'should', 'something', 'want', 'will', 'shall', 'do', 'you', 'know',
            'get', 'old', 'new', 'latest', 'top', 'high', 'low', 'rating',
            'rated'
        ]
        self.stop_words.extend(stopwords.words('english'))
        self.stop_words = set(self.stop_words)
        self.genres_alternatives = {
            'romantic': 'romance',
            'criminal': 'crime',
            'dramatical': 'drama',
            'sports': 'sport',
            'funny': 'comedy',
            'historical': 'history',
            'animated': 'animation'
        }
        # merging actors and directors
        self.person_names = {}
        for slot in [Slots.ACTORS.value, Slots.DIRECTORS.value]:
            self.person_names.update(deepcopy(self.slot_values[slot]))

    def slot_annotation(self, slot, user_utterance):
        """

        Args:
            slot: 
            utterance: 
            raw_utterance: 

        """
        # utterance = utterance.replace('?','')
        if slot in [x.value for x in [Slots.ACTORS, Slots.DIRECTORS]]:
            params = self._person_name_annotator(user_utterance)
        else:
            func = getattr(self, '_' + slot + '_annotator')
            if slot == Slots.YEAR.value:
                params = func(slot, user_utterance)
                if params:
                    for p in params:
                        if p.op != Operator.EQ:
                            p.value = f'{str(p.op)} {p.value}'
                            p.op = Operator.EQ
            else:
                params = func(slot, user_utterance)
        return params

    def _genres_annotator(self, slot, user_utterance):
        """

        Args:
            slot: 
            utterance: 

        """
        param = None
        values = self.slot_values[slot]
        tokens = user_utterance.get_tokens()
        utterance = sum(tokens).lemma if tokens else ''
        for value, lem_value in values.items():
            if lem_value in utterance:
                if not param:
                    param = ItemConstraint(slot, Operator.EQ, value.lower())
                else:
                    param.value += ' ' + value.lower()
        for key, value in self.genres_alternatives.items():
            if self._process_value(key) in utterance:
                if not param:
                    param = ItemConstraint(slot, Operator.EQ, key.lower())
                else:
                    param.value += ' ' + key.lower()
        if param:
            return [param]

    def _title_annotator(self, slot, user_utterance):
        """This annotator is used to check the movie title.
        Sometimes the user can just enter a part of the name.

        Args:
            slot: 
            utterance: 

        """
        tokens = user_utterance.get_tokens()
        values = self.slot_values[slot]
        processed_values = set(values.values())
        # split into n-grams
        for ngram_size in range(min(self.ngram_size[slot], len(tokens)), 0, -1):
            options = {}
            for gram_list in ngrams(tokens, ngram_size):
                gram = sum(gram_list).lemma
                for processed_value in processed_values:
                    if processed_value == gram and len([
                            x.lemma
                            for x in gram_list
                            if x.lemma in self.stop_words
                    ]) < ngram_size:
                        # TODO (Ivica Kostric): This is ready to be converted to
                        # the SemanticAnnotation class
                        param = ItemConstraint(slot, Operator.EQ, gram)
                        return [param]
                if len([
                        x.lemma for x in gram_list if x.lemma in self.stop_words
                ]) == 0 and len(
                    [int(val) for val in re.findall(r'\b\d+', gram)]) == 0:
                    # check if
                    # all words are in the list of stop words and no numbers
                    if ngram_size == 1:
                        gram_occurrence = len([
                            value for value in processed_values if gram == value
                        ])
                    else:
                        gram_occurrence = len([
                            value for value in processed_values
                            if f' {gram} ' in f' {value} '
                        ])
                    if gram_occurrence:
                        options[gram] = gram_occurrence
            if options:
                options = {
                    k: v for k, v in sorted(options.items(),
                                            key=lambda item: item[1])
                }
                for gram in options:
                    param = ItemConstraint(slot, Operator.EQ, gram.strip())
                    return [param]

    def _keywords_annotator(self, slot, user_utterance):
        """This annotator is used to check the movie keywords.
        If the ngram has only keywords, it will be ignored.

        Args:
            slot: 
            utterance: 

        """
        tokens = user_utterance.get_tokens()
        values = self.slot_values[slot]
        for ngram_size in range(min(self.ngram_size[slot], len(tokens)), 0, -1):
            for gram_list in ngrams(tokens, ngram_size):
                gram = sum(gram_list).lemma

                # TODO (Ivica Kostric): maybe 'no numbers' should be changed
                # since there are some numbers in keywords (.44, 007, age).
                # Same goes for stopwords. There are some stopwords as part of
                # keywords.
                # Alternatively, there is possibility to put a stopword flag
                # directly on tokens beforehand.

                if len([
                        int(val) for val in re.findall(r'\b\d+', gram)
                ]) == 0 and len([
                        x.lemma for x in gram_list if x.lemma in self.stop_words
                ]) == 0:

                    for value, lem_value in values.items():
                        # TODO (Ivica Kostric): This is ready to be converted to
                        # SemanticAnnotation class
                        if lem_value == gram:
                            param = ItemConstraint(slot, Operator.EQ, gram)
                            return [param]
                        elif (ngram_size == 1 and gram == lem_value) or (
                                ngram_size > 1
                                and f' {gram} ' in f' {lem_value} '):
                            param = ItemConstraint(slot, Operator.EQ, gram)
                            return [param]

    def _person_name_annotator(self, user_utterance, slots=None):
        """This annotator is used to check the movie actor and/or director
        names. Sometimes the user can just enter a part of the name.

        Args:
            utterance: 
            slots:  (Default value = None)

        """
        tokens = user_utterance.get_tokens()
        if not slots:
            slots = [Slots.ACTORS.value, Slots.DIRECTORS.value]
            person_names = self.person_names
        else:
            slots = [slots]
            person_names = self.slot_values[slots]
        params = []
        for ngram_size in range(self.ngram_size['person'], 0, -1):
            for gram_list in ngrams(tokens, ngram_size):
                gram = sum(gram_list).lemma
                for value, lem_value in person_names.items():
                    if f' {gram} ' in f' {lem_value} ' and \
                        gram not in self.stop_words:
                        # gramR = self.find_in_raw_utterance(raw_utterance,
                        #                                   ngram_size,
                        #                                   gram)
                        for slot in slots:
                            if gram in self.slot_values[slot].values():

                                # TODO (Ivica Kostric): This is ready to be
                                # converted to the SemanticAnnotation class

                                params.append(
                                    ItemConstraint(slot, Operator.EQ, gram))
                        break
            if len(params) > 0:
                return params

    def _year_annotator(self, slot, user_utterance):
        """

        Args:
            slot: 
            user_utterance: 
            utterance: 

        """
        raw_utterance = user_utterance.get_text()
        tokens = user_utterance.get_tokens()
        utterance = sum(tokens).lemma if tokens else ''
        # fitst option is to find if any value is in the possible values
        possible_years = [
            int(val) for val in re.findall(r'\b\d+', raw_utterance)
        ]
        for year in possible_years:
            _year = str(year)
            if _year + 's' in raw_utterance:  # check if it's 1990s instead of
                # 1990 or 90s
                if len(_year) == 4:
                    if year % 10 == 0:
                        return [
                            ItemConstraint(slot, Operator.BETWEEN,
                                           f'{_year} AND {str(year + 10)}')
                        ]
                    else:
                        return [ItemConstraint(slot, Operator.EQ, _year)]
                elif len(_year) == 2:
                    if year <= 20:
                        _year = '20' + _year
                        if year % 10 == 0:
                            return [
                                ItemConstraint(
                                    slot, Operator.BETWEEN, f'{_year} AND'
                                    f' {str(int(_year) + 10)}')
                            ]
                        else:
                            return [ItemConstraint(slot, Operator.EQ, _year)]
                    else:
                        _year = '19' + _year
                        if year % 10 == 0:
                            return [
                                ItemConstraint(
                                    slot, Operator.BETWEEN, f'{_year} AND'
                                    f' {str(int(_year) + 10)}')
                            ]
                        else:
                            return [ItemConstraint(slot, Operator.EQ, _year)]
            elif _year + 'th' in raw_utterance:
                # it can be string like 19th, 20th
                if len(_year) == 2:
                    return [
                        ItemConstraint(slot, Operator.BETWEEN, f'{_year}00 AND'
                                       f' {str(year + 1)}00')
                    ]
            if len(_year) == 4:
                return [ItemConstraint(slot, Operator.EQ, _year)]
        # adding a few more elements
        if 'new' in utterance or 'latest' in utterance:
            return [ItemConstraint(slot, Operator.GT, '2010')]
        if 'old' in utterance:
            return [ItemConstraint(slot, Operator.LT, '2010')]

    def find_in_raw_utterance(self, raw_utterance, gram, ngram_size):
        """

        Args:
            raw_utterance: 
            gram: 
            ngram_size: 

        """
        n_grams = ngrams(raw_utterance.split(), ngram_size)
        for _gram in n_grams:
            gramR = ' '.join(_gram)
            if self._lemmatize_value(gramR) == gram:
                while gramR[-1] in string.punctuation:
                    gramR = gramR[:-1]
                return gramR
