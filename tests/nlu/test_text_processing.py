from moviebot.nlu.text_processing import Tokenizer
import pytest


@pytest.fixture
def utterance():
    return 'Document.with, punctuation:   With?spaces\tTabs\nwith newlines\n\n'


def test_number_of_tokens(utterance):
    # Setup
    tp = Tokenizer()

    # Exercise
    result = tp.process_text(utterance)

    # Results
    assert len(result) == 8

    # Cleanup - none


def test_first_token(utterance):
    # Setup
    tp = Tokenizer()

    # Exercise
    result = tp.process_text(utterance)

    # Results
    assert result[0].text == 'Document'

    # Cleanup - none


def test_token_is_not_stopword(utterance):
    # Setup
    tp = Tokenizer()

    # Exercise
    result = tp.process_text(utterance)

    # Results
    assert result[0].is_stop == False

    # Cleanup - none


def test_token_is_stopword(utterance):
    # Setup
    tp = Tokenizer()

    # Exercise
    result = tp.process_text(utterance)

    # Results
    assert result[3].is_stop == True

    # Cleanup - none


def test_token_lemma(utterance):
    # Setup
    tp = Tokenizer()

    # Exercise
    result = tp.process_text(utterance)

    # Results
    assert result[5].lemma == 'tab'

    # Cleanup - none


def test_token_start(utterance):
    # Setup
    tp = Tokenizer()

    # Exercise
    result = tp.process_text(utterance)

    # Results
    assert result[5].start == 42

    # Cleanup - none


def test_token_end(utterance):
    # Setup
    tp = Tokenizer()

    # Exercise
    result = tp.process_text(utterance)

    # Results
    assert result[5].end == 46

    # Cleanup - none
