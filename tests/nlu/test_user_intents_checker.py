from moviebot.nlu.user_intents_checker import UserIntentsChecker
from moviebot.utterance.utterance import UserUtterance
from tests.mocks.mock_data_loader import MockDataLoader

from unittest.mock import patch
import pytest

# global setup
config = {
    'ontology': '',
    'database': '',
    'slot_values_path': '',
    'tag_words_slots_path': '',
}


@pytest.mark.parametrize('utterance',
                         [('Hello.'),
                          ('Hi, do you know of any interesting movies?')])
@patch('moviebot.nlu.user_intents_checker.DataLoader', new=MockDataLoader)
def test_check_hi_intent(utterance):
    # Setup
    uic = UserIntentsChecker(config)

    # Exercise
    uu = UserUtterance({'text': utterance})
    result = uic.check_hi_intent(uu)

    # Results
    assert len(result) == 1
    assert result[0].intent.name == 'HI'

    # Cleanup - none


@pytest.mark.parametrize('utterance',
                         [(''), ('I would like to watch an action movie.')])
@patch('moviebot.nlu.user_intents_checker.DataLoader', new=MockDataLoader)
def test_check_hi_intent_empty(utterance):
    # Setup
    uic = UserIntentsChecker(config)

    # Exercise
    uu = UserUtterance({'text': utterance})
    result = uic.check_hi_intent(uu)

    # Results
    assert len(result) == 0

    # Cleanup - none


@pytest.mark.parametrize('utterance', [
    ('I\'m happy with my result, bye'),
    ('exit'),
    ('I quit'),
])
@patch('moviebot.nlu.user_intents_checker.DataLoader', new=MockDataLoader)
def test_check_bye_intent(utterance):
    # Setup
    uic = UserIntentsChecker(config)

    # Exercise
    uu = UserUtterance({'text': utterance})
    result = uic.check_bye_intent(uu)

    # Results
    assert len(result) == 1
    assert result[0].intent.name == 'BYE'

    # Cleanup - none


@pytest.mark.parametrize('utterance',
                         [(''), ('Hi.'),
                          ('I would like to watch an action movie.')])
@patch('moviebot.nlu.user_intents_checker.DataLoader', new=MockDataLoader)
def test_check_bye_intent_empty(utterance):
    # Setup
    uic = UserIntentsChecker(config)

    # Exercise
    uu = UserUtterance({'text': utterance})
    result = uic.check_bye_intent(uu)

    # Results
    assert len(result) == 0

    # Cleanup - none
