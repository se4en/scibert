""" Module including helper functions for feature extraction and other stuff including metrics, jsonhandler, etc"""
import json
from collections import Counter
import logging
import re
from typing import List, Dict

import numpy as np
import string

from scibert.resources.lexicons import ALL_CONCEPT_LEXICONS, ALL_ACTION_LEXICONS

logger = logging.getLogger('classifier')

regex_find_citation = re.compile(r"\(\s?(([A-Za-z\-]+\s)+([A-Za-z\-]+\.?)?,?\s\d{2,4}[a-c]?(;\s)?)+\s?\)|"
                                 r"\[(\d{1,3},\s?)+\d{1,3}\]|"
                                 r"\[[\d,-]+\]|(\([A-Z][a-z]+, \d+[a-c]?\))|"
                                 r"([A-Z][a-z]+ (et al\.)? \(\d+[a-c]?\))|"
                                 r"[A-Z][a-z]+ and [A-Z][a-z]+ \(\d+[a-c]?\)]")

def print_top_words(model, feature_names, n_top_words):
    """ Prints top words in each topics for an LDA topic model"""
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


def get_values_from_list(inplst, key, is_class=True):
    """ gets a value of an obj for a list of dicts (inplst)
    Args:
        inplst: list of objects
        key: key of interest
        is_class: ist the input object a class or a dictionary obj
    """
    return [getattr(elem, key) for elem in inplst] if is_class \
        else [elem[key] for elem in inplst]


def partial_fmeasure_multilabel(y_true,
                                y_pred,
                                pos_labels_index: list,
                                neg_labels_index: list):
    """Calculate F-measure when partial annotations for each class is available
     In calculating the f-measure, this function only considers examples that are annotated for each class
     and ignores instances that are not annotated for that class
     A set of positive and negative labels identify the samples that are annotated
     This functions expects the input to be one hot encoding of labels plus negative labels
     For example if labels set are ['cat', 'dog'] we also want to encode ['not-cat', 'not-dog']
     This is because if an annotator only examines an instance for `cat` and says this is `cat`
     we want to ignore this instance in calculation of f-score for `dog` category.
     Therefore, the input should have a shape of (num_instances, 2 * num_classes)
     e.g., A one hot encoding corresponding to [`cat`, `dog`] would be:
        [`cat`, `dog`, `not-cat`, `not-dog`]
    A list of pos_labels_index and negative_labels_index identify the corresponding pos and neg labels
    e.g., For our example above, pos_labels_index=[0,1] and neg_labels_idnex=[2,3]

     Args:
        y_true: A 2D array of true class, shape = (num_instances, 2*num_classes)
            e.g. [[1,0,0,0], [0,0,1,0], [0,0,0,1], [0,1,0,0], [0,1,0,0], ...]
        y_pred: A 2D array of precitions, shape = (num_instances, 2*num_classes)
        pos_labels_index: 1D array of shape (num_classes)
        neg_labels_index: 1D array of shape (num_classes)

    returns:
        list of precision scores, list of recall scores, list of F1 scores
        The list is in the order of positive labels for each class
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    precisions = []
    recalls = []
    f1s = []
    supports = []
    for pos_class, neg_class in zip(pos_labels_index, neg_labels_index):
        predictions_pos = y_pred[:, pos_class]
        predictions_neg = y_pred[:, neg_class]

        gold_pos = y_true[:, pos_class]
        gold_neg = y_true[:, pos_class]

        # argmax_predictions = predictions.max(-1)[1].float().squeeze(-1)
        # True Negatives: correct non-positive predictions.
        correct_null_predictions = (predictions_neg == gold_neg).astype(float) * gold_neg
        _true_negatives = (correct_null_predictions.astype(float)).sum()

        # True Positives: correct positively labeled predictions.
        correct_non_null_predictions = (predictions_pos == gold_pos).astype(np.float) * predictions_pos
        _true_positives = correct_non_null_predictions.sum()

        # False Negatives: incorrect negatively labeled predictions.
        incorrect_null_predictions = (predictions_pos != gold_pos).astype(np.float) * gold_pos
        _false_negatives = incorrect_null_predictions.sum()

        # False Positives: incorrect positively labeled predictions
        incorrect_non_null_predictions = (predictions_pos != gold_pos).astype(np.float) * predictions_pos
        _false_positives = incorrect_non_null_predictions.sum()

        precision = float(_true_positives) / float(_true_positives + _false_positives + 1e-13)
        recall = float(_true_positives) / float(_true_positives + _false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))

        support = (gold_pos + gold_neg).sum()

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1_measure)
        supports.append(support)

    return precisions, recalls, f1s, supports

def format_classification_report(precisions, recalls, f1s, supports, labels, digits=4):
    last_line_heading = 'avg / total'

    if labels is None:
        target_names = [u'%s' % l for l in labels]
    else:
        target_names = labels
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'
    rows = zip(labels, precisions, recalls, f1s, supports)
    for row in rows:
        report += row_fmt.format(*row, width=5, digits=digits)

    report += u'\n'

    # compute averages
    report += row_fmt.format(last_line_heading,
                             np.average(precisions, weights=supports),
                             np.average(recalls, weights=supports),
                             np.average(f1s, weights=supports),
                             np.sum(supports),
                             width=width, digits=digits)
    return report


class JsonFloatEncoder(json.JSONEncoder):
    """ numpy floats are not json serializable
    This class is a json encoder that enables dumping
    json objects that have numpy numbers in them
    use: json.dumps(obj, cls=JsonFLoatEncoder)"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonFloatEncoder, self).default(obj)


MIN_TOKEN_COUNT = 8
MAX_TOKEN_COUNT = 200
MIN_WORD_TOKENS_RATIO = 0.40
MIN_LETTER_CHAR_RATIO = 0.50
MIN_PRINTABLE_CHAR_RATIO = 0.95


# util function adopted from github.com/allenai/relex
def is_sentence(sentence: str) -> bool:
    if not isinstance(sentence, str):
        return False
    num_chars = len(sentence)
    tokens = sentence.split(' ')
    num_tokens = len(tokens)
    if num_tokens < MIN_TOKEN_COUNT:
        return False
    if num_tokens > MAX_TOKEN_COUNT:
        return False
    # Most tokens should be words
    if sum([t.isalpha() for t in tokens]) / num_tokens < MIN_WORD_TOKENS_RATIO:
        return False
    # Most characters should be letters, not numbers and not special characters
    if sum([c in string.ascii_letters for c in sentence]) / num_chars < MIN_LETTER_CHAR_RATIO:
        return False
    # Most characters should be printable
    if sum([c in string.printable for c in sentence]) / num_chars < MIN_PRINTABLE_CHAR_RATIO:
        return False
    return True

def _is_in_lexicon(lexicon, sentence, si, ArgType=None, required_pos=None):
    for cur_phrase in lexicon:
        phrase = cur_phrase.split(' ')

        # Can't match phrases that would extend beyond this sentence
        if len(phrase) + si > len(sentence):
            continue

        found = True
        found_arg = False

        for i, lemma in enumerate(phrase):
            # Check the word form too, just to prevent weird lemmatization
            # issues (usually for adjectives)
            if not (sentence[si+i]['lemma'] == lemma or sentence[si+i]['word'] == lemma) \
                    or not (required_pos is None or sentence[si+i]['pos'][0] == required_pos):
                found = False
                break
            if ArgType is not None and sentence[si+i]['ArgType'] == ArgType:
                found_arg = True
        if found and (ArgType is None or found_arg):
            #if len(phrase) > 1:
            #    print '~~~~~~Matched %s' % (' '.join(phrase))
            return True, len(phrase)

    return False, 0


def find(pattern: List[str], sentence: List[dict], must_have_subj_value: bool, feature=None) -> int:
    # if debug is not None:
    #    print json.dumps(sentence)

    # if debug is not None:
    #     print('\n\ntesting %s (%s) against "%s" (must be subj? %s)' % (pattern, feature, debug, must_have_subj_value))

    # For each position in the sentence
    for sent_pos in range(0, (len(sentence) - len(pattern)) + 1):

        match = True
        is_subj = False
        # This is the adjustment to the sentence's token offset based on finding
        # a MWE match in a lexicon
        k = 0

        # if debug is not None:
        #     print('starting search at ' + sentence[sent_pos]['word'])

        for pat_pos in range(0, len(pattern)):

            # if debug is not None:
            #     print('%d:%d:%d -> "%s" in "%s"?' % (sent_pos, pat_pos, k, sentence[sent_pos + pat_pos + k]['lemma'], pattern[pat_pos]))

            # Check that we won't search outside the sentence length due to
            # finding a MWE lexicon entry at the end of the sentence
            if sent_pos + pat_pos + k >= len(sentence):
                # if debug is not None:
                #     print('moved beyond end of sentence :(')
                match = False
                break

            # print '%d %s' % (sent_pos+pat_pos+k, sentence[sent_pos+pat_pos+k]['ArgType'])
            if sentence[sent_pos + pat_pos + k]['ArgType'] == 'subj':
                is_subj = True

            cur_pat_i = pattern[pat_pos]
            # if debug is not None:
            #     print('Testing %d/%d: %s' % (pat_pos + 1, pat_len, cur_pat_i))

            # If this is a category that we have to look up
            if cur_pat_i[0] == '@':
                label = cur_pat_i[1:]

                # if debug is not None:
                #     print('Checking if "%s" is in %s' % (sentence[sent_pos + pat_pos + k]['lemma'], label))

                lexicon = None
                required_pos = None
                if label in ALL_CONCEPT_LEXICONS:
                    lexicon = ALL_CONCEPT_LEXICONS[label]
                elif label in ALL_ACTION_LEXICONS:
                    lexicon = ALL_ACTION_LEXICONS[label]
                    required_pos = 'V'

                if lexicon is None:
                    # raise BaseException(("unknown lexicon ref: '%s' in %s, %s" % (label, feature, pattern)))
                    return -1

                (is_match, matched_phrased_length) = _is_in_lexicon(lexicon, sentence, \
                                                                    sent_pos+pat_pos+k, required_pos=required_pos)

                # print 'found %s (%d) in %s? %s (%d)' % (sentence[sent_pos+pat_pos+k]['lemma'], sent_pos+pat_pos+k, label, is_match, matched_phrased_length)

                if not is_match:
                    match = False
                    break
                # else:
                #     if debug is not None:
                #         print('YAY:: "%s" is in set %s in %s' % (sentence[sent_pos + pat_pos + k]['lemma'], label, debug))

                # If we did find a match, recognize that some phrases are
                # multi-word expressions, so we may need to skip ahead more than
                # one token.  Note that we were already going to skip one token
                # anyway, so substract 1 from the phrase length
                k += (matched_phrased_length - 1)

                # if not sentence[sent_pos+pat_pos+k]['lemma'] not in lexicon:
                #    if debug is not None:
                #        print '"%s" is not in set %s in %s' % (sentence[sent_pos+pat_pos+k]['lemma'], label, debug)
                #    match = False
                #    break
                # else:
                #    if debug is not None:

            elif cur_pat_i == 'SELFCITATION':
                # if debug is not None:
                #     print('Checking if "%s" is %s' % (sentence[sent_pos + pat_pos + k]['pos'][0], cur_pat_i[1]))

                if sentence[sent_pos + pat_pos + k]['word'] != cur_pat_i:
                    # if debug is not None:
                    #     print('"%s" is not a %s in %s' % (sentence[sent_pos + pat_pos + k]['lemma'], cur_pat_i, debug))
                    match = False
                    break
                # else:
                #     if debug is not None:
                #         print('YAY:: "%s" is a %s in %s' % (sentence[sent_pos + pat_pos + k]['lemma'], cur_pat_i, debug))

            elif cur_pat_i == 'CITATION':
                # if debug is not None:
                #     print('Checking if "%s" is %s' % (sentence[sent_pos + pat_pos + k]['pos'][0], cur_pat_i[1]))

                if not sentence[sent_pos + pat_pos + k]['word'].endswith(cur_pat_i):
                    # if debug is not None:
                    #     print('"%s" is not a %s in %s' % (sentence[sent_pos + pat_pos + k]['lemma'], cur_pat_i, debug))
                    match = False
                    break
                # else:
                #     if debug is not None:
                #         print('YAY:: "%s" is a %s in %s' % (sentence[sent_pos + pat_pos + k]['lemma'], cur_pat_i, debug))


            # Not sure if this is entirely right...
            elif cur_pat_i == 'CREF':
                if sentence[sent_pos + pat_pos + k]['pos'] != 'CD' \
                        or sentence[sent_pos + pat_pos + k]['word'] != 'CREF':
                    match = False
                    break

            # If this is POS-match
            elif cur_pat_i[0] == '#':
                # if debug is not None:
                #     print('Checking if "%s" is  %s' % (sentence[sent_pos + pat_pos + k]['pos'][0], cur_pat_i[1]))
                # NOTE: we compare only the coarsest POS tag level (N/V/J)
                #
                # NOTE Check for weird POS-tagging issues with verbal adjectives
                if sentence[sent_pos + pat_pos + k]['pos'][0] != cur_pat_i[1] \
                        and not (cur_pat_i[1] == 'J' and sentence[sent_pos + pat_pos + k]['pos'] == 'VBN'):
                    match = False
                    break
                    # if debug is not None:
                    #     print('"%s" is not %s in %s' % (sentence[sent_pos + pat_pos + k]['pos'][0], cur_pat_i[1], debug))
                # else:
                #     if debug is not None:
                #         print('"YAY:: %s" is %s in %s' % (sentence[sent_pos + pat_pos + k]['pos'][0], cur_pat_i[1], debug))

            # Otherwise, we have to match the word
            else:
                # if debug is not None:
                #     print('Checking if "%s" is %s' % (sentence[sent_pos + pat_pos + k]['lemma'], cur_pat_i))
                if sentence[sent_pos + pat_pos + k]['lemma'] != cur_pat_i:
                    # if debug is not None:
                    #     print('"%s" is not %s in %s' % (sentence[sent_pos + pat_pos + k]['lemma'], cur_pat_i, debug))
                    match = False
                    break
                # else:
                #     if debug is not None:
                #         print('YAY:: "%s" is %s in %s' % (sentence[sent_pos + pat_pos + k]['lemma'], cur_pat_i, debug))

        if match and (must_have_subj_value is not None) and (is_subj is not must_have_subj_value):
            # if debug is not None:
            #     print(
            #         'needed a subject for %s but this isn\'t one (%s != %s)' % (feature, is_subj, must_have_subj_value))
            continue

        # TODO: confirm we can skip 'pat_pos' items so sent_pos += pat_pos
        if match:
            # if debug is not None:
            #     print('match!\n\n')
            return sent_pos
        # else:
        #     if debug is not None:
        #         print('no match (%d, %d, %d)\n\n' % (sent_pos, pat_pos, k))

    # if debug is not None:
    #     print('\n\n')

    return -1
