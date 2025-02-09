"""
The `train_multitask` subcommand that can be used to train the model in the multitask fashion
It requires a configuration file and a directory in
which to write the results.
.. code-block:: bash
   $ allennlp train --help
   usage: allennlp train [-h] -s SERIALIZATION_DIR [-r] [-o OVERRIDES]
                         [--file-friendly-logging]
                         [--include-package INCLUDE_PACKAGE]
                         param_path
   Train the specified model on the specified dataset.
   positional arguments:
   param_path            path to parameter file describing the model to be
                           trained
   optional arguments:
   -h, --help            show this help message and exit
   -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                           directory in which to save the model and its logs
   -r, --recover         recover training from the state in serialization_dir
   -o OVERRIDES, --overrides OVERRIDES
                           a JSON structure used to override the experiment
                           configuration
   --include-package INCLUDE_PACKAGE
                           additional packages to include
   --file-friendly-logging
                           outputs tqdm status on separate lines and slows tqdm
                           refresh rate
"""
import random
from typing import Dict, Iterable, Tuple
import argparse
import logging
import os
import re

import torch

from allennlp.commands.evaluate import evaluate
from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common import Params
from allennlp.common.util import prepare_environment, prepare_global_logging, \
                                 get_frozen_and_tunable_parameter_names, dump_metrics
from allennlp.data import Vocabulary
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.archival import archive_model, CONFIG_NAME
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from allennlp.training.trainer import Trainer

from scibert.training.multitask_trainer_two_tasks import MultiTaskTrainer2
from scibert.training.vocabulary_multitask import VocabularyMultitask

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TrainMultiTask2(Subcommand):
    """ Class for training the model with two scaffold tasks """
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Train the specified model on the specified dataset.'''
        subparser = parser.add_parser(name, description=description, help='Train a model')

        subparser.add_argument('param_path',
                               type=str,
                               help='path to parameter file describing the model to be trained')

        subparser.add_argument('-s', '--serialization-dir',
                               required=True,
                               type=str,
                               help='directory in which to save the model and its logs')

        subparser.add_argument('-r', '--recover',
                               action='store_true',
                               default=False,
                               help='recover training from the state in serialization_dir')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        subparser.add_argument('--file-friendly-logging',
                               action='store_true',
                               default=False,
                               help='outputs tqdm status on separate lines and slows tqdm refresh rate')

        subparser.set_defaults(func=train_model_from_args)

        return subparser

def train_model_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to string paths.
    """
    train_model_from_file(args.param_path,
                          args.serialization_dir,
                          args.overrides,
                          args.file_friendly_logging,
                          args.recover)


def train_model_from_file(parameter_filename: str,
                          serialization_dir: str,
                          overrides: str = "",
                          file_friendly_logging: bool = False,
                          recover: bool = False) -> Model:
    """
    A wrapper around :func:`train_model` which loads the params from a file.

    Parameters
    ----------
    param_path : ``str``
        A json parameter file specifying an AllenNLP experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs. We just pass this along to
        :func:`train_model`.
    overrides : ``str``
        A JSON string that we will use to override values in the input parameter file.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we make our output more friendly to saved model files.  We just pass this
        along to :func:`train_model`.
    recover : ``bool`, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
    """
    # Load the experiment config from a file and pass it to ``train_model``.
    params = Params.from_file(parameter_filename, overrides)
    return train_model(params, serialization_dir, file_friendly_logging, recover)


def datasets_from_params(params: Params) -> Tuple[Dict[str, Iterable[Instance]], Dict[str, Iterable[Instance]], Dict[str, Iterable[Instance]]]:
    """
    Load all the datasets specified by the config.
    This includes the main dataset and the two scaffold auxiliary datasets
    """
    dataset_reader = DatasetReader.from_params(params.pop('dataset_reader'))
    validation_dataset_reader_params = params.pop("validation_dataset_reader", None)

    validation_and_test_dataset_reader: DatasetReader = dataset_reader
    if validation_dataset_reader_params is not None:
        logger.info("Using a separate dataset reader to load validation and test data.")
        validation_and_test_dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)

    train_data_path = params.pop('train_data_path')
    logger.info("Reading training data from %s", train_data_path)
    train_data = dataset_reader.read(train_data_path)

    datasets: Dict[str, Iterable[Instance]] = {"train": train_data}

    # 2. Auxillary training data.
    dataset_reader_aux = DatasetReader.from_params(params.pop('dataset_reader_aux'))
    train_data_path_aux = params.pop('train_data_path_aux')
    logger.info("Reading auxiliary training data from %s", train_data_path_aux)
    train_data_aux = dataset_reader_aux.read(train_data_path_aux)

    dataset_reader_aux2 = DatasetReader.from_params(params.pop('dataset_reader_aux2'))
    train_data_path_aux2 = params.pop('train_data_path_aux2')
    logger.info("Reading second auxiliary training data for from %s", train_data_path_aux2)
    train_data_aux2 = dataset_reader_aux2.read(train_data_path_aux2)

    # If only using a fraction of the auxiliary data.
    aux_sample_fraction = params.pop("aux_sample_fraction", 1.0)
    if aux_sample_fraction < 1.0:
        sample_size = int(aux_sample_fraction * len(train_data_aux))
        train_data_aux = random.sample(train_data_aux, sample_size)
        train_data_aux2 = random.sample(train_data_aux2, sample_size)

    # Balance the datasets by inflating the size of the smaller dataset to the size of the larger dataset.
    train_size = len(train_data)
    aux_train_size = len(train_data_aux)
    aux2_train_size = len(train_data_aux2)

    # Make second auxillary dataset the same size of the first auxiliary dataset
    if aux2_train_size > aux_train_size:
        train_data_aux2 = random.sample(train_data_aux2, aux_train_size)
    else:
        train_data_aux = random.sample(train_data_aux, aux2_train_size)

    # inflate training size to be as large as auxiliary training data
    if train_size > aux_train_size:
        difference = train_size - aux_train_size
        aux_sample = [random.choice(train_data_aux) for _ in range(difference)]
        train_data_aux = train_data_aux + aux_sample
        logger.info("Inflating auxiliary train data from {} to {} samples".format(
            aux_train_size, len(train_data_aux)))
    else:
        difference = aux_train_size - train_size
        train_sample = [random.choice(train_data) for _ in range(difference)]
        train_data = train_data + train_sample
        logger.info("Inflating train data from {} to {} samples".format(
            train_size, len(train_data)))

    datasets["train"] = train_data
    datasets_aux = {"train_aux": train_data_aux}
    datasets_aux2 = {"train_aux": train_data_aux2}

    validation_data_path = params.pop('validation_data_path', None)
    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = validation_and_test_dataset_reader.read(validation_data_path)
        datasets["validation"] = validation_data

    # Auxiliary validation data.
    validation_data_path_aux = params.pop('validation_data_path_aux', None)
    if validation_data_path_aux is not None:
        logger.info(f"Reading auxilliary validation data from {validation_data_path_aux}")
        validation_data_aux = dataset_reader_aux.read(validation_data_path_aux)
        datasets_aux["validation_aux"] = validation_data_aux
    else:
        validation_data_aux = None
    validation_data_path_aux2 = params.pop('validation_data_path_aux2', None)
    if validation_data_path_aux2 is not None:
        logger.info(f"Reading auxilliary validation data from {validation_data_path_aux2}")
        validation_data_aux2 = dataset_reader_aux2.read(validation_data_path_aux2)
        datasets_aux2["validation_aux"] = validation_data_aux2
    else:
        validation_data_aux2 = None

    test_data_path = params.pop("test_data_path", None)
    if test_data_path is not None:
        logger.info("Reading test data from %s", test_data_path)
        test_data = validation_and_test_dataset_reader.read(test_data_path)
        datasets["test"] = test_data

    # Auxillary test data
    test_data_path_aux = params.pop("test_data_path_aux", None)
    if test_data_path_aux is not None:
        logger.info(f"Reading auxiliary test data from {test_data_path_aux}")
        test_data_aux = dataset_reader_aux.read(test_data_path_aux)
        datasets_aux["test_aux"] = test_data_aux
    else:
        test_data_aux = None

    test_data_path_aux2 = params.pop("test_data_path_aux2", None)
    if test_data_path_aux2 is not None:
        logger.info(f"Reading auxillary test data from {test_data_path_aux2}")
        test_data_aux2 = dataset_reader_aux2.read(test_data_path_aux2)
        datasets_aux2["test_aux"] = test_data_aux2
    else:
        test_data_aux2 = None

    return datasets, datasets_aux, datasets_aux2

def create_serialization_dir(params: Params, serialization_dir: str, recover: bool) -> None:
    """
    This function creates the serialization directory if it doesn't exist.  If it already exists
    and is non-empty, then it verifies that we're recovering from a training with an identical configuration.

    Parameters
    ----------
    params: ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir: ``str``
        The directory in which to save results and logs.
    recover: ``bool``
        If ``True``, we will try to recover from an existing serialization directory, and crash if
        the directory doesn't exist, or doesn't match the configuration we're given.
    """
    if os.path.exists(serialization_dir) and os.listdir(serialization_dir):
        if not recover:
            raise ConfigurationError(f"Serialization directory ({serialization_dir}) already exists and is "
                                     f"not empty. Specify --recover to recover training from existing output.")

        logger.info(f"Recovering from prior training at {serialization_dir}.")

        recovered_config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(recovered_config_file):
            raise ConfigurationError("The serialization directory already exists but doesn't "
                                     "contain a config.json. You probably gave the wrong directory.")
        else:
            loaded_params = Params.from_file(recovered_config_file)

            # Check whether any of the training configuration differs from the configuration we are
            # resuming.  If so, warn the user that training may fail.
            fail = False
            flat_params = params.as_flat_dict()
            flat_loaded = loaded_params.as_flat_dict()
            for key in flat_params.keys() - flat_loaded.keys():
                logger.error(f"Key '{key}' found in training configuration but not in the serialization "
                             f"directory we're recovering from.")
                fail = True
            for key in flat_loaded.keys() - flat_params.keys():
                logger.error(f"Key '{key}' found in the serialization directory we're recovering from "
                             f"but not in the training config.")
                fail = True
            for key in flat_params.keys():
                if flat_params.get(key, None) != flat_loaded.get(key, None):
                    logger.error(f"Value for '{key}' in training configuration does not match that the value in "
                                 f"the serialization directory we're recovering from: "
                                 f"{flat_params[key]} != {flat_loaded[key]}")
                    fail = True
            if fail:
                raise ConfigurationError("Training configuration does not match the configuration we're "
                                         "recovering from.")
    else:
        if recover:
            raise ConfigurationError(f"--recover specified but serialization_dir ({serialization_dir}) "
                                     "does not exist.  There is nothing to recover from.")
        os.makedirs(serialization_dir, exist_ok=True)


def train_model(params: Params,
                serialization_dir: str,
                file_friendly_logging: bool = False,
                recover: bool = False) -> Model:
    """
    Trains the model specified in the given :class:`Params` object, using the data and training
    parameters also specified in that object, and saves the results in ``serialization_dir``.

    Parameters
    ----------
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    recover : ``bool``, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.

    Returns
    -------
    best_model: ``Model``
        The model with the best epoch weights.
    """
    prepare_environment(params)

    create_serialization_dir(params, serialization_dir, recover)
    prepare_global_logging(serialization_dir, file_friendly_logging)

    check_for_gpu(params.get('trainer').get('cuda_device', -1))

    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    all_datasets, all_datasets_aux, all_datasets_aux2 = datasets_from_params(params)
    datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))
    datasets_for_vocab_creation_aux = set(params.pop("auxiliary_datasets_for_vocab_creation", all_datasets_aux))
    datasets_for_vocab_creation_aux2 = set(params.pop("auxiliary_datasets_for_vocab_creation_2", all_datasets_aux2))


    mixing_ratio = params.pop_float("mixing_ratio")
    mixing_ratio2 = params.pop_float("mixing_ratio2")

    cutoff_epoch = params.pop("cutoff_epoch", -1)
    unfreeze_epoch = params.pop("unfreeze_epoch", -1)    

    for dataset in datasets_for_vocab_creation:
        if dataset not in all_datasets:
            raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

    logger.info("From dataset instances, %s will be considered for vocabulary creation.",
                ", ".join(datasets_for_vocab_creation))
    vocab_instances_aux = [
        instance for key, dataset in all_datasets_aux.items()
        for instance in dataset
        if key in datasets_for_vocab_creation_aux
    ]
    vocab_instances_aux.extend([
        instance for key, dataset in all_datasets_aux2.items()
        for instance in dataset
        if key in datasets_for_vocab_creation_aux2
    ])
    vocab = VocabularyMultitask.from_params(
            params.pop("vocabulary", {}),
            (instance for key, dataset in all_datasets.items()
             for instance in dataset
             if key in datasets_for_vocab_creation),
            instances_aux=vocab_instances_aux
    )
    model = Model.from_params(vocab=vocab, params=params.pop('model'))

    # Initializing the model can have side effect of expanding the vocabulary
    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

    iterator = DataIterator.from_params(params.pop("iterator"))
    iterator.index_with(vocab)

    iterator_aux = DataIterator.from_params(params.pop("iterator_aux"))
    iterator_aux.index_with(vocab)

    iterator_aux2 = DataIterator.from_params(params.pop("iterator_aux2"))
    iterator_aux2.index_with(vocab)

    validation_iterator_params = params.pop("validation_iterator", None)
    if validation_iterator_params:
        validation_iterator = DataIterator.from_params(validation_iterator_params)
        validation_iterator.index_with(vocab)
    else:
        validation_iterator = None

    # TODO: if validation in multi-task need to add validation iterator as above

    train_data = all_datasets.get('train')
    validation_data = all_datasets.get('validation')
    test_data = all_datasets.get('test')

    train_data_aux = all_datasets_aux.get('train_aux')
    validation_data_aux = all_datasets_aux.get('validation_aux')
    test_data_aux = all_datasets_aux.get('test_aux')

    train_data_aux2 = all_datasets_aux2.get('train_aux')
    validation_data_aux2 = all_datasets_aux2.get('validation_aux')
    test_data_aux2 = all_datasets_aux2.get('test_aux')

    trainer_params = params.pop("trainer")
    no_grad_regexes = trainer_params.pop("no_grad", ())
    for name, parameter in model.named_parameters():
        if any(re.search(regex, name) for regex in no_grad_regexes):
            parameter.requires_grad_(False)

    frozen_parameter_names, tunable_parameter_names = \
                   get_frozen_and_tunable_parameter_names(model)
    logger.info("Following parameters are Frozen  (without gradient):")
    for name in frozen_parameter_names:
        logger.info(name)
    logger.info("Following parameters are Tunable (with gradient):")
    for name in tunable_parameter_names:
        logger.info(name)

    trainer = MultiTaskTrainer2.from_params(model=model,
                                            serialization_dir=serialization_dir,
                                            iterator=iterator,
                                            iterator_aux=iterator_aux,
                                            iterator_aux2=iterator_aux2,
                                            train_data=train_data,
                                            train_data_aux=train_data_aux,
                                            train_data_aux2=train_data_aux2,
                                            mixing_ratio=mixing_ratio,
                                            mixing_ratio2=mixing_ratio2,
                                            cutoff_epoch=cutoff_epoch,
                                            unfreeze_epoch=unfreeze_epoch,
                                            validation_data_aux=validation_data_aux,
                                            validation_data_aux2=validation_data_aux2,
                                            validation_data=validation_data,
                                            params=trainer_params,
                                            validation_iterator=validation_iterator)

    evaluate_on_test = params.pop_bool("evaluate_on_test", False)
    evaluate_aux_on_test = params.pop_bool("evaluate_aux_on_test", False)
    params.assert_empty('base train command')

    try:
        metrics = trainer.train()
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
            logging.info("Training interrupted by the user. Attempting to create "
                         "a model archive using the current best epoch weights.")
            archive_model(serialization_dir, files_to_archive=params.files_to_archive)
        raise

    # Now tar up results
    archive_model(serialization_dir, files_to_archive=params.files_to_archive)

    logger.info("Loading the best epoch weights.")
    best_model_state_path = os.path.join(serialization_dir, 'best.th')
    best_model_state = torch.load(best_model_state_path)
    best_model = model
    best_model.load_state_dict(best_model_state)

    if test_data and evaluate_on_test:
        logger.info("The model will be evaluated using the best epoch weights.")
        test_metrics = evaluate(
                best_model, test_data, validation_iterator or iterator,
                cuda_device=trainer._cuda_devices[0] # pylint: disable=protected-access
        )
        for key, value in test_metrics.items():
            metrics["test_" + key] = value

    elif test_data:
        logger.info("To evaluate on the test set after training, pass the "
                    "'evaluate_on_test' flag, or use the 'allennlp evaluate' command.")

    if test_data_aux and evaluate_aux_on_test:
        # for instance in test_data_aux:
        #     instance.index_fields(vocab)
        # for instance in test_data_aux2:
        #     instance.index_fields(vocab)
        test_metrics_aux = evaluate(best_model, test_data_aux, iterator_aux,
                                    cuda_device=trainer._cuda_devices[0])  # pylint: disable=protected-access
        test_metrics_aux2 = evaluate(best_model, test_data_aux2, iterator_aux2,
                                     cuda_device=trainer._cuda_devices[0])  # pylint: disable=protected-access

        for key, value in test_metrics_aux.items():
            metrics["test_aux_" + key] = value
        for key, value in test_metrics_aux2.items():
            metrics["test_aux2_" + key] = value

    elif test_data_aux:
        logger.info("To evaluate on the auxiliary test set after training, pass the "
                    "'evaluate_on_test' flag, or use the 'allennlp evaluate' command.")

    dump_metrics(os.path.join(serialization_dir, "metrics.json"), metrics, log=True)

    return best_model
