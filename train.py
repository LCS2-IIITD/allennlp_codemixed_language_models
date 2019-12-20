import argparse
import logging
import os

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu
from allennlp.common import Params
from allennlp.common.util import (
    prepare_environment,
    prepare_global_logging,
    cleanup_global_logging,
    dump_metrics,
)
from allennlp.models.archival import archive_model, CONFIG_NAME
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from allennlp.training.trainer import Trainer
from allennlp.training.trainer_base import TrainerBase
from allennlp.training.trainer_pieces import TrainerPieces
from allennlp.training.util import create_serialization_dir, evaluate
from lm.modules.trainer import MultiTrainer

logger = logging.getLogger(__name__)


class Train(Subcommand):
    def add_subparser(
        self, name: str, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:

        description = """Train the specified model on the specified dataset."""
        subparser = parser.add_parser(name, description=description, help="Train a model.")

        subparser.add_argument(
            "param_path", type=str, help="path to parameter file describing the model to be trained"
        )

        subparser.add_argument(
            "-s",
            "--serialization-dir",
            required=True,
            type=str,
            help="directory in which to save the model and its logs",
        )

        subparser.add_argument(
            "-r",
            "--recover",
            action="store_true",
            default=False,
            help="recover training from the state in serialization_dir",
        )

        subparser.add_argument(
            "-f",
            "--force",
            action="store_true",
            required=False,
            help="overwrite the output directory if it exists",
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help="a JSON structure used to override the experiment configuration",
        )

        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help="outputs tqdm status on separate lines and slows tqdm refresh rate",
        )

        subparser.add_argument(
            "--cache-directory",
            type=str,
            default="",
            help="Location to store cache of data preprocessing",
        )

        subparser.add_argument(
            "--cache-prefix",
            type=str,
            default="",
            help="Prefix to use for data caching, giving current parameter "
            "settings a name in the cache, instead of computing a hash",
        )

        subparser.set_defaults(func=train_model_from_args)

        return subparser


def train_model_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to string paths.
    """
    train_model_from_file(
        args.param_path,
        args.serialization_dir,
        args.overrides,
        args.file_friendly_logging,
        args.recover,
        args.force,
        args.cache_directory,
        args.cache_prefix,
    )


def train_model_from_file(
    parameter_filename: str,
    serialization_dir: str,
    overrides: str = "",
    file_friendly_logging: bool = False,
    recover: bool = False,
    force: bool = False,
    cache_directory: str = None,
    cache_prefix: str = None,
) -> Model:
    # Load the experiment config from a file and pass it to ``train_model``.
    params = Params.from_file(parameter_filename, overrides)
    return train_model(
        params,
        serialization_dir,
        file_friendly_logging,
        recover,
        force,
        cache_directory,
        cache_prefix,
    )


def train_model(
    params: Params,
    serialization_dir: str,
    file_friendly_logging: bool = False,
    recover: bool = False,
    force: bool = False,
    cache_directory: str = None,
    cache_prefix: str = None,
) -> Model:
    create_serialization_dir(params, serialization_dir, recover, force)
    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    stdout_handler = prepare_global_logging(serialization_dir, file_friendly_logging)
    prepare_environment(params)

    cuda_device = params.params.get("trainer").get("cuda_device", -1)
    check_for_gpu(cuda_device)

    evaluate_on_test = params.pop_bool("evaluate_on_test", False)

    trainer_type = params.get("trainer", {}).get("type", "default")

    if True:
        # Special logic to instantiate backward-compatible trainer.
        pieces = TrainerPieces.from_params(
            params, serialization_dir, recover, cache_directory, cache_prefix
        )
        logger.info("Using MultiTrainer")
        trainer = MultiTrainer.from_params(
            model=pieces.model,
            serialization_dir=serialization_dir,
            iterator=pieces.iterator,
            train_data=pieces.train_dataset,
            validation_data=pieces.validation_dataset,
            params=pieces.params,
            validation_iterator=pieces.validation_iterator,
        )

        evaluation_iterator = pieces.validation_iterator or pieces.iterator
        evaluation_dataset = pieces.test_dataset

    else:
        if evaluate_on_test:
            raise ValueError(
                "--evaluate-on-test only works with the default Trainer. "
                "If you're using the CallbackTrainer you can use a callback "
                "to evaluate at Events.TRAINING_END; otherwise you'll have "
                "to run allennlp evaluate separately."
            )

        """
        The only main difference
        """
        print("Using MultuTrainer")
        logger.info("Using MultiTrainer")
        trainer = MultiTrainer.from_params(
            params, serialization_dir, recover, cache_directory, cache_prefix
        )
        evaluation_dataset = None

    params.assert_empty("base train command")

    try:
        metrics = trainer.train()
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
            logging.info(
                "Training interrupted by the user. Attempting to create "
                "a model archive using the current best epoch weights."
            )
            archive_model(serialization_dir, files_to_archive=params.files_to_archive)
        raise

    # Evaluate
    if evaluation_dataset and evaluate_on_test:
        logger.info("The model will be evaluated using the best epoch weights.")
        test_metrics = evaluate(
            trainer.model,
            evaluation_dataset,
            evaluation_iterator,
            cuda_device=trainer._cuda_devices[0],
            # TODO(brendanr): Pass in an arg following Joel's trainer refactor.
            batch_weight_key="",
        )

        for key, value in test_metrics.items():
            metrics["test_" + key] = value

    elif evaluation_dataset:
        logger.info(
            "To evaluate on the test set after training, pass the "
            "'evaluate_on_test' flag, or use the 'allennlp evaluate' command."
        )

    cleanup_global_logging(stdout_handler)

    # Now tar up results
    archive_model(serialization_dir, files_to_archive=params.files_to_archive)
    dump_metrics(os.path.join(serialization_dir, "metrics.json"), metrics, log=True)

    # We count on the trainer to have the model with best weights
    return trainer.model


if __name__ == '__main__':

    import os
    if os.environ.get("ALLENNLP_DEBUG"):
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=LEVEL)


    import allennlp.commands as comd

    subcommand_overrides = {
        "train": Train()
    }
    comd.main("sup", subcommand_overrides)