import click
import os
import json

from jigsaw import main

@click.group()
def cli():
    """CLI for jigsaw"""


@cli.command()
@click.option('-c', '--config-filename', default=None, type=str,
              help='config file path (default: None)')
@click.option('-r', '--resume', default=None, type=str,
              help='path to latest checkpoint (default: None)')
@click.option('-d', '--device', default=None, type=str,
              help='indices of GPUs to enable (default: all)')
def train(config_filename, resume, device):
    if config_filename:
        # load config file
        with open(config_filename) as fh:
            config = json.load(fh)
        # setting path to save trained models and log files
        # path = os.path.join(config['trainer']['save_dir'], config['name'])

    elif resume:
        # load config from checkpoint if new config file is not given.
        # Use '--config' and '--resume' together to fine-tune trained model with changed configurations.
        config = torch.load(resume)['config']

    else:
        raise AssertionError('Configuration file need to be specified. '
                             'Add "-c experiments/config.json", for example.')

    if device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device

    main.train(config, resume)


@cli.command()
@click.option('-r', '--resume', default=None, type=str,
              help='path to latest checkpoint (default: None)')
@click.option('-d', '--device', default=None, type=str,
              help='indices of GPUs to enable (default: all)')
def test(resume, device):
    if resume:
        config = torch.load(resume)['config']
    if device:
        os.environ["CUDA_VISIBLE_DEVICES"] = device

    main.test(config, resume)
