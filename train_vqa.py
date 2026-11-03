"""Train a decoder to answer questions about images with two shapes, e.g. "what color is the star ?" -> "magenta".

The image encoder is taken from a checkpoint of train_multi_captioner.py and frozen by default
(similar to LLaVA, which first aligns image and text with captions and then fine-tunes on instructions).
The decoder is trained from scratch; the loss is computed on the answer only, not on the question.
"""
import argparse
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from my_utils import get_device, log_message
from torch.utils.data import DataLoader

import data.multi_shapes as ms
import data.vqa_shapes as vqa
from captioning import create_run, decode_tokens, log_wrong_predictions, plot_examples, save_checkpoint, train_epoch
from train_multi_captioner import load_multi_captioner
from vlm_models import CaptionDecoder, SpatialImageEncoder


def predict_answers(image_encoder, decoder, loader, vocabulary, inverse_vocabulary, device, blind=False) -> dict:
    """Generates an answer for every question of a loader (created with vqa.collate_eval_batch).

    :param blind: (bool) If True, every image is replaced by an empty (white) image. Shows how well the
        questions can be answered without looking at the image.
    :return: (dict) lists of questions, true answers, predicted answers, question types, question objects,
        and image objects
    """
    image_encoder.eval()
    decoder.eval()
    results = defaultdict(list)
    with torch.no_grad():
        for images, prompts, questions, answers, question_types, about_objects, image_objects in loader:
            images = images.to(device)
            if blind:
                images = torch.ones_like(images)
            token_ids = decoder.generate(image_encoder(images), prompts.to(device), vocabulary['<eos>'])
            results['predictions'].extend(
                decode_tokens(token_ids, inverse_vocabulary, vocabulary['<eos>'], vocabulary['<pad>'])
            )
            results['questions'].extend(questions)
            results['answers'].extend(answers)
            results['question_types'].extend(question_types)
            results['about_objects'].extend(about_objects)
            results['image_objects'].extend(image_objects)
    return results


def evaluate(image_encoder, decoder, loader, vocabulary, inverse_vocabulary, held_out_objects, device,
             blind=False) -> dict:
    """Answers all questions of a loader and computes the accuracies.

    :return: (dict) accuracy overall, per question type, for location questions on images with two objects
        of the same color, and for questions about held-out / other objects, counts of wrong answers,
        and the lists returned by predict_answers()
    """
    results = predict_answers(image_encoder, decoder, loader, vocabulary, inverse_vocabulary, device, blind)
    correct, total = defaultdict(lambda: 0), defaultdict(lambda: 0)
    wrong_examples = defaultdict(lambda: 0)
    for question, answer, prediction, question_type, about_objects, (left_object, right_object) in zip(
            results['questions'], results['answers'], results['predictions'],
            results['question_types'], results['about_objects'], results['image_objects']):
        is_correct = prediction == answer
        about = 'held-out object' if any(obj in held_out_objects for obj in about_objects) else 'other objects'
        groups = ['all', question_type, about]
        # The hard case for location: the object cannot be found by its color alone
        if question_type == 'location' and left_object[0] == right_object[0]:
            groups.append('location same color')
        for group in groups:
            correct[group] += is_correct
            total[group] += 1
        if not is_correct:
            wrong_examples[(f'{question} {answer}', prediction)] += 1
    groups = ('all', *vqa.QUESTION_TYPES, 'location same color', 'held-out object', 'other objects')
    results['accuracy'] = {group: correct[group] / total[group] for group in groups if total[group]}
    results['wrong_examples'] = wrong_examples
    return results


def format_results(name, results) -> str:
    return f'{name}: ' + ', '.join(f'{group} {value:.1%}' for group, value in results['accuracy'].items())


def plot_dataset_examples(train_dataset, test_dataset, n_examples, figure_name):
    """Saves a figure with training samples (top row) and held-out test samples (bottom row)."""
    figure, axes = plt.subplots(2, n_examples, figsize=(3 * n_examples, 7.5), squeeze=False)
    for row_axes, dataset, name in zip(axes, (train_dataset, test_dataset), ('Train', 'Held-out')):
        for index, axis in enumerate(row_axes):
            image, question, answer, question_type, *_ = dataset[index]
            axis.imshow(image.permute(1, 2, 0))
            axis.set_title(f'{name} ({question_type}):\n{question}\n-> {answer}', fontsize=9)
            axis.axis('off')
    figure.tight_layout()
    figure.savefig(figure_name)
    plt.close(figure)


# Arguments which can be changed when resuming a training; all others are taken from the checkpoint
RESUME_ARGUMENTS = ('resume_from', 'epochs', 'save_path', 'no_gpu', 'visual_examples', 'plot_dataset_only')


def resume_arguments(args) -> tuple:
    """Loads the checkpoint given by args.resume_from and combines its config with the resume arguments.

    :return: (tuple) arguments for the resumed training, loaded checkpoint
    """
    checkpoint = torch.load(args.resume_from, map_location='cpu', weights_only=True)
    resumed_args = argparse.Namespace(**checkpoint['config'])
    for name in RESUME_ARGUMENTS:
        setattr(resumed_args, name, getattr(args, name))
    # A frozen encoder can be unfrozen when resuming (but not the other way round)
    if args.train_encoder:
        resumed_args.train_encoder = True
    # Keep the learning rate and the training data of the checkpoint unless new values are given
    if args.lr is not None:
        resumed_args.lr = args.lr
    if args.same_color_fraction is not None:
        resumed_args.same_color_fraction = args.same_color_fraction
    elif not hasattr(resumed_args, 'same_color_fraction'):
        # Checkpoints from before this option
        resumed_args.same_color_fraction = 0.0
    return resumed_args, checkpoint


def trainable_parameter_names(image_encoder, decoder, train_encoder) -> list:
    """Names of the trained parameters, in the order in which they are passed to the optimizer."""
    names = [f'image_encoder.{name}' for name, _ in image_encoder.named_parameters()] if train_encoder else []
    return names + [f'decoder.{name}' for name, _ in decoder.named_parameters()]


def load_optimizer_state(optimizer, saved_state, saved_names, names):
    """Loads the optimizer state of a checkpoint, matching the parameters by name.

    The saved state only contains the parameters which were trained before. If more parameters are trained now
    (e.g. the encoder was unfrozen), these start with a fresh state.

    :param saved_state: (dict) optimizer state dict of the checkpoint
    :param saved_names: (list) names of the parameters in saved_state, in optimizer order
    :param names: (list) names of the parameters of optimizer, in optimizer order
    """
    saved_index = {name: index for index, name in enumerate(saved_names)}
    state = {
        index: saved_state['state'][saved_index[name]]
        for index, name in enumerate(names)
        if name in saved_index and saved_index[name] in saved_state['state']
    }
    group = dict(saved_state['param_groups'][0], params=list(range(len(names))))
    optimizer.load_state_dict({'state': state, 'param_groups': [group]})


def create_image_encoder(args, device, log_file, state_dict=None) -> SpatialImageEncoder:
    """Loads the image encoder from a train_multi_captioner.py checkpoint, or creates a new one.

    The encoder settings in args (image size, grid size, channels) are set to those of the checkpoint,
    so that the saved config describes the model correctly.

    :param state_dict: (dict | None) encoder weights of a train_vqa.py checkpoint when resuming a training;
        args then already contains the encoder settings of that checkpoint
    """
    if state_dict is not None:
        image_encoder = SpatialImageEncoder(
            img_channels=3, out_channels=args.encoder_channels, grid_size=args.grid_size,
        ).to(device)
        image_encoder.load_state_dict(state_dict)
        if args.checkpoint_path and not args.train_encoder:
            image_encoder.requires_grad_(False)
        return image_encoder
    if not args.checkpoint_path:
        return SpatialImageEncoder(
            img_channels=3, out_channels=args.encoder_channels, grid_size=args.grid_size,
        ).to(device)
    image_encoder, _, _, captioner_config = load_multi_captioner(args.checkpoint_path, device)
    args.image_size = captioner_config.image_size
    args.grid_size = captioner_config.grid_size
    args.encoder_channels = captioner_config.encoder_channels
    log_message(
        log_file, f'Loaded image encoder from {args.checkpoint_path} '
                  f'(grid size {args.grid_size}, {args.encoder_channels} channels, {args.image_size} px)'
    )
    if not args.train_encoder:
        image_encoder.requires_grad_(False)
    return image_encoder


def create_datasets(args, train_objects, held_out_objects) -> tuple:
    """Returns the training, train-evaluation and held-out test data sets (with the same images as the captioner)."""
    question_types = tuple(args.question_types)
    # Only the training images get more objects of the same color; the evaluation sets keep the natural mix
    train_dataset = vqa.VQAShapeDataset(
        ms.MultiShapeDataset(
            train_objects, args.train_samples, image_size=args.image_size, seed=args.seed,
            same_color_fraction=args.same_color_fraction,
        ),
        question_types, seed=args.seed + 300000,
    )
    # Fresh images with training objects only
    train_eval_dataset = vqa.VQAShapeDataset(
        ms.MultiShapeDataset(train_objects, args.test_samples, image_size=args.image_size, seed=args.seed + 200000),
        question_types, seed=args.seed + 400000,
    )
    # Every test image contains one held-out object and one training object
    test_dataset = vqa.VQAShapeDataset(
        ms.MultiShapeDataset(
            train_objects, args.test_samples, image_size=args.image_size, seed=args.seed + 100000,
            required_objects=held_out_objects,
        ),
        question_types, seed=args.seed + 500000,
    )
    return train_dataset, train_eval_dataset, test_dataset


def main(args):
    """Train the VQA decoder (and optionally the image encoder) and evaluate it on held-out objects."""
    checkpoint, start_epoch = None, 0
    if args.resume_from:
        args, checkpoint = resume_arguments(args)
        start_epoch = checkpoint['epoch']
    else:
        args.lr = 1e-4 if args.lr is None else args.lr
        args.same_color_fraction = args.same_color_fraction or 0.0
    # + start_epoch: a resumed training does not repeat the data order of the first epochs
    torch.manual_seed(args.seed + start_epoch)
    random.seed(args.seed + start_epoch)
    np.random.seed(args.seed + start_epoch)
    device = get_device(args)
    run_directory, log_file = create_run(args, device)
    if checkpoint is not None:
        log_message(log_file, f'Resumed from {args.resume_from} (epoch {start_epoch})')

    image_encoder = create_image_encoder(
        args, device, log_file, state_dict=None if checkpoint is None else checkpoint['image_encoder'],
    )
    vocabulary = vqa.make_vocabulary()
    inverse_vocabulary = {index: word for word, index in vocabulary.items()}
    _, train_objects, held_out_objects = ms.get_objects()

    train_dataset, train_eval_dataset, test_dataset = create_datasets(args, train_objects, held_out_objects)
    plot_dataset_examples(
        train_dataset, test_dataset, args.visual_examples, os.path.join(run_directory, 'dataset_examples.png')
    )
    if args.plot_dataset_only:
        log_file.close()
        return
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda batch: vqa.collate_train_batch(batch, vocabulary),
    )
    eval_collate_fn = lambda batch: vqa.collate_eval_batch(batch, vocabulary)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=args.batch_size, collate_fn=eval_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=eval_collate_fn)

    decoder = CaptionDecoder(
        len(vocabulary), image_encoder.latent_dim, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, max_length=vqa.MAX_PROMPT_LENGTH + vqa.MAX_ANSWER_WORDS,
        dropout=args.dropout, padding_idx=vocabulary['<pad>'], n_image_tokens=image_encoder.n_tokens,
    ).to(device)
    parameters = [p for p in (*image_encoder.parameters(), *decoder.parameters()) if p.requires_grad]
    log_message(log_file, f'Trainable parameters: {sum(p.numel() for p in parameters):,}')
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    train_encoder = not args.checkpoint_path or args.train_encoder
    if checkpoint is not None:
        decoder.load_state_dict(checkpoint['decoder'])
        saved_config = checkpoint['config']
        saved_train_encoder = not saved_config['checkpoint_path'] or saved_config['train_encoder']
        # Restores Adam's running averages, so the training continues smoothly
        load_optimizer_state(
            optimizer, checkpoint['optimizer'],
            trainable_parameter_names(image_encoder, decoder, saved_train_encoder),
            trainable_parameter_names(image_encoder, decoder, train_encoder),
        )
        for group in optimizer.param_groups:
            group['lr'] = args.lr
        if train_encoder and not saved_train_encoder:
            log_message(log_file, 'Image encoder unfrozen; its optimizer state starts fresh')

    for epoch in range(start_epoch, start_epoch + args.epochs):
        mean_loss = train_epoch(
            image_encoder, None, decoder, train_loader, optimizer, vocabulary['<pad>'], train_encoder, device,
        )
        train_results = evaluate(
            image_encoder, decoder, train_eval_loader, vocabulary, inverse_vocabulary, held_out_objects, device,
        )
        test_results = evaluate(
            image_encoder, decoder, test_loader, vocabulary, inverse_vocabulary, held_out_objects, device,
        )
        blind_results = evaluate(
            image_encoder, decoder, test_loader, vocabulary, inverse_vocabulary, held_out_objects, device,
            blind=True,
        )
        log_message(log_file, f'Epoch {epoch + 1:3d} | Loss: {mean_loss:.4f}')
        log_message(log_file, '    ' + format_results('Train objects  ', train_results))
        log_message(log_file, '    ' + format_results('Held-out object', test_results))
        log_message(log_file, '    ' + format_results('Blind (no image)', blind_results))
        save_checkpoint(run_directory, epoch + 1, decoder, image_encoder, None, optimizer, vocabulary, args)

    log_wrong_predictions(log_file, test_results['wrong_examples'], max_lines=20)
    log_message(log_file, 'Held-out objects: ' + ', '.join(f'{color} {shape}' for color, shape in held_out_objects))
    plot_examples(
        test_dataset, test_results['answers'], test_results['predictions'], args.visual_examples,
        os.path.join(run_directory, 'held_out_examples.png'), seed=args.seed, prompts=test_results['questions'],
    )
    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a decoder to answer questions about images with two colored shapes'
    )
    parser.add_argument(
        '--resume-from',
        help='checkpoint of train_vqa.py to continue training from (for --epochs more epochs); all settings except '
             '--epochs, --lr, --train-encoder, --same-color-fraction, --save-path, --no-gpu, --visual-examples '
             'are taken from it',
    )
    parser.add_argument(
        '--checkpoint-path',
        default='out_multi_captioner/20261031-224448-encoder-dim-32/captioner_epoch_020.pt',
        help="checkpoint of train_multi_captioner.py whose image encoder is used; "
             "pass '' to train a new image encoder from scratch",
    )
    parser.add_argument(
        '--train-encoder', action='store_true',
        help='fine-tune the loaded image encoder instead of keeping it frozen',
    )
    parser.add_argument(
        '--question-types', nargs='+', choices=vqa.QUESTION_TYPES, default=vqa.QUESTION_TYPES,
        help='question types used for training and evaluation',
    )
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    # Only used without a checkpoint; otherwise taken from the checkpoint
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--grid-size', type=int, default=4)
    parser.add_argument('--encoder-channels', type=int, default=32)
    parser.add_argument(
        '--same-color-fraction', type=float,
        help='fraction of the training images whose two objects have the same color (but different shapes), '
             'so that objects cannot be found by color alone (default: 0, i.e. both objects are chosen '
             'independently, about 15%% same color; when resuming: the value of the checkpoint)',
    )
    parser.add_argument('--train-samples', type=int, default=20000)
    parser.add_argument('--test-samples', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument(
        '--lr', type=float,
        help='learning rate (default: 1e-4, or the learning rate of the checkpoint when resuming)',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--visual-examples', type=int, default=4,
        help='number of examples per row in dataset_examples.png and held_out_examples.png',
    )
    parser.add_argument(
        '--plot-dataset-only', action='store_true',
        help='only save dataset_examples.png to check the generated questions, no training',
    )
    parser.add_argument('--save-path', type=str, default='out_vqa')
    parser.add_argument('--no-gpu', action='store_true')
    main(parser.parse_args())
