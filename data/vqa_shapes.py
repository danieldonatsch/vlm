"""Visual question answering on the two-shape images of data/multi_shapes.py.

Every sample is an image with one question about it, e.g. "what color is the star ?" -> "magenta".
The decoder input is a prompt followed by the answer:
    <pad> ... <pad> <bos> what color is the star ? <ans> magenta <eos>
Prompts are padded on the left to a fixed length, so every answer starts at the same position.
"""

import random

import torch
from torch.utils.data import Dataset

from data.colored_shapes import COLORS, SHAPES, tokenize
from data.multi_shapes import parse_caption

QUESTION_TYPES = ('color', 'shape', 'relation', 'location')

# Longest question: "what is right of the red circle ?" (8 words), plus <bos> and <ans>
MAX_PROMPT_LENGTH = 10
# Longest answer: "red circle"
MAX_ANSWER_WORDS = 2


def make_vocabulary() -> dict:
    """Creates a map (dict) which maps each word of the questions and answers to an index."""
    words = (
        '<pad>', '<unk>', *COLORS, *SHAPES, 'left', 'right', 'of',
        'what', 'color', 'shape', 'is', 'the', 'object', 'where', '?', '<bos>', '<ans>', '<eos>',
    )
    return {word: index for index, word in enumerate(words)}


def candidate_questions(left_object, right_object, question_type) -> list:
    """Returns all questions of one type which have exactly one correct answer for this image.

    :return: (list) tuples of question, answer, and the objects the question is about
    """
    objects = (left_object, right_object)
    candidates = []
    if question_type == 'color':
        # sorted(): the order of a set of strings changes from run to run
        for shape in sorted({obj[1] for obj in objects}):
            referents = [obj for obj in objects if obj[1] == shape]
            # Two stars are fine, as long as they have the same color
            if len({obj[0] for obj in referents}) == 1:
                candidates.append((f'what color is the {shape} ?', referents[0][0], referents))
    elif question_type == 'shape':
        for color in sorted({obj[0] for obj in objects}):
            referents = [obj for obj in objects if obj[0] == color]
            if len({obj[1] for obj in referents}) == 1:
                candidates.append((f'what shape is the {color} object ?', referents[0][1], referents))
    elif left_object != right_object:
        # Relation and location questions need a unique referent, so the two objects must differ
        left_name, right_name = ' '.join(left_object), ' '.join(right_object)
        if question_type == 'relation':
            candidates.append((f'what is left of the {right_name} ?', left_name, list(objects)))
            candidates.append((f'what is right of the {left_name} ?', right_name, list(objects)))
        elif question_type == 'location':
            candidates.append((f'where is the {left_name} ?', 'left', [left_object]))
            candidates.append((f'where is the {right_name} ?', 'right', [right_object]))
    return candidates


class VQAShapeDataset(Dataset):
    """Adds one question per image to a MultiShapeDataset.

    Items are tuples of image, question, answer, question type, the objects the question is about,
    and both objects of the image (left, right).
    """

    def __init__(self, image_dataset, question_types=QUESTION_TYPES, seed=0):
        """
        :param image_dataset: (MultiShapeDataset) data set providing the images and captions
        :param question_types: (tuple) question types to choose from (subset of QUESTION_TYPES)
        :param seed: (int) random seed for choosing the questions
        """
        self.image_dataset = image_dataset
        self.question_types = question_types
        self.seed = seed

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, index):
        image, caption = self.image_dataset[index]
        left_object, right_object = parse_caption(caption)
        rng = random.Random(self.seed * 1000003 + index)
        candidates_per_type = {
            question_type: candidate_questions(left_object, right_object, question_type)
            for question_type in self.question_types
        }
        # First choose the type (so all types are equally frequent), then a question of this type.
        # Color and shape questions are always possible, relation and location not for two equal objects.
        question_type = rng.choice([t for t, candidates in candidates_per_type.items() if candidates])
        question, answer, about_objects = rng.choice(candidates_per_type[question_type])
        return image, question, answer, question_type, about_objects, (left_object, right_object)


def tokenize_prompts(questions, vocabulary) -> torch.Tensor:
    """Returns "<bos> question <ans>" token ids, padded on the left to MAX_PROMPT_LENGTH, shape (B, MAX_PROMPT_LENGTH)."""
    prompts = torch.full((len(questions), MAX_PROMPT_LENGTH), vocabulary['<pad>'], dtype=torch.long)
    for row, question in enumerate(questions):
        token_ids = tokenize(f'<bos> {question} <ans>', vocabulary)
        prompts[row, MAX_PROMPT_LENGTH - len(token_ids):] = token_ids
    return prompts


def collate_train_batch(batch, vocabulary) -> tuple:
    """Collates samples for training: the loss is computed on the answer and <eos> only.

    :return: (tuple) images, input ids, and target ids. Target ids are <pad> (= no loss) for the prompt.
    """
    images, questions, answers = zip(*(item[:3] for item in batch))
    prompts = tokenize_prompts(questions, vocabulary)
    answer_ids = torch.nn.utils.rnn.pad_sequence(
        [tokenize(f'{answer} <eos>', vocabulary) for answer in answers],
        batch_first=True, padding_value=vocabulary['<pad>'],
    )
    token_ids = torch.cat([prompts, answer_ids], dim=1)
    input_ids = token_ids[:, :-1]
    target_ids = token_ids[:, 1:].clone()
    # Position i predicts token i + 1: everything before the first answer word belongs to the prompt
    target_ids[:, :MAX_PROMPT_LENGTH - 1] = vocabulary['<pad>']
    return torch.stack(images), input_ids, target_ids


def collate_eval_batch(batch, vocabulary) -> tuple:
    """Collates samples for evaluation.

    :return: (tuple) images, prompt ids, and lists of questions, answers, question types, question objects
        and image objects
    """
    images, questions, answers, question_types, about_objects, image_objects = zip(*batch)
    return (
        torch.stack(images), tokenize_prompts(questions, vocabulary),
        questions, answers, question_types, about_objects, image_objects,
    )
