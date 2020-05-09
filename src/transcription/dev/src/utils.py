import numpy as np
import os
import tensorflow as tf

from google.colab import files

from tensor2tensor import models
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib
from tensor2tensor import problems

import magenta.music as mm
from magenta.models.score2perf import score2perf

