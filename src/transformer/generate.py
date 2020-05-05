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

# Upload a MIDI file and convert to NoteSequence.
def upload_midi():
    data = list(files.upload().values())
    if len(data) > 1:
        print('Multiple files uploaded; using only one.')
        return mm.midi_to_note_sequence(data[0])

# Decode a list of IDs.
def decode(ids, encoder):
    ids = list(ids)
    if text_encoder.EOS_ID in ids:
        ids = ids[:ids.index(text_encoder.EOS_ID)]
        return encoder.decode(ids)

class PianoPerformanceLanguageModelProblem(score2perf.Score2PerfProblem):
    @property
    def add_eos_symbol(self):
        return True



def generate():
    model_name = 'transformer'
    hparams_set = 'transformer_tpu'
    ckpt_path = '/data/transformer/checkpoints/unconditional_model_16.ckpt'

    problem = PianoPerformanceLanguageModelProblem()
    unconditional_encoders = problem.get_feature_encoders()

# Set up HParams.
    hparams = trainer_lib.create_hparams(hparams_set=hparams_set)
    trainer_lib.add_problem_hparams(hparams, problem)
    hparams.num_hidden_layers = 16
    hparams.sampling_method = 'random'

# Set up decoding HParams.
    decode_hparams = decoding.decode_hparams()
    decode_hparams.alpha = 0.0
    decode_hparams.beam_size = 1

# Create Estimator.
    run_config = trainer_lib.create_run_config(hparams)
    estimator = trainer_lib.create_estimator(
        model_name, hparams, run_config,
        decode_hparams=decode_hparams)
# Create input generator (so we can adjust priming and
# decode length on the fly).
    def input_generator():
        while True:
            yield {
                'targets': np.array([[]], dtype=np.int32),
                'decode_length': np.array(1024, dtype=np.int32)
            }


    inputs = input_generator()
    input_fn = decoding.make_input_fn_from_generator(inputs)
    unconditional_samples = estimator.predict(
    input_fn, checkpoint_path=ckpt_path)
    _ = next(unconditional_samples)
    sample_ids = next(unconditional_samples)['outputs']
    midi_filename = decode(
        sample_ids,
        encoder=unconditional_encoders['targets'])
    unconditional_ns = mm.midi_file_to_note_sequence(midi_filename)

    return unconditional_ns

def generateFromPrimer(primer_ns):
    model_name = 'transformer'
    hparams_set = 'transformer_tpu'
    ckpt_path = '/data/transformer/checkpoints/unconditional_model_16.ckpt'

    problem = PianoPerformanceLanguageModelProblem()
    unconditional_encoders = problem.get_feature_encoders()

# Set up HParams.
    hparams = trainer_lib.create_hparams(hparams_set=hparams_set)
    trainer_lib.add_problem_hparams(hparams, problem)
    hparams.num_hidden_layers = 16
    hparams.sampling_method = 'random'

# Set up decoding HParams.
    decode_hparams = decoding.decode_hparams()
    decode_hparams.alpha = 0.0
    decode_hparams.beam_size = 1

# Create Estimator.
    run_config = trainer_lib.create_run_config(hparams)
    estimator = trainer_lib.create_estimator(
        model_name, hparams, run_config,
        decode_hparams=decode_hparams)
# Create input generator (so we can adjust priming and
# decode length on the fly).
    def input_generator():
        while True:
            targets = unconditional_encoders['targets'].encode_note_sequence(primer_ns)

            # Remove the end token from the encoded primer.
            targets = targets[:-1]

            decode_length = max(0, 4096 - len(targets))
            if len(targets) >= 4096:
                print('Primer has more events than maximum sequence length; nothing will be generated.')
            yield {
                'targets': np.array([[]], dtype=np.int32),
                'decode_length': np.array(1024, dtype=np.int32)
            }


    inputs = input_generator()
    input_fn = decoding.make_input_fn_from_generator(inputs)
    unconditional_samples = estimator.predict(input_fn, checkpoint_path=ckpt_path)
    _ = next(unconditional_samples)















    # Generate sample events.
    sample_ids = next(unconditional_samples)['outputs']

    # Decode to NoteSequence.
    midi_filename = decode(
        sample_ids,
        encoder=unconditional_encoders['targets'])
    ns = mm.midi_file_to_note_sequence(midi_filename)

    # Append continuation to primer.
    continuation_ns = mm.concatenate_sequences([primer_ns, ns])
    return continuation_ns

def playMIDI(ns, sample_rate=16000,
             sf2_path='/data/transformer/Yamaha-C5-Salamander-JNv5.1.sf2'):
    # Play and plot the primer.
    mm.play_sequence(ns,
        synth=mm.fluidsynth, sample_rate=sample_rate, sf2_path=sf2_path)
    mm.plot_sequence(ns)


def loadPrimer(filename):
    primer_ns = mm.midi_file_to_note_sequence(filename)

    # Handle sustain pedal in the primer.
    primer_ns = mm.apply_sustain_control_changes(primer_ns)

    # Trim to desired number of seconds.
    max_primer_seconds = 20  #@param {type:"slider", min:1, max:120}
    if primer_ns.total_time > max_primer_seconds:
      print('Primer is longer than %d seconds, truncating.' % max_primer_seconds)
      primer_ns = mm.extract_subsequence(
          primer_ns, 0, max_primer_seconds)

    # Remove drums from primer if present.
    if any(note.is_drum for note in primer_ns.notes):
      print('Primer contains drums; they will be removed.')
      notes = [note for note in primer_ns.notes if not note.is_drum]
      del primer_ns.notes[:]
      primer_ns.notes.extend(notes)

    # Set primer instrument and program.
    for note in primer_ns.notes:
      note.instrument = 1
      note.program = 0

    return primer_ns

