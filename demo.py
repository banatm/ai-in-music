#!/usr/bin/python3
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from note_seq.protobuf import generator_pb2
from note_seq.protobuf import music_pb2
from note_seq import fluidsynth,notebook_utils
from os import path

import numpy as np
from scipy.io import wavfile

def writeWavFile(notes, filename):
  array_of_floats = fluidsynth(notes, sample_rate=44100)
  normalizer = float(np.iinfo(np.int16).max)
  array_of_ints = np.array(
      np.asarray(array_of_floats) * normalizer, dtype=np.int16)  
  wavfile.write(filename, 44100, array_of_ints)

def inputMusic():
  twinkle_twinkle = music_pb2.NoteSequence()
  # Add the notes to the sequence.
  twinkle_twinkle.notes.add(pitch=60, start_time=0.0, end_time=0.5, velocity=80)
  twinkle_twinkle.notes.add(pitch=60, start_time=0.5, end_time=1.0, velocity=80)
  twinkle_twinkle.notes.add(pitch=67, start_time=1.0, end_time=1.5, velocity=80)
  twinkle_twinkle.notes.add(pitch=67, start_time=1.5, end_time=2.0, velocity=80)
  twinkle_twinkle.notes.add(pitch=69, start_time=2.0, end_time=2.5, velocity=80)
  twinkle_twinkle.notes.add(pitch=69, start_time=2.5, end_time=3.0, velocity=80)
  twinkle_twinkle.notes.add(pitch=67, start_time=3.0, end_time=4.0, velocity=80)
  twinkle_twinkle.notes.add(pitch=65, start_time=4.0, end_time=4.5, velocity=80)
  twinkle_twinkle.notes.add(pitch=65, start_time=4.5, end_time=5.0, velocity=80)
  twinkle_twinkle.notes.add(pitch=64, start_time=5.0, end_time=5.5, velocity=80)
  twinkle_twinkle.notes.add(pitch=64, start_time=5.5, end_time=6.0, velocity=80)
  twinkle_twinkle.notes.add(pitch=62, start_time=6.0, end_time=6.5, velocity=80)
  twinkle_twinkle.notes.add(pitch=62, start_time=6.5, end_time=7.0, velocity=80)
  twinkle_twinkle.notes.add(pitch=60, start_time=7.0, end_time=8.0, velocity=80) 
  twinkle_twinkle.total_time = 8
  twinkle_twinkle.tempos.add(qpm=60)
  return twinkle_twinkle


def continueByModel(input, model):
  #download model
  if not(path.isfile(f'./content/{model}.mag')):
    print(f'Downloading {model} bundle. This will take less than a minute...')
    notebook_utils.download_bundle(f'{model}.mag', './content/')
  #init
  bundle = sequence_generator_bundle.read_bundle_file(f'./content/{model}.mag')
  generator_map = melody_rnn_sequence_generator.get_generator_map()
  melody_rnn = generator_map[model](checkpoint=None, bundle=bundle)
  melody_rnn.initialize()

  # do
  input_sequence = input 
  num_steps = 128 # change this for shorter or longer sequences
  temperature = 10.0 # the higher the temperature the more random the sequence.

  # Set the start time to begin on the next step after the last note ends.
  last_end_time = (max(n.end_time for n in input_sequence.notes)
                    if input_sequence.notes else 0)
  qpm = input_sequence.tempos[0].qpm 
  seconds_per_step = 60.0 / qpm / melody_rnn.steps_per_quarter
  total_seconds = num_steps * seconds_per_step

  generator_options = generator_pb2.GeneratorOptions()
  generator_options.args['temperature'].float_value = temperature
  generate_section = generator_options.generate_sections.add(
    start_time=last_end_time + seconds_per_step,
    end_time=total_seconds)

  # Ask the model to continue the sequence.
  return melody_rnn.generate(input_sequence, generator_options)


input = inputMusic()
writeWavFile(input,"./content/input.wav")
for model in ['basic_rnn', 'mono_rnn', 'lookback_rnn', 'attention_rnn']:
  writeWavFile(continueByModel(input,model),f"./content/continued_{model}.wav")
