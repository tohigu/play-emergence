import pygame


import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple


tf.reset_default_graph()
sess = tf.InteractiveSession()

tf.__version__
PAD = 0
EOS = 1

vocab_size = 6
input_embedding_size = 20

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units * 2

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

encoder_cell = LSTMCell(encoder_hidden_units)

((encoder_fw_outputs,
  encoder_bw_outputs),
 (encoder_fw_final_state,
  encoder_bw_final_state)) = (
    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32, time_major=True)
    )

# Echo encoder params
encoder_fw_outputs
encoder_bw_outputs
encoder_fw_final_state
encoder_bw_final_state

encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

encoder_final_state_c = tf.concat(
    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

encoder_final_state_h = tf.concat(
    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

encoder_final_state = LSTMStateTuple(
    c=encoder_final_state_c,
    h=encoder_final_state_h
)

decoder_cell = LSTMCell(decoder_hidden_units)
encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))

decoder_lengths = encoder_inputs_length + 3
# +2 additional steps, +1 leading <EOS> token for decoder inputs

W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

assert EOS == 1 and PAD == 0

eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    initial_input = eos_step_embedded
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    initial_loop_state = None  # we don't need to pass any additional information
    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)

def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

    def get_next_input():
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        prediction = tf.argmax(output_logits, axis=1)
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        return next_input

    elements_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size]
                                                  # defining if corresponding sequence has ended

    finished = tf.reduce_all(elements_finished) # -> boolean scalar
    input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
    state = previous_state
    output = previous_output
    loop_state = None

    return (elements_finished,
            input,
            state,
            output,
            loop_state)


def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:    # time == 0
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()

decoder_outputs

decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))

decoder_prediction = tf.argmax(decoder_logits, 2)

stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)
sess.run(tf.global_variables_initializer())

batch_size = 1000

def batch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used

    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active
            time steps in each input sequence
    """

    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths


def random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     batch_size):
    """ Generates batches of random integer sequences,
        sequence length in [length_from, length_to],
        vocabulary in [vocab_lower, vocab_upper]
    """
    if length_from > length_to:
            raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)

    while True:
        yield [
            np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()
            for _ in range(batch_size)
        ]

batches = random_sequences(length_from=4, length_to=4,
                                   vocab_lower=2, vocab_upper=6,
                                   batch_size=batch_size)

# print('head of the batch:')
# for seq in next(batches)[:10]:
#     print(seq)

def next_feed():
    nextbatch = next(batches)
    encoder_inputs_, encoder_input_lengths_ = batch(nextbatch)
    decoder_targets_, _ = batch(
        [(sequence) + [EOS] + [PAD] * 2 for sequence in nextbatch]
    )
    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
    }

def player_feed(pFeed):
    nextbatch = [pFeed]
    encoder_inputs_, encoder_input_lengths_ = batch(nextbatch)
    decoder_targets_, _ = batch(
        [(sequence) + [EOS] + [PAD] * 2 for sequence in nextbatch]
    )
    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
    }

loss_track = []

max_batches = 50
batches_in_epoch = 100

try:
    for _batch in range(max_batches):
        fd = next_feed()
        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)

        if _batch == 0 or _batch % batches_in_epoch == 0:
            print('batch {}'.format(_batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(decoder_prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(inp))
                print('    predicted > {}'.format(pred))
                if i >= 2:
                    break
            print()

except KeyboardInterrupt:
    print('training interrupted')

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (200,200,200)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Set cell Size
WIDTH = 10
HEIGHT = 10

# Set Margin
MARGIN = 1


# Agent and player initial positions
aPos = [20,7]
pPos = [10,23]

#player moveList
pMov = [0,0,0,0]

# Create Grid
grid = []
for row in range(30):
    # Add an empty array that will hold each cell
    # in this row
    grid.append([])
    for column in range(30):
        grid[row].append(0)  # Append a cell

# grid[1][5] = 1

# Initialize
pygame.init()

# Set window H and W
WINDOW_SIZE = [330, 330]
screen = pygame.display.set_mode(WINDOW_SIZE)

# Set window title
pygame.display.set_caption("Play-emergence")

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

while not done:
    for event in pygame.event.get():  # Got event
        if event.type == pygame.QUIT:  # On close
            done = True  # Let program go
        # Debug Mouse functionality
        # elif event.type == pygame.MOUSEBUTTONDOWN:
        #     pos = pygame.mouse.get_pos()
        #     # Change the x/y screen coordinates to grid coordinates
        #     column = pos[0] // (WIDTH + MARGIN)
        #     row = pos[1] // (HEIGHT + MARGIN)
        #     # Set that location to one
        #     grid[row][column] = 1
        #     print("Click ", pos, "Grid coordinates: ", row, column)
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                print("Pressed w")
                pMov.pop()
                pMov.insert(0,2)
                pPos[0] = pPos[0]-1
                if pPos[0] < 0:
                    pPos[0] = 29
            elif event.key == pygame.K_s:
                print("Pressed s")
                pMov.pop()
                pMov.insert(0,3)
                pPos[0] = pPos[0]+1
                if pPos[0] > 29:
                    pPos[0] = 0
            elif event.key == pygame.K_a:
                print("Pressed a")
                pMov.pop()
                pMov.insert(0,4)
                pPos[1] = pPos[1]-1
                if pPos[1] < 0:
                    pPos[1] = 29
            elif event.key == pygame.K_d:
                print("Pressed d")
                pMov.pop()
                pMov.insert(0,5)
                pPos[1] = pPos[1]+1
                if pPos[1] > 29:
                    pPos[1] = 0
            fd = player_feed(pMov)
            predict_ = sess.run(decoder_prediction,fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(inp))
                print('    predicted > {}'.format(pred))
                if pred[0] == 2:
                    aPos[0] = aPos[0]-1
                    if aPos[0] < 0:
                        aPos[0] = 29
                elif pred[0] == 3:
                    aPos[0] = aPos[0]+1
                    if aPos[0] > 29:
                        aPos[0] = 0
                elif pred[0] == 4:
                    aPos[1] = aPos[1]-1
                    if aPos[1] < 0:
                        aPos[1] = 29
                elif pred[0] == 5:
                    aPos[1] = aPos[1]+1
                    if aPos[1] > 29:
                        aPos[1] = 0
                if i >= 2:
                    break
            print()

    # Set the screen background
    screen.fill(GREY)

    # Draw the grid
    for row in range(30):
        for column in range(30):
            color = WHITE
            if grid[row][column] == 1:
                color = GREEN
            if aPos[0] == row and aPos[1] == column:
                color = RED
            if pPos[0] == row and pPos[1] == column:
                color = BLUE
            pygame.draw.rect(screen,
                             color,
                             [(MARGIN + WIDTH) * column + MARGIN,
                              (MARGIN + HEIGHT) * row + MARGIN,
                              WIDTH,
                              HEIGHT])

    # 60 fps
    clock.tick(60)

    # Update screen.
    pygame.display.flip()

pygame.quit()
