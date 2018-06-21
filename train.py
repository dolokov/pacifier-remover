from __future__ import print_function

import time
import os 
import math
import numpy as np 
import cv2 

import tensorflow as tf 

import model
import argument_parse
import database 

CROP_SIZE = 64
RESIZED = 64
B,H,W,N = 2, RESIZED, RESIZED, 3
batch_shape = [B,H,W,N]

def train(a):
    if not os.path.isdir(a.output_dir): os.makedirs(a.output_dir)

    dbA = {'train': database.Database('pacifier','train',batch_shape),'test': database.Database('pacifier','test',batch_shape)}
    dbB = {'train': database.Database('wopacifier','train',batch_shape),'test': database.Database('wopacifier','test',batch_shape)}

    
    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # define inputs placeholder
    inputs_A = tf.placeholder(tf.float32,shape=[None,H,W,N])
    inputs_B = tf.placeholder(tf.float32,shape=[None,H,W,N])

    # inputs and targets are [batch_size, height, width, channels]
    m = model.create_model(inputs_A, inputs_B)

	# summaries
    converted_inputs = {'A':inputs_A,'B':inputs_B}
    for k in ['A','B']:
        with tf.name_scope("inputs_summary%s"%k):
            tf.summary.image("inputs", converted_inputs[k])
    
    
    xAB,xBA,xABA,xBAB = m.xAB,m.xBA,m.xABA,m.xBAB
    x = { 'AB':xAB,'BA':xBA,'ABA':xABA,'BAB':xBAB }        
    for k in x.keys():
        with tf.name_scope("x%s"%k):
            tf.summary.image("x%s"%k, x[k] )

    tf.summary.scalar("discriminator_loss", m.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", m.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", m.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in m.discrim_grads_and_vars + m.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = dbA['train'].steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        # training
        start = time.time()

        for step in xrange(max_steps):
            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

            options = None
            run_metadata = None
            if should(a.trace_freq):
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            fetches = {
                "train": m.train,
                "global_step": sv.global_step,
            }

            if should(a.progress_freq):
                fetches["discrim_loss"] = m.discrim_loss
                fetches["gen_loss_GAN"] = m.gen_loss_GAN
                fetches["gen_loss_L1"] = m.gen_loss_L1

            if should(a.summary_freq):
                fetches["summary"] = sv.summary_op

            if should(a.display_freq):
                fetches["display"] = display_fetches

            feed_dict = {inputs_A:dbA['train'].get_next_batch(B),inputs_B:dbB['train'].get_next_batch(B)}
            results = sess.run(fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata)

            if should(a.summary_freq):
                print("recording summary")
                sv.summary_writer.add_summary(results["summary"], results["global_step"])

            if should(a.display_freq):
                print("saving display images")
                filesets = save_images(results["display"], step=results["global_step"])
                append_index(filesets, step=True)

            if should(a.trace_freq):
                print("recording trace")
                sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

            if should(a.progress_freq):
                # global_step will have the correct step count if we resume from a checkpoint
                train_epoch = math.ceil(results["global_step"] / dbA['train'].steps_per_epoch)
                train_step = (results["global_step"] - 1) % dbA['train'].steps_per_epoch + 1
                rate = (step + 1) * a.batch_size / (time.time() - start)
                remaining = (max_steps - step) * a.batch_size / rate
                print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                print("discrim_loss", results["discrim_loss"])
                print("gen_loss_GAN", results["gen_loss_GAN"])
                print("gen_loss_L1", results["gen_loss_L1"])

            if should(a.save_freq):
                print("saving model")
                saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

            if sv.should_stop():
                break

if __name__ == '__main__':
    a = argument_parse.parse_args()
    if a.mode == 'train':
        train(a)