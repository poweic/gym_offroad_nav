#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from attrdict import AttrDict
from gym_offroad_nav.vehicle_model import VehicleModel
from gym_offroad_nav.vehicle_model_tf import VehicleModelGPU
import itertools
import scipy.optimize

class TrajectoryFitter(object):
    def __init__(self, sess, vehicle_model, margin, n_steps, learning_rate=1e-3):

        self.sess = sess

        # create a GPU counterpart of vehicle model
        self.vehicle_model = VehicleModelGPU(
            vehicle_model.timestep, vehicle_model.wheelbase, vehicle_model.drift
        )

        self.margin = margin
        self.n_steps = n_steps
        self.lr = learning_rate

        self.inputs = self.create_inputs_and_vars()

        self.model = self.build_model(**self.inputs)

    def create_inputs_and_vars(self):

        actions = tf.placeholder(tf.float32, [None, None, 2], name="actions")
        s_target = tf.placeholder(tf.float32, [None, None, 6], name="s_target")

        x_0 = tf.placeholder(tf.float32, [None, 4], name="x_0")
        s_0 = tf.placeholder(tf.float32, [None, 6], name="s_0")

        return AttrDict(actions=actions, s_target=s_target, x_0=x_0, s_0=s_0)

    def build_model(self, actions, s_target, x_0, s_0):

        margin_square = tf.constant(self.margin ** 2, dtype=tf.float32)
        vm = self.vehicle_model

        # 1) initialize (i = 0)
        # transpose to column-vector
        x_0 = tf.transpose(x_0)
        s_0 = tf.transpose(s_0)
        i_0 = 0
        loss_0 = tf.zeros([], dtype=tf.float32)

        seq_length = tf.shape(s_target)[0]
        batch_size = tf.shape(s_target)[1]

        def cond(i, s_i, x_i, loss_i):
            return i < seq_length - 1

        def body(i, s_i, x_i, loss_i):
            st_ip1 = tf.transpose(s_target[i+1])
            a_t = tf.transpose(actions[i])

            s_ip1, x_ip1 = vm.create_predict_op(s_i, x_i, a_t, self.n_steps)

            dxdy = s_ip1[:2] - st_ip1[:2]
            dist = tf.reduce_sum(dxdy ** 2)

            loss_i += tf.maximum(dist - margin_square, 0)

            return i+1, s_ip1, x_ip1, loss_i

        i, s_final, x_final, loss = tf.while_loop(
            cond, body, loop_vars=[i_0, s_0, x_0, loss_0],
        )

        loss /= tf.to_float(seq_length * batch_size)

        grad = tf.gradients(loss, actions)[0]

        return AttrDict(loss=loss, grad=grad, actions=actions,
                        s_final=s_final, x_final=x_final)

    def fit(self, s_target, maxiter=50):

        s_0 = s_target[0]
        self.vehicle_model.reset(s_0.T)
        x_0 = self.vehicle_model.x.T

        action_shape = s_target.shape[:2] + (2,)
        actions = np.zeros(action_shape, dtype=np.float32)

        feed_dict = {
            self.inputs.actions: actions,
            self.inputs.s_target: s_target,
            self.inputs.x_0: x_0,
            self.inputs.s_0: s_0,
        }

        bounds_min = np.zeros(action_shape, dtype=np.float32)
        bounds_max = np.zeros(action_shape, dtype=np.float32)

        bounds_min[..., 0] = 0
        bounds_max[..., 0] = 5
        bounds_min[..., 1] = -0.52
        bounds_max[..., 1] = +0.52

        bounds_min = bounds_min.flatten()
        bounds_max = bounds_max.flatten()

        bounds = zip(bounds_min, bounds_max)

        # Use L-BFGS-B algorithm
        counter = itertools.count()
        def loss_and_grad(actions):
            feed_dict[self.inputs.actions] = actions.reshape(action_shape)
            # loss, grad = self.sess.run([self.model.loss, self.model.grad], feed_dict=feed_dict)
            out = self.sess.run(self.model, feed_dict=feed_dict)
            loss = out.loss
            grad = out.grad.flatten().astype('float64')

            print "s_target[-1] = \n\33[93m{}\33[0m".format(s_target[-1])
            print "s_final = \n\33[93m{}\33[0m".format(out.s_final.T)
            # print "actions = \n\33[93m{}\33[0m".format(actions.reshape(action_shape).squeeze())
            print "#{:02d}: loss = \33[92m{}\33[0m".format(counter.next(), loss)

            return (loss, grad)

        actions, loss, opt_info = scipy.optimize.fmin_l_bfgs_b(loss_and_grad, actions, maxiter=maxiter, bounds=bounds)
        actions = actions.reshape(action_shape)

        """
        # Use SGD
        lr = 1e-3
        actions = np.zeros(action_shape, dtype=np.float32)
        feed_dict[self.inputs.actions] = actions
        for i in range(maxiter):
            loss, grad = self.sess.run([self.model.loss, self.model.grad], feed_dict=feed_dict)
            actions -= lr * grad
            print "#{:02}: loss = {}".format(i, loss)
            # print "actions = {}".format(actions)
        """

        return loss, actions

def gen_s_target(seq_length, batch_size):
    s_target = np.random.randn(seq_length, batch_size, 6).astype(np.float32) * 0.1
    s_target = np.repeat(s_target, [batch_size], axis=1)
    return s_target

def test():
    # Settings
    timestep = 0.01
    command_freq = 5
    n_steps = (1. / command_freq) / timestep

    vehicle_model = VehicleModel(timestep)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        traj_fitter = TrajectoryFitter(sess, vehicle_model, n_steps)

        s_target = gen_s_target(seq_length=10, batch_size=2)

        loss, actions = traj_fitter.fit(s_target)

if __name__ == "__main__":
    test()
