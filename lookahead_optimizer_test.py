# ==============================================================================
# Author: Feiteng Li
# ==============================================================================

"""Tests for LookaheadOptimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

from lookahead_optimizer import LookaheadOptimizer
from radam_optimizer import RAdamOptimizer


def radam_update_numpy(param,
                       g_t,
                       t,
                       m,
                       v,
                       lr=0.001,
                       beta1=0.9,
                       beta2=0.999,
                       epsilon=1e-8,
                       init_v=None,
                       alpha=None,
                       interval_steps=1):
  beta1_power = beta1 ** t
  beta2_power = beta2 ** t

  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = beta2 * v + (1 - beta2) * g_t * g_t

  m_t_hat = m_t / (1.0 - beta1_power)
  rho_inf = 2.0 / (1.0 - beta2) - 1.0
  rho_t = rho_inf - 2.0 * t * beta2_power / (1.0 - beta2_power)

  v_t_hat = np.sqrt(v_t / (1.0 - beta2_power)) + epsilon
  r_t = np.sqrt(((rho_t - 4.0) * (rho_t - 2.0) * rho_inf) / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))

  update = m_t_hat
  if rho_t > 5.0:
    update = r_t * m_t_hat / v_t_hat

  param_t = param - lr * update

  if t % interval_steps == 0:
    param_t = (1.0 - alpha) * init_v + alpha * param_t

  return param_t, m_t, v_t


class LookaheadOptimizerTest(test.TestCase):

  def doTestSparse(self, use_resource=False):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.test_session():
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        init_vars = [np.zeros_like(var0_np), np.zeros_like(var1_np)]

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(var0_np)
          var1 = resource_variable_ops.ResourceVariable(var1_np)
        else:
          var0 = variables.Variable(var0_np)
          var1 = variables.Variable(var1_np)
        grads0_np_indices = np.array([0, 1], dtype=np.int32)
        grads0 = ops.IndexedSlices(
            constant_op.constant(grads0_np),
            constant_op.constant(grads0_np_indices), constant_op.constant([2]))
        grads1_np_indices = np.array([0, 1], dtype=np.int32)
        grads1 = ops.IndexedSlices(
            constant_op.constant(grads1_np),
            constant_op.constant(grads1_np_indices), constant_op.constant([2]))
        opt = LookaheadOptimizer(RAdamOptimizer(), interval_steps=2)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        variables.global_variables_initializer().run()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())

        _, beta1_power, beta2_power = opt._opt._get_beta_accumulators()

        # Run 3 steps of Adam
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9 ** t, beta1_power.eval())
          self.assertAllCloseAccordingToType(0.999 ** t, beta2_power.eval())
          update.run()

          var0_np, m0, v0 = radam_update_numpy(var0_np, grads0_np, t, m0, v0,
                                               init_v=init_vars[0],
                                               alpha=opt._alpha,
                                               interval_steps=opt._interval_steps)
          var1_np, m1, v1 = radam_update_numpy(var1_np, grads1_np, t, m1, v1,
                                               init_v=init_vars[1],
                                               alpha=opt._alpha,
                                               interval_steps=opt._interval_steps)

          if t % opt._interval_steps == 0:
            init_vars = [var0_np, var1_np]

          if not use_resource:
            # Validate updated params
            self.assertAllCloseAccordingToType(var0_np, var0.eval())
            self.assertAllCloseAccordingToType(var1_np, var1.eval())

  def testSparse(self):
    self.doTestSparse(use_resource=False)

  def testResourceSparse(self):
    self.doTestSparse(use_resource=True)

  def testSparseDevicePlacement(self):
    for index_dtype in [dtypes.int32, dtypes.int64]:
      with self.test_session(force_gpu=test.is_gpu_available()):
        # If a GPU is available, tests that all optimizer ops can be placed on
        # it (i.e. they have GPU kernels).
        var = variables.Variable([[1.0], [2.0]])
        indices = constant_op.constant([0, 1], dtype=index_dtype)
        gathered_sum = math_ops.reduce_sum(array_ops.gather(var, indices))
        optimizer = LookaheadOptimizer(RAdamOptimizer(3.0), interval_steps=2)
        minimize_op = optimizer.minimize(gathered_sum)
        variables.global_variables_initializer().run()
        minimize_op.run()

  def testSparseRepeatedIndices(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.test_session():
        repeated_index_update_var = variables.Variable(
            [[1.0], [2.0]], dtype=dtype)
        aggregated_update_var = variables.Variable(
            [[1.0], [2.0]], dtype=dtype)
        grad_repeated_index = ops.IndexedSlices(
            constant_op.constant(
                [0.1, 0.1], shape=[2, 1], dtype=dtype),
            constant_op.constant([1, 1]),
            constant_op.constant([2, 1]))
        grad_aggregated = ops.IndexedSlices(
            constant_op.constant(
                [0.2], shape=[1, 1], dtype=dtype),
            constant_op.constant([1]),
            constant_op.constant([2, 1]))
        repeated_update = LookaheadOptimizer(RAdamOptimizer(), interval_steps=2).apply_gradients(
            [(grad_repeated_index, repeated_index_update_var)])
        aggregated_update = LookaheadOptimizer(RAdamOptimizer(), interval_steps=2).apply_gradients(
            [(grad_aggregated, aggregated_update_var)])
        variables.global_variables_initializer().run()
        self.assertAllClose(aggregated_update_var.eval(),
                            repeated_index_update_var.eval())
        for _ in range(3):
          repeated_update.run()
          aggregated_update.run()
          self.assertAllClose(aggregated_update_var.eval(),
                              repeated_index_update_var.eval())

  def doTestBasic(self, use_resource=False, use_callable_params=False):
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.session(graph=ops.Graph()):
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        init_vars = [np.zeros_like(var0_np), np.zeros_like(var1_np)]

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(
              var0_np, name="var0_%d" % i)
          var1 = resource_variable_ops.ResourceVariable(
              var1_np, name="var1_%d" % i)
        else:
          var0 = variables.Variable(var0_np)
          var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        learning_rate = lambda: 0.001
        beta1 = lambda: 0.9
        beta2 = lambda: 0.999
        epsilon = lambda: 1e-8
        if not use_callable_params:
          learning_rate = learning_rate()
          beta1 = beta1()
          beta2 = beta2()
          epsilon = epsilon()

        opt = LookaheadOptimizer(RAdamOptimizer(learning_rate=learning_rate), interval_steps=3)

        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        opt_variables = opt._opt.variables()
        _, beta1_power, beta2_power = opt._opt._get_beta_accumulators()
        self.assertTrue(beta1_power is not None)
        self.assertTrue(beta2_power is not None)
        self.assertIn(beta1_power, opt_variables)
        self.assertIn(beta2_power, opt_variables)

        if not context.executing_eagerly():
          with ops.Graph().as_default():
            # Shouldn't return non-slot variables from other graphs.
            self.assertEqual(0, len(opt.variables()))
          self.evaluate(variables.global_variables_initializer())
          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        _, beta1_power, beta2_power = opt._opt._get_beta_accumulators()

        # Run 8 steps of Adam
        for t in range(1, 9):
          if not context.executing_eagerly():
            self.evaluate(update)
          elif t > 1:
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

          self.assertAllCloseAccordingToType(0.9 ** (t + 1),
                                             self.evaluate(beta1_power))
          self.assertAllCloseAccordingToType(0.999 ** (t + 1),
                                             self.evaluate(beta2_power))

          var0_np, m0, v0 = radam_update_numpy(var0_np, grads0_np, t, m0, v0,
                                               init_v=init_vars[0],
                                               alpha=opt._alpha,
                                               interval_steps=opt._interval_steps)
          var1_np, m1, v1 = radam_update_numpy(var1_np, grads1_np, t, m1, v1,
                                               init_v=init_vars[1],
                                               alpha=opt._alpha,
                                               interval_steps=opt._interval_steps)

          if t % opt._interval_steps == 0:
            init_vars = [var0_np, var1_np]

          if not use_resource:
            # Validate updated params
            self.assertAllCloseAccordingToType(var0_np, var0.eval())
            self.assertAllCloseAccordingToType(var1_np, var1.eval())

          if use_resource:
            self.assertEqual("var0_%d/RAdam:0" % (i,),
                             opt._opt.get_slot(var=var0, name="m").name)

  def testBasic(self):
    with self.test_session():
      self.doTestBasic(use_resource=False)

  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testResourceBasic(self):
    self.doTestBasic(use_resource=True)

  def testBasicCallableParams(self):
    with context.eager_mode():
      self.doTestBasic(use_resource=True, use_callable_params=True)

  def testTensorLearningRate(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.test_session():
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        opt = LookaheadOptimizer(RAdamOptimizer(constant_op.constant(0.001)), interval_steps=2)

        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())

        _, beta1_power, beta2_power = opt._opt._get_beta_accumulators()

        init_vars = [np.zeros_like(var0_np), np.zeros_like(var1_np)]
        # Run 3 steps of Adam
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9 ** t, beta1_power.eval())
          self.assertAllCloseAccordingToType(0.999 ** t, beta2_power.eval())
          update.run()

          var0_np, m0, v0 = radam_update_numpy(var0_np, grads0_np, t, m0, v0,
                                               init_v=init_vars[0],
                                               alpha=opt._alpha,
                                               interval_steps=opt._interval_steps)
          var1_np, m1, v1 = radam_update_numpy(var1_np, grads1_np, t, m1, v1,
                                               init_v=init_vars[1],
                                               alpha=opt._alpha,
                                               interval_steps=opt._interval_steps)

          if t % opt._interval_steps == 0:
            init_vars = [var0_np, var1_np]

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, var0.eval())
          self.assertAllCloseAccordingToType(var1_np, var1.eval())

  def testSharing(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.test_session():
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        opt = LookaheadOptimizer(RAdamOptimizer(), interval_steps=2)

        update1 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        update2 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        _, beta1_power, beta2_power = opt._opt._get_beta_accumulators()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())

        init_vars = [np.zeros_like(var0_np), np.zeros_like(var1_np)]
        # Run 3 steps of intertwined Adam1 and Adam2.
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9 ** t, beta1_power.eval())
          self.assertAllCloseAccordingToType(0.999 ** t, beta2_power.eval())
          if t % 2 == 0:
            update1.run()
          else:
            update2.run()

          var0_np, m0, v0 = radam_update_numpy(var0_np, grads0_np, t, m0, v0,
                                               init_v=init_vars[0],
                                               alpha=opt._alpha,
                                               interval_steps=opt._interval_steps)
          var1_np, m1, v1 = radam_update_numpy(var1_np, grads1_np, t, m1, v1,
                                               init_v=init_vars[1],
                                               alpha=opt._alpha,
                                               interval_steps=opt._interval_steps)

          if t % opt._interval_steps == 0:
            init_vars = [var0_np, var1_np]

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, var0.eval())
          self.assertAllCloseAccordingToType(var1_np, var1.eval())

  def testTwoSessions(self):
    optimizer = LookaheadOptimizer(RAdamOptimizer())

    with context.eager_mode():
      var0 = variables.Variable(np.array([1.0, 2.0]), name="v0")
      grads0 = constant_op.constant(np.array([0.1, 0.1]))
      optimizer.apply_gradients([(grads0, var0)])

    g = ops.Graph()
    with g.as_default():
      with session.Session():
        var0 = variables.Variable(np.array([1.0, 2.0]), name="v0")
        grads0 = constant_op.constant(np.array([0.1, 0.1]))
        optimizer.apply_gradients([(grads0, var0)])

    gg = ops.Graph()
    with gg.as_default():
      with session.Session():
        var0 = variables.Variable(np.array([1.0, 2.0]), name="v0")
        grads0 = constant_op.constant(np.array([0.1, 0.1]))

        # If the optimizer saves any state not keyed by graph the following line
        # fails.
        optimizer.apply_gradients([(grads0, var0)])


if __name__ == "__main__":
  test.main()

