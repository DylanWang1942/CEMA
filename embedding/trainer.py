import tensorflow as tf
tf.reset_default_graph()
from gate.gate import GATE
from gate.utils import process

class Trainer():

    def __init__(self, args):

        self.args = args
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
        self.build_placeholders()
        gate = GATE(args.hidden_dims, args.lambda_)
        self.loss, self.H, self.C = gate(self.A, self.X, self.R, self.S)
        self.optimize(self.loss)
        self.build_session()

    def build_placeholders(self):
        self.A = tf.sparse_placeholder(dtype=tf.float32)
        self.X = tf.placeholder(dtype=tf.float32)
        self.S = tf.placeholder(tf.int64)
        self.R = tf.placeholder(tf.int64)

    def build_session(self, gpu= True):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        if gpu == False:
            config.intra_op_parallelism_threads = 0
            config.inter_op_parallelism_threads = 0
        self.session = tf.compat.v1.Session(config=config)
        self.session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.global_variables_initializer()])


    def optimize(self, loss):
        optimizer =  tf.compat.v1.train.AdamOptimizer(learning_rate=self.args.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.args.gradient_clipping)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))


    def __call__(self, A, X, S, R):
        for epoch in range(self.args.n_epochs):
            self.run_epoch(epoch, A, X, S, R)


    def run_epoch(self, epoch, A, X, S, R):
        
        loss, _ = self.session.run([self.loss, self.train_op],
                                         feed_dict={self.A: A,
                                                    self.X: X,
                                                    self.S: S,
                                                    self.R: R})

        #print("Epoch: %s, Loss: %.2f" % (epoch, loss))
        return loss

    def infer(self, A, X, S, R):
        H, C = self.session.run([self.H, self.C],
                           feed_dict={self.A: A,
                                      self.X: X,
                                      self.S: S,
                                      self.R: R})


        return H, process.conver_sparse_tf2np(C)




