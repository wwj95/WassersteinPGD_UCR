
import cv2
import numpy as np
import keras.backend as K
import tensorflow.keras as keras
import tensorflow as tf
#from scipy.stats import wasserstein_distance
class wasserstein_PGD():
    def __init__(self,model):
        super().__init__()
        self.model=model
        #self.device=torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    def generate(self, x, y, y_ori, **param):
        self.parse_params(**param)
        label = y
        y_ori = y_ori
        x_adv = self.attack(x, label, y_ori)
        return x_adv
    def parse_params(self,  
                    eps  = 0.1,
                    iters = 100,
                    norm_clips = 0.1,
                    wass_clips = 0.05,
                    wass = True,
                    norm = 'l1'
                    ):
        self.eps = eps
        self.iters = iters
        self.norm_clips = norm_clips
        self.wass_clips = wass_clips
        self.wass = wass
        self.norm = norm

    def wass_diss(self, seq1, seq2):
        seq1 = seq1.numpy()
        seq2 = seq2.numpy()
        sig1 = np.empty((seq1.size, 2), dtype=np.float32)
        count = 0
        for i in range(seq1.size):
            sig1[count] = np.array([seq1[0][i], i])
            count+=1
        sig2 = np.empty((seq2.size, 2), dtype=np.float32)
        count = 0
        for i in range(seq2.size):
            sig2[count] = np.array([seq2[0][i], i])
            count+=1

        dist, _, flow = cv2.EMD(sig1, sig2, cv2.DIST_L2)
        return dist, flow

    def _cdf_distance(self, p, u_values, v_values):

        u_sorter = tf.argsort(u_values) ### axis = -1 order = acsecnd
        v_sorter = tf.argsort(v_values) ### axis = -1 order = acsecnd

        all_values = tf.concat((u_values, v_values),0)
        all_values = tf.sort(all_values) ### acsecend
        # Compute the differences between pairs of successive values of u and v. see numpy.diff api
        deltas = tf.experimental.numpy.diff(all_values)
        # tf.searchsorted(seq, values)
        u_cdf_indices = tf.searchsorted(np.array(u_values)[u_sorter], all_values[:-1], side = "right")
        v_cdf_indices = tf.searchsorted(np.array(v_values)[v_sorter], all_values[:-1], side = "right")
        u_cdf = u_cdf_indices / u_values.shape
        v_cdf = v_cdf_indices / v_values.shape

        if p == 1:
            return tf.reduce_sum(tf.multiply(tf.abs(u_cdf - v_cdf), deltas))
        if p == 2:
            return tf.sqrt(tf.reduce_sum(tf.multiply(tf.square(u_cdf - v_cdf), deltas)))
        return tf.power(tf.reduce_sum(tf.multiply(tf.power(tf.abs(u_cdf - v_cdf), p),
                                        deltas)), 1/p)

    def wasserstein_distance(self, u_values, v_values):
        return self._cdf_distance(1, u_values, v_values)
    def wass_clipping(self, x_adv, x, init_loss, clip):
        loss = init_loss
        count = 0
        while loss > clip:
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                loss = self.wasserstein_distance(x_adv, x)
            grads = tape.gradient(loss, x_adv)
            x_adv = x_adv-grads*0.5
            loss = self.wasserstein_distance(x_adv, x)
            count+=1
            if count>1000:
                break
        return tf.expand_dims(x_adv, 1)


    def l1_clipping(self, x_adv, x, init_loss, clip):
        loss = init_loss
        count = 0
        while loss > clip:
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                loss = tf.reduce_sum(tf.abs(x_adv - x))
            grads = tape.gradient(loss, x_adv)
            x_adv = x_adv-grads*0.01
            loss = tf.reduce_sum(tf.abs(x_adv - x))
            count+=1
            if count>10000:
                break
        return x_adv


    def single_step_attack(self, x, x_adv, perturbation, y, y_ori):
        x = tf.convert_to_tensor(x)
        x = tf.transpose(x, [1,0])
        x_adv = tf.convert_to_tensor(x_adv)
        x_adv = tf.transpose(x_adv, [1,0])
        #### input shape timestep *1 to 1*timestep
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            pred = self.model(x_adv)
            pred = K.cast(pred, 'float32')
            y = K.cast(y, 'float32')        
            loss = K.categorical_crossentropy(y, pred[0])
        grads = tape.gradient(loss, x_adv)
        perturbation = tf.sign(grads)*self.eps
        x_adv = x_adv+perturbation
        # perturbation = self.wass_diss(x_adv, x)
        if self.norm == 'l1':
            
            if self.wass == 'True':
                x_adv = self.l1_clipping(x_adv, x, tf.reduce_sum(tf.abs(x_adv - x)), self.norm_clips)
                x_adv = tf.convert_to_tensor(x_adv)

                perturbation = self.wasserstein_distance(x_adv[0], x[0])
                x_adv = self.wass_clipping(x_adv[0], x[0], perturbation, self.wass_clips)
            else:
                
                x_adv = self.l1_clipping(x_adv, x, tf.reduce_sum(tf.abs(x_adv - x)), self.norm_clips)
                x_adv = tf.transpose(x_adv, [1,0])

        else:
            if self.wass == 'True':
                x_adv = np.clip(x_adv, x-self.norm_clips, x+self.norm_clips)
                x_adv = tf.convert_to_tensor(x_adv)
                perturbation = self.wasserstein_distance(x_adv[0], x[0])
                x_adv = self.wass_clipping(x_adv[0], x[0], perturbation, self.wass_clips)
            else:
                x_adv = np.clip(x_adv, x-self.norm_clips, x+self.norm_clips).transpose()
        return x_adv


    def attack(self, x, y, y_ori):
        ### init perturbation
        x_adv = x + np.random.normal(0, 1, size = x.shape)
        perturbation=tf.zeros(x.shape)
        for i in range(self.iters):
            x_adv = self.single_step_attack(x, x_adv, perturbation, y, y_ori)
            pred = self.model(tf.transpose(x_adv, [1,0]))
        x_adv_final = x_adv   
        import pdb; pdb.set_trace()
        return x_adv_final

    


    
