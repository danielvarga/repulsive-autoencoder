import numpy as np
from sklearn.preprocessing import normalize
import scipy.misc
from keras import backend as K
from keras.engine.topology import Layer
from keras.regularizers import Regularizer, activity_l1


class ArmLayer(Layer):
    def __init__(self, dict_size, weights = None, iteration = 10, threshold = 0.02, reconsCoef = 1, **kwargs):
        self.np_weights = weights
        self.iteration = iteration
        self.threshold = threshold
        self.reconsCoef = reconsCoef
        self.dict_size = dict_size
        super(ArmLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        nb_features = input_shape[1]

        if self.np_weights is not None:
            print "Using provided weights"
        else:
            self.np_weights = np.random.normal(size=[self.dict_size, nb_features])
            self.np_weights = np.float32(normalize(self.np_weights, axis=1))           
        
        self.W = K.variable(self.np_weights, name='{}_W'.format(self.name))

        Wzero = np.float32(np.zeros(shape=[nb_features, self.dict_size]))
        self.Wzero = K.variable(Wzero, name='{}_Wzero'.format(self.name))
        
        self.trainable_weights = [self.W]

        # set initial alpha
#        eigvals = np.linalg.eigvals(self.np_weights.dot(K.transpose(self.np_weights)))
#        maxEigval = np.max(np.absolute(eigvals))
#        self.alpha = np.float32(1/maxEigval)

        self.activity_regularizer = activity_l1(self.threshold/nb_features)
        self.activity_regularizer.set_layer(self)
        self.regularizers.append(self.activity_regularizer)

        self.recons_regularizer = reconsRegularizer(l2=self.reconsCoef)
        self.recons_regularizer.set_layer(self)
        self.regularizers.append(self.recons_regularizer)
        

    def armderiv(self,x, y, domineigval):
        hard_thresholding = False
        linout = y - (1/domineigval) * K.dot(K.dot(y,self.W) - x,K.transpose(self.W))
        if hard_thresholding:
            out = K.greater(K.abs(linout),self.threshold) * linout
        else:
            out = K.sign(linout) * K.maximum(K.abs(linout) - self.threshold,0)
        return out

    def arm(self, x, domineigval, iteration):
        if iteration==0:
            outApprox = K.dot(x, self.Wzero)
        else:
            outApprox = self.arm(x, domineigval, iteration-1)
        return self.armderiv(x, outApprox, domineigval)

    def call(self, x, mask=None):
        #POWER METHOD FOR APPROXIMATING THE DOMINANT EIGENVECTOR (9 ITERATIONS):
        WW = K.dot(self.W,K.transpose(self.W))
        o = K.ones([self.dict_size,1]) #initial values for the dominant eigenvector
        domineigvec = K.dot(WW,K.dot(WW,K.dot(WW,K.dot(WW,K.dot(WW,K.dot(WW,K.dot(WW,K.dot(WW,K.dot(WW,o)))))))))
        WWd = K.transpose(K.dot(WW,domineigvec))
        domineigval = K.dot(WWd,domineigvec)/K.dot(K.transpose(domineigvec),domineigvec) #THE CORRESPONDING DOMINANT EIGENVALUE
        
        y = self.arm(x, domineigval, self.iteration)        
        return y
    
    def get_output_shape_for(self,input_shape):
        return(input_shape[0], self.dict_size)

class reconsRegularizer(Regularizer):

    def __init__(self, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.uses_learning_phase = True
        self.layer = None

    def set_layer(self,layer):
        if self.layer is not None:
            raise Exception('Regularizers cannot be reused')
        self.layer = layer

    def __call__(self,loss):
        if self.layer is None:
            raise Exception("Need to call 'set_layer' on reconsRegularizer first")
        regularized_loss = loss
        x = self.layer.input
        y = self.layer.output
        recons = K.dot(y,self.layer.W)
        if self.l1:
            regularized_loss += K.mean(self.l1 * K.abs(x-recons))
        if self.l2:
            regularized_loss += K.mean(self.l2 * K.square(x-recons))
        return K.in_train_phase(regularized_loss, loss)
