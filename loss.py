from keras import objectives
import keras.backend as K
import tensorflow as tf
import numpy as np

# loss_features = [z, z_mean, z_log_var, sparse_input, sparse_output]
def loss_factory(model, encoder, loss_features, args):
    original_dim = np.float32(np.prod(args.original_shape))

    def xent_loss(x, x_decoded):
        loss = original_dim * objectives.binary_crossentropy(x, x_decoded)
        return args.xent_weight * K.mean(loss)
    def mse_loss(x, x_decoded):
        loss = original_dim * objectives.mean_squared_error(x, x_decoded)
        return K.mean(loss)
    def hidden_mse_loss(x, x_decoded):
        hidden_x_decoded = encoder(x_decoded)
        x = K.reshape(x, K.shape(x_decoded))
        hidden_x = encoder(x)
        loss= original_dim * objectives.mean_squared_error(hidden_x, hidden_x_decoded)
        return K.mean(loss)        
    def mae_loss(x, x_decoded):
        loss = original_dim * objectives.mean_absolute_error(x, x_decoded)
        return K.mean(loss)
    def arm_loss(x, x_decoded):
        loss = original_dim * objectives.mean_absolute_error(loss_features[3], loss_features[4])
        return K.mean(loss)
    def size_loss(x, x_decoded): # pushing the means towards the origo
        loss = 0.5 * K.sum(K.square(loss_features[1]), axis=-1)
        return K.mean(loss)
    def variance_loss(x, x_decoded): # pushing the variance towards 1
        loss = 0.5 * K.sum(-1 - loss_features[2] + K.exp(loss_features[2]), axis=-1)
        return K.mean(loss)
    def edge_loss(x, x_decoded):
        edge_x = edgeDetect(x, args.original_shape)
        edge_x_decoded = edgeDetect(x_decoded, args.original_shape)
        loss = original_dim * objectives.mean_squared_error(edge_x, edge_x_decoded)
        return K.mean(loss)
    def sphere_distance(x, x_decoded):
        z = loss_features[0]
        squared_distances = K.sum(K.square(z), axis=-1)
        return K.mean(squared_distances)
    # pushing latent points towards unit sphere surface, both from inside and out.
    def sphere_loss(x, x_decoded):
        z = loss_features[0]
        squared_radius = np.float32(args.latent_dim)
        squared_distances = K.sum(K.square(z), axis=-1)
        FUDGE = 0.01
        offset = -squared_radius + squared_radius * K.log(squared_radius)
        loss = -squared_radius * K.log(squared_distances + FUDGE) + FUDGE + squared_distances + offset
#        loss = 0.5 * (-1 - K.log(K.max(0.1,squared_distances) + FUDGE) + FUDGE + squared_distances)
        return K.mean(loss)
    def repulsive_loss(x, x_decoded): #pushing points away from each other
        z = loss_features[1]
        z_squared = K.sum(K.square(z), axis=1)
        z_squared = tf.tile([z_squared],[args.batch_size,1])
        square_sum = z_squared + K.transpose(z_squared)
        epsilon = 0.0001
        distances = (square_sum - 2.0 * K.dot(z, K.transpose(z))) ** 0.5
        loss = K.mean(-distances)
        loss = 1e-6 * loss
        loss = K.clip(K.mean(-distances), 0, 1e-12)
        return loss
    def phantom_loss(x, x_decoded):
        edge_x = edgeDetect(x, args.original_shape)
        loss = original_dim * objectives.mean_squared_error(edge_x, x_decoded)
        return 0.01 * K.mean(loss)
    def covariance_loss(x, x_decoded):
        z = loss_features[1]
        z_centered = z - K.mean(z, axis=0)
        loss = K.sum(K.square(K.eye(K.int_shape(z_centered)[1]) - K.dot(K.transpose(z_centered), z_centered)))
        return loss
    def layerwise_loss(x, x_decoded):
        model_nodes = model.nodes_by_depth
        encOutputs = []
        decInputs = []
        for j in reversed(range(len(model_nodes))):
            node = model_nodes[j][0]
            outLayer = node.outbound_layer
            if outLayer.name.find("dec_conv") != -1:
                decInputs.append(node.input_tensors[0])
            if outLayer.name.find("enc_act") != -1:
                encOutputs.append(node.output_tensors[0])
        loss = 0
        for i in range(len(encOutputs)):
            encoder_output = encOutputs[i]
            decoder_input = decInputs[len(decInputs)-1-i]
            enc_shape = K.int_shape(encoder_output)[1:]
            dec_shape = K.int_shape(decoder_input)[1:]
            assert enc_shape == dec_shape, "encoder ({}) - decoder ({}) shape mismatch at layer {}".format(enc_shape, dec_shape, i)
            current_loss = original_dim * K.mean(K.batch_flatten(K.square(decoder_input - encoder_output)), axis=-1)
            loss += current_loss
        return K.mean(loss)

    metrics = []
    for loss in args.losses:
        metrics.append(locals().get(loss))

    def lossFun(x, x_decoded):
        loss = 0
        for metric in metrics:
            loss += metric(x, x_decoded)
        return loss
    return lossFun, metrics

def edgeDetect(images, shape):
    (width, height) = shape[:2]
    verticalEdges   = images[:,:width-1,:height-1,:] - images[:,1:      ,:height-1,:]
    horizontalEdges = images[:,:width-1,:height-1,:] - images[:,:width-1,1:       ,:]    
    diagonalEdges   = images[:,:width-1,:height-1,:] - images[:,1:       ,1:      ,:]
    edges = horizontalEdges + verticalEdges + diagonalEdges
    edges = K.asymmetric_spatial_2d_padding(edges, top_pad=0, left_pad=0, bottom_pad=1, right_pad=1)
    return edges

