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
        return args.xent_weight * K.mean(loss)
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
        loss = original_dim * objectives.mean_absolute_error(loss_features[4], loss_features[4])
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
    # pushing latent points towards unit sphere surface, both from inside and out.
    def sphere_loss(x, x_decoded):
        z_mean = loss_features[1]
        average_distances = K.mean(K.square(z_mean), axis=1)
        FUDGE = 0.01
        loss = -1 -K.log(average_distances + FUDGE) + FUDGE + average_distances
        return 10 * args.latent_dim * K.mean(loss)
    def mean_loss(x, x_decoded): # pushing the average of the points to zero
        z_mean = loss_features[1]
        loss = K.abs(K.mean(z_mean, axis = 0))
        return 10 * args.latent_dim * K.mean(loss)
    def repulsive_loss(x, x_decoded): #pushing points away from each other
        z_normed = loss_features[3]
        epsilon = 0.0001
        distances = (2 + epsilon - 2.0 * K.dot(z_normed, K.transpose(z_normed))) ** 0.5
        loss = K.mean(-distances)
        return args.latent_dim * loss
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
    for metric in args.metrics:
        metrics.append(locals().get(metric))
    losses = []
    for loss in args.losses:
        losses.append(locals().get(loss))

    def lossFun(x, x_decoded):
        lossValue = 0
        for loss in losses:
            lossValue += loss(x, x_decoded)
        return lossValue
    return lossFun, metrics

def edgeDetect(images, shape):
    (width, height) = shape[:2]
    verticalEdges   = images[:,:width-1,:height-1,:] - images[:,1:      ,:height-1,:]
    horizontalEdges = images[:,:width-1,:height-1,:] - images[:,:width-1,1:       ,:]    
    diagonalEdges   = images[:,:width-1,:height-1,:] - images[:,1:       ,1:      ,:]
    edges = horizontalEdges + verticalEdges + diagonalEdges
    edges = K.asymmetric_spatial_2d_padding(edges, top_pad=0, left_pad=0, bottom_pad=1, right_pad=1)
    return edges

