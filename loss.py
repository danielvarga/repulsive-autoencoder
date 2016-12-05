from keras import objectives
import keras.backend as K
import numpy as np

def loss_factory(model, encoder, latent_layers, args):
    original_dim = np.prod(args.original_shape)

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
    def size_loss(x, x_decoded): # pushing the means towards the origo
        loss = 0.5 * K.sum(K.square(latent_layers[1]), axis=-1)
        return K.mean(loss)
    def variance_loss(x, x_decoded): # pushing the variance towards 1
        loss = 0.5 * K.sum(-1 - layers[2] + K.exp(layers[2]), axis=-1)
        return K.mean(loss)
    def edge_loss(x, x_decoded):
        edge_x = edgeDetect(x, args.original_shape)
        edge_x_decoded = edgeDetect(x_decoded, args.original_shape)
        loss = original_dim * objectives.mean_squared_error(edge_x, edge_x_decoded)
        return K.mean(loss)
    def covariance_loss(x, x_decoded):
        z = layers[1]
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
    return edges
