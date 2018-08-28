from keras import objectives
import keras.backend as K
import tensorflow as tf
import numpy as np

import eigen

# loss_features is an AttrDict with z_sampled, z_mean, z_log_var, sparse_input, sparse_output.
def loss_factory(model, encoder, loss_features, args):
    original_dim = np.float32(np.prod(args.original_shape))

    def xent_loss(x, x_decoded):
        loss = original_dim * objectives.binary_crossentropy(x, x_decoded)
        return K.mean(loss)

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

    # def arm_loss(x, x_decoded):
    #     loss = original_dim * objectives.mean_absolute_error(loss_features.sparse_input, loss_features.sparse_output)
    #     return K.mean(loss)

    def size_loss(x, x_decoded): # pushing the means towards the origo
        loss = 0.5 * K.sum(K.square(loss_features.z_mean), axis=-1)
        return K.mean(loss)

    def center_loss(x, x_decoded): # pushing the means towards the origo
        loss = 0.5 * K.square(K.sum(loss_features.z_mean, axis=-1))
        return K.mean(loss)

    def size_loss_l1(x, x_decoded): # pushing the means towards the origo
        loss = 0.5 * K.sum(K.abs(loss_features.z_mean), axis=-1)
        return K.mean(loss)

    def sampled_size_loss(x, x_decoded): # pushing the sampleds towards the origo
        loss = 0.5 * K.sum(K.square(loss_features.z_sampled), axis=-1)
        return K.mean(loss)

    def variance_loss(x, x_decoded): # pushing the variance towards 1
        loss = 0.5 * K.sum(-1 - loss_features.z_log_var + K.exp(loss_features.z_log_var), axis=-1)
        return K.mean(loss)

    # augmented variance loss:
    # pushing z_sampled variance towards 1, when x ~ X and z_mean and z_var are calculated by the encoder.
    # We work with the theoretically hard-to-justify, but true-in-practice
    # assumption that z_mean and z_var are independent random variables when x ~ X,
    # and z_mean is normally distributed.
    # This means that z_sampled is normally distributed, and its variance
    # is the sum of var(z_mean) and z_var.
    # This assumption leads to the modified form of the variance_loss,
    # where we add the minibatch-estimated variance of z_mean to z_var.
    #
    # The idea is that this does not punish small z_vars when var(z_mean) is already big.
    # Unfortunately the flipside is that it does not punish large z_vars when var(z_mean) is already small.
    #
    # Surprisingly, on dcgan_vae_lsun_newkl.iniit quickly converged to
    # an mvvm diagram that has 135 0-mv-1-vm coords and 65 1/2-mv-0-vm coords.
    # (mv means the mean's variance, the 1/2 is the surprise here,
    # it's supposed to be 1, might be an implementation bug.)
    def augmented_variance_loss(x, x_decoded):
        variance = K.exp(loss_features.z_log_var)
        # TODO Are you really-really sure it's not axis=1?
        mean_variance = K.var(loss_features.z_mean, axis=0, keepdims=True)
        total_variance = variance + mean_variance
        loss = 0.5 * K.sum(-1 - K.log(total_variance) + total_variance, axis=-1)
        return K.mean(loss)

    # energy distance from the standard normal distribution.
    def energy_distance_loss(x, x_decoded):
        z = loss_features.z_mean
        assert args.batch_size % 2 == 0
        n = args.batch_size // 2
        z_1 = z[:n]
        z_2 = z[n:]
        distances = K.sqrt(K.abs(eigen.mean_pairwise_squared_distances(z_1, z_2, n, n)))
        exx = K.mean(distances)

        m = n * 20
        normals = K.random_normal((2 * m, args.latent_dim), 0, 1)
        distances_from_normals = K.sqrt(K.abs(eigen.mean_pairwise_squared_distances(z_1, normals[:m], n, m)))
        exn = K.mean(distances_from_normals)

        distances_amongst_normals = K.sqrt(K.abs(eigen.mean_pairwise_squared_distances(normals[:n], normals[m:], n, m)))
        enn = K.mean(distances_amongst_normals)

        energy_distance = 2 * exn - exx - enn
        return energy_distance * 100

    def edge_loss(x, x_decoded):
        edge_x = edgeDetect(x, args.original_shape)
        edge_x_decoded = edgeDetect(x_decoded, args.original_shape)
        loss = original_dim * objectives.mean_squared_error(edge_x, edge_x_decoded)
        return K.mean(loss)

    # pushing latent points towards unit sphere surface, both from inside and out.
    def sphere_loss(x, x_decoded):
        # pre sampling!
        average_distances = K.mean(K.square(loss_features.z_mean), axis=1)
        FUDGE = 0.01
        loss = -1 -K.log(average_distances + FUDGE) + FUDGE + average_distances
        return args.latent_dim * K.mean(loss)

    # Forces z_1^2+z_2^2 toward 1, similarly with z_3^2+z_4^2 etc.
    def toroid_loss(x, x_decoded):
        z = loss_features.z_mean
        z_odd = z[:, 1::2]
        z_even = z[:, 0::2]
        loss = K.mean((z_odd ** 2 + z_even ** 2 - 1) ** 2, axis=1)
        return loss

    def mean_loss(x, x_decoded): # pushing the average of the points to zero
        # pre sampling!
        loss = K.abs(K.mean(loss_features.z_mean, axis = 0))
        return args.latent_dim * K.mean(loss)

    def repulsive_loss(x, x_decoded): # pushing points away from each other
        z_normed = loss_features.z_normed
        epsilon = 0.0001
        distances = (2 + epsilon - 2.0 * K.dot(z_normed, K.transpose(z_normed))) ** 0.5
        loss = K.mean(-distances)
        return args.latent_dim * loss

    def phantom_loss(x, x_decoded):
        edge_x = edgeDetect(x, args.original_shape)
        loss = original_dim * objectives.mean_squared_error(edge_x, x_decoded)
        return 0.01 * K.mean(loss)

    def covariance_loss(x, x_decoded):
        print("pre-sampling covariance loss!")
        z = loss_features.z_mean # pre sampling!
#        z = loss_features.z_sampled
        z_centered = z - K.mean(z, axis=0)
        cov = K.dot(K.transpose(z_centered), z_centered) / args.batch_size

        # The (args.batch_size ** 2) is there to keep it in sync
        # with previous version, will get rid of it TODO:

        # SCALE_HACK = 1.53
        # print("Hey hey hey, hacked scaling of covariance loss!")
        # loss = K.mean(K.square(K.eye(K.int_shape(z_centered)[1])*SCALE_HACK - cov)) * (args.batch_size ** 2)
        loss = K.mean(K.square(K.eye(K.int_shape(z_centered)[1]) - cov))
        use_diag_loss = False
        if use_diag_loss:
            print("Adding extra diagonal penalty term to covariance_loss")
            diag = K.maximum(tf.diag_part(cov), 0.01)
            extra_diag_loss = K.mean(diag - K.log(diag) - 1)
            loss += extra_diag_loss
        return loss

    def covariance_loss2(x, x_decoded): # give infinitely large loss for zero eigenvalues (taken from ladder networks)
        z = loss_features.z_mean
        z_centered = z - K.mean(z, axis=0)
        cov = K.dot(K.transpose(z_centered), z_centered) / args.batch_size
        loss = tf.trace(cov - tf.linalg.logm(cov) - K.eye(K.int_shape(z_centered)[1]))
        return loss

    def mean_monitor(x, x_decoded):
        z = loss_features.z_mean # pre sampling!
        return K.mean(K.mean(z, axis=0) ** 2)

    def variance_monitor(x, x_decoded):
        z = loss_features.z_mean # pre sampling!
        z_centered = z - K.mean(z, axis=0)
        cov = K.dot(K.transpose(z_centered), z_centered) / args.batch_size
        return K.mean((tf.diag_part(cov) - 1) ** 2)

    def covariance_monitor(x, x_decoded):
        z = loss_features.z_mean # pre sampling!
        z_centered = z - K.mean(z, axis=0)
        cov = K.dot(K.transpose(z_centered), z_centered) / args.batch_size
        cov = cov - tf.diag(cov)
        return K.mean(cov ** 2)

    def intermediary_loss(x, x_decoded):
        intermediary_outputs = loss_features.intermediary_outputs
        loss = 0
        for intermediary_output in intermediary_outputs:
            loss += mse_loss(x, intermediary_output)
        loss /= len(intermediary_outputs)
        return loss

    # Pushes latent datapoints in the direction of the hyperplane that is
    # orthogonal to the dominant eigenvector of the covariance matrix of the minibatch.
    # Note: the eigenvector calculation assumes that the latent minibatch is zero-centered.
    # We do not do this zero-centering.
    def dominant_eigenvector_loss(x, x_decoded):
        z = loss_features.z_sampled # post sampling
        domineigvec, domineigval = eigen.eigvec_of_cov(z, args.batch_size, latent_dim=args.latent_dim, iterations=3, inner_normalization=False)
        loss = K.square(K.dot(z, domineigvec))
        return K.mean(loss)

    def eigenvalue_gap_loss(x, x_decoded):
        z = loss_features.z_sampled # post sampling
        WW = K.dot(K.transpose(z), z)
        mineigval, maxeigval = eigen.extreme_eigvals(WW, args.batch_size, latent_dim=args.latent_dim, iterations=3, inner_normalization=False)
        loss = K.square(maxeigval-1) + K.square(mineigval-1)
        return loss

    def kstest_loss(x, x_decoded):
        z = loss_features.z_sampled # post sampling
        aggregator = lambda diff: K.mean(K.square(diff)) # That's the default anyway.
        loss = eigen.kstest_loss(z, args.latent_dim, args.batch_size, aggregator)
        return loss

    def layerwise_loss(x, x_decoded):
        model_nodes = model.nodes_by_depth
        encOutputs = []
        decInputs = []
        for j in reversed(list(range(len(model_nodes)))):
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
    def random_mse_loss(x, x_decoded):
        loss = K.square(x - x_decoded)
        loss = original_dim * K.mean(loss, axis=0)
        loss = K.reshape(loss, [np.prod(K.int_shape(loss))])
        indices = np.random.choice(K.int_shape(loss)[0], 1000).astype('int32')
        loss = K.gather(loss, indices)
        return K.mean(loss)
    def threshold_loss(x, x_decoded):
        loss = K.square(x - x_decoded)
        loss = K.maximum(loss - 0.1, 0)
        loss = K.mean(loss, axis=-1)
        return original_dim * K.mean(loss)
    def min_loss(x, x_decoded):
        z = loss_features.z_mean # pre sampling!
        loss = 0.5 * K.sum(K.square(z), axis=-1)
        return K.min(loss)
    def max_loss(x, x_decoded):
        z = loss_features.z_mean # pre sampling!
        return K.max(z)
    def dummy_loss(x, x_decoded):
        z = loss_features.z_mean # pre sampling!
        loss = K.mean(z, axis=1)
        return K.mean(loss)
    def nat_loss(x, x_decoded):
        assert hasattr(loss_features,"nat_input")
        nat = loss_features.nat_input
        z = loss_features.z_mean
        squared_distances = K.sum(K.square(z - nat), axis=1)
        return K.mean(squared_distances)
    def nat_monitor(x, x_decoded):
        assert hasattr(loss_features,"nat_input")
        nat = loss_features.nat_input
        z = loss_features.z_mean
        squared_distances = K.sum(K.square(z - nat), axis=1)
        std_distances = K.std(squared_distances)
        return std_distances
    def quasi_randomness(x, x_decoded):
        A = loss_features.z_mean # pre sampling!

        # zero mean and unit variance for the whole matrix
        mean, variance = tf.nn.moments(A, axes=[0,1])
        A = (A-mean) / tf.sqrt(variance + 1e-16)

        k, n = K.int_shape(A)
        if k is None:
            k = args.batch_size
        AAT = tf.matmul(A, tf.transpose(A))
        qr = ((k*n) ** 2) * tf.reduce_sum(tf.square(AAT)) - (tf.reduce_sum(A) ** 4)
        return qr / ((k*n) ** 4)

    def var_maximizer_loss(x, x_decoded): # pushing the variance of the means up.
        weight = - 0.5 * args.batch_size #  Weighting ensures that this loss cancels size_loss
        loss = weight * K.var(loss_features.z_mean, axis=0)
        return K.sum(loss)


    metrics = []
    for metric in args.metrics:
        metrics.append(locals().get(metric))
    losses = []
    for loss in args.losses:
        losses.append(locals().get(loss))

    weightDict = {}
    for schedule in args.weight_schedules:
        weightDict[schedule[0]] = schedule[5]
    print("weight dict", weightDict)

    def lossFun(x, x_decoded):
        lossValue = 0
        for i in range(len(losses)):
            loss = losses[i]
            lossName = args.losses[i]
            currentLoss = loss(x, x_decoded)
            weight = weightDict.get(lossName, 1.0)
            currentLoss *= weight
            print(lossName, "weight", weight)
            lossValue += currentLoss
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

