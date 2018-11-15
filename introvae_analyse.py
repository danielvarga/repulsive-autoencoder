import matplotlib
matplotlib.use("Agg")
import numpy as np

path = "pictures/introvae_128_m=60_beta=0.1/"

eqbins = np.linspace(0,150,50)
eqbins_var = np.linspace(0,50,50)

def size_loss(mean):
    return np.sum(np.square(mean), axis=-1)

def variance_loss(z_log_var): # pushing the variance towards 1
        loss = 0.5 * np.sum(-1 - z_log_var + np.exp(z_log_var), axis=-1)
        return loss
i = 0
for ep in range(10, 210, 10):

    i +=1
    iter = 29000 // 64 * ep

    test_log_var = np.load(path + "introvae_128_m=60_beta=0.1_test_log_var_epoch"+str(ep)+"_iter"+str(iter)+".npy")
    gen_log_var = np.load(path + "introvae_128_m=60_beta=0.1_gen_log_var_epoch"+str(ep)+"_iter"+str(iter)+".npy")
    test_mean = np.load(path + "introvae_128_m=60_beta=0.1_test_mean_epoch"+str(ep)+"_iter"+str(iter)+".npy")
    gen_mean = np.load(path + "introvae_128_m=60_beta=0.1_gen_mean_epoch"+str(ep)+"_iter"+str(iter)+".npy")
    train_log_var = np.load(path + "introvae_128_m=60_beta=0.1_train_log_var_epoch"+str(ep)+"_iter"+str(iter)+".npy")
    rec_log_var = np.load(path + "introvae_128_m=60_beta=0.1_rec_log_var_epoch"+str(ep)+"_iter"+str(iter)+".npy")
    train_mean = np.load(path + "introvae_128_m=60_beta=0.1_train_mean_epoch"+str(ep)+"_iter"+str(iter)+".npy")
    rec_mean = np.load(path + "introvae_128_m=60_beta=0.1_rec_mean_epoch"+str(ep)+"_iter"+str(iter)+".npy")

    train_var = np.exp(train_log_var)

    gen_var = np.exp(gen_log_var)
    test_var = np.exp(test_log_var)
    rec_var = np.exp(rec_log_var)

    #print(rec_var[0])
    #print(train_var[0])
    #print(gen_var[0])


    #print(test_mean.shape)
    gen_size_loss = size_loss(gen_mean)
    train_size_loss = size_loss(train_mean)
    rec_size_loss = size_loss(rec_mean)


    gen_var_loss = variance_loss(gen_log_var)
    train_var_loss = variance_loss(train_log_var)
    rec_var_loss = variance_loss(rec_log_var)

    import matplotlib.pyplot as plt

    #bins=np.histogram(np.hstack((rec_size_loss,gen_size_loss,train_size_loss)), bins=100)[1] #get the bin edges
    plt.hist(gen_size_loss, bins=eqbins, label='gen size loss',  fc=(1,0,0,0.4))
    plt.hist(rec_size_loss, bins=eqbins, label='rec size loss',  fc=(0.5,0.5,0.5,0.4))
    plt.hist(train_size_loss, bins=eqbins, label='train size loss', fc=(0,1,0,0.4))
    plt.ylim(0,10000)
    plt.title("Size loss histogram at epoch "+ str(ep) +"\n Introvae trained on CelebA 128x128 m=60, alpha=0.25, beta=0.1")
    plt.legend()
    plt.savefig(str(i).zfill(3)+'size_loss_hist_'+str(iter)+'.png')
    plt.clf()

    bins=np.histogram(np.hstack((rec_var_loss,gen_var_loss,train_var_loss)), bins=100)[1] #get the bin edges
    plt.hist(gen_var_loss, bins=eqbins_var, label='gen var loss',  fc=(1,0,0,0.4))
    plt.hist(rec_var_loss, bins=eqbins_var, label='rec var loss',  fc=(0.5,0.5,0.5,0.4))
    plt.hist(train_var_loss, bins=eqbins_var, label='train var loss', fc=(0,1,0,0.4))
    plt.ylim(0,10000)
    plt.title("Var loss histogram at epoch "+ str(ep) +"\n Introvae trained on CelebA 128x128 m=60, alpha=0.25, beta=0.1")
    plt.legend()
    plt.savefig(str(i).zfill(3)+'var_hist_'+str(iter)+'.png')
    plt.clf()
    continue

    bins=np.histogram(np.hstack((rec_mean.flatten(),gen_mean.flatten(),train_mean.flatten())), bins=100)[1] #get the bin edges
    plt.hist(gen_mean.flatten(), bins=bins, label='gen mean',  fc=(1,0,0,0.4))
    plt.hist(rec_mean.flatten(), bins=bins, label='rec mean',  fc=(0.5,0.5,0.5,0.4))
    plt.hist(train_mean.flatten(), bins=bins, label='train mean', fc=(0,1,0,0.4))
    plt.legend()
    plt.yscale("log")
    plt.savefig(str(i).zfill(3)+'mean_hist_'+str(iter)+'.png')
    plt.clf()


    bins=np.histogram(np.hstack((rec_var,gen_var,train_var)), bins=100)[1] #get the bin edges
    plt.hist(gen_var.flatten(), bins=bins, label='gen var', fc=(1,0,0,0.4))
    plt.hist(rec_var.flatten(), bins=bins, label='rec var', fc=(0.5,0.5,0.5,0.4))
    plt.hist(train_var.flatten(), bins=bins, label='train var', fc=(0,1,0,0.4))
    #plt.hist(test_var.flatten(), bins=bins, label='test var', fc=(1,0,1,1.0))
    plt.legend()
    plt.savefig(str(i).zfill(3)+'variance_hist_'+str(iter)+'.png')
    plt.clf()


print(gen_size_loss.shape)
print(gen_size_loss)
print(train_size_loss)
print(rec_size_loss)
sads
print(size_loss(train_mean))
print(np.mean(gen_mean, axis=0))
print(np.mean(train_mean, axis=0))



import matplotlib.pyplot as plt

plt.hist(gen_mean[1], bins=50)
plt.savefig('gen_mean_1.png')

plt.clf()

plt.hist(train_mean[1], bins=50)
plt.savefig('train_mean_1.png')

plt.clf()
plt.hist(gen_var[1], bins=50)
plt.savefig('gen_var_1.png')


plt.clf()
plt.hist(train_var[1], bins=50)
plt.savefig('train_var_1.png')


quit()
plt.plot()
h, b = np.histogram(gen_mean[0])
print(h)
print(b)
h, b = np.histogram(train_mean[0])
print(h)
print(b)


print(np.sum(np.mean(np.square(rec_mean), axis=1)))
print(np.sum(np.mean(np.square(train_mean), axis=1)))
print(np.sum(np.mean(np.square(test_mean), axis=1)))



print(np.sum(np.mean(np.square(gen_mean), axis=1)))
print(np.sum(np.mean(np.square(rec_mean), axis=1)))
print(np.sum(np.mean(np.square(train_mean), axis=1)))
print(np.sum(np.mean(np.square(test_mean), axis=1)))

print(np.sum(np.mean(np.square(gen_var), axis=1)))
print(np.sum(np.mean(np.square(rec_var), axis=1)))
print(np.sum(np.mean(np.square(train_var), axis=1)))
print(np.sum(np.mean(np.square(test_var), axis=1)))