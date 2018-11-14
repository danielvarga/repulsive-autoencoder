import matplotlib
matplotlib.use("Agg")
import numpy as np

path = "pictures/introvae_128x128_m=40_beta=0.1/"

test_log_var = np.load(path + "introvae_128x128_m=40_beta=0.1_test_log_var_epoch200_iter90600.npy")
gen_log_var = np.load(path + "introvae_128x128_m=40_beta=0.1_gen_log_var_epoch200_iter90600.npy")
test_mean = np.load(path + "introvae_128x128_m=40_beta=0.1_test_mean_epoch200_iter90600.npy")
gen_mean = np.load(path + "introvae_128x128_m=40_beta=0.1_gen_mean_epoch200_iter90600.npy")
train_log_var = np.load(path + "introvae_128x128_m=40_beta=0.1_train_log_var_epoch200_iter90600.npy")
rec_log_var = np.load(path + "introvae_128x128_m=40_beta=0.1_rec_log_var_epoch200_iter90600.npy")
train_mean = np.load(path + "introvae_128x128_m=40_beta=0.1_train_mean_epoch200_iter90600.npy")
rec_mean = np.load(path + "introvae_128x128_m=40_beta=0.1_rec_mean_epoch200_iter90600.npy")

train_var = np.exp(train_log_var)
gen_var = np.exp(gen_log_var)
test_var = np.exp(test_log_var)
rec_var = np.exp(rec_log_var)

print(rec_var[0])
print(train_var[0])
print(gen_var[0])

def size_loss(mean):
    return np.sum(np.square(mean), axis=-1)

print(test_mean.shape)
gen_size_loss = size_loss(gen_mean)
train_size_loss = size_loss(train_mean)
rec_size_loss = size_loss(rec_mean)


import matplotlib.pyplot as plt

bins=np.histogram(np.hstack((rec_size_loss,gen_size_loss,train_size_loss)), bins=100)[1] #get the bin edges
plt.hist(gen_size_loss, bins=bins, label='gen size loss',  fc=(1,0,0,0.4))
plt.hist(rec_size_loss, bins=bins, label='rec size loss',  fc=(0,0,1,0.4))
plt.hist(train_size_loss, bins=bins, label='train size loss', fc=(0,1,0,0.4))
plt.legend()
plt.savefig('size_loss_hist.png')
plt.clf()

bins=np.histogram(np.hstack((rec_mean.flatten(),gen_mean.flatten(),train_mean.flatten())), bins=100)[1] #get the bin edges
plt.hist(gen_mean.flatten(), bins=bins, label='gen mean',  fc=(1,0,0,0.4))
plt.hist(rec_mean.flatten(), bins=bins, label='rec mean',  fc=(0,0,1,0.4))
plt.hist(train_mean.flatten(), bins=bins, label='train mean', fc=(0,1,0,0.4))
plt.legend()
plt.yscale("log")
plt.savefig('mean_hist.png')
plt.clf()


bins=np.histogram(np.hstack((rec_var,gen_var,train_var)), bins=100)[1] #get the bin edges
plt.hist(gen_var.flatten(), bins=bins, label='gen var', fc=(1,0,0,0.4))
plt.hist(rec_var.flatten(), bins=bins, label='rec var', fc=(0,0,1,0.4))
plt.hist(train_var.flatten(), bins=bins, label='train var', fc=(0,1,0,0.4))
#plt.hist(test_var.flatten(), bins=bins, label='test var', fc=(1,0,1,1.0))
plt.legend()
plt.savefig('variance_hist.png')
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