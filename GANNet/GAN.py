import os, time, itertools, imageio, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class GAN:
    #GAN Generator and Discriminator hidden layer config
    Generator_config = {"layer_1":256,"layer_2":512,"layer_3":1024,"layer_4":784}
    Discriminator_config = {"layer_1":1024,"layer_2":512,"layer_3":256,"layer_4":1}

    # initializers weights and bias function
    w_init_fun = tf.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init_fun = tf.constant_initializer(0.)

    # G(z)
    @staticmethod
    def Generator(x):
        """GAN Generator Net
        Args:
            x(tensor):input noise
        Returns:
            output(array):output one dimensional image data
        """
        config = GAN.Generator_config
        w_init = GAN.w_init_fun
        b_init = GAN.b_init_fun

        # 1st hidden layer
        w0 = tf.get_variable('G_w0', [x.get_shape()[1],config["layer_1"]], initializer=w_init)
        b0 = tf.get_variable('G_b0', [config["layer_1"]], initializer=b_init)
        h0 = tf.nn.relu(tf.matmul(x, w0) + b0)

        # 2nd hidden layer
        w1 = tf.get_variable('G_w1', [h0.get_shape()[1], config["layer_2"]], initializer=w_init)
        b1 = tf.get_variable('G_b1', [config["layer_2"]], initializer=b_init)
        h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)

        # 3rd hidden layer
        w2 = tf.get_variable('G_w2', [h1.get_shape()[1],config["layer_3"]], initializer=w_init)
        b2 = tf.get_variable('G_b2', [config["layer_3"]], initializer=b_init)
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

        # output hidden layer
        w3 = tf.get_variable('G_w3', [h2.get_shape()[1],config["layer_4"]], initializer=w_init)
        b3 = tf.get_variable('G_b3', [config["layer_4"]], initializer=b_init)
        output = tf.nn.tanh(tf.matmul(h2, w3) + b3)

        return output

    # D(x)
    @staticmethod
    def Discriminator(x, drop_out):

        config = GAN.Discriminator_config
        w_init = GAN.w_init_fun
        b_init = GAN.b_init_fun

        # 1st hidden layer
        w0 = tf.get_variable('D_w0', [x.get_shape()[1],config["layer_1"]], initializer=w_init)
        b0 = tf.get_variable('D_b0', [config["layer_1"]], initializer=b_init)
        h0 = tf.nn.relu(tf.matmul(x, w0) + b0)
        h0 = tf.nn.dropout(h0, drop_out)

        # 2nd hidden layer
        w1 = tf.get_variable('D_w1', [h0.get_shape()[1], config["layer_2"]], initializer=w_init)
        b1 = tf.get_variable('D_b1', [config["layer_2"]], initializer=b_init)
        h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)
        h1 = tf.nn.dropout(h1, drop_out)

        # 3rd hidden layer
        w2 = tf.get_variable('D_w2', [h1.get_shape()[1], config["layer_3"]], initializer=w_init)
        b2 = tf.get_variable('D_b2', [config["layer_3"]], initializer=b_init)
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
        h2 = tf.nn.dropout(h2, drop_out)

        # output layer
        w3 = tf.get_variable('D_w3', [h2.get_shape()[1], config["layer_4"]], initializer=w_init)
        b3 = tf.get_variable('D_b3', [config["layer_4"]], initializer=b_init)
        output = tf.sigmoid(tf.matmul(h2, w3) + b3)

        return output


def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    """plot hist image for Generator Net loss and Discriminator Net loss
    Args:
        hist(dict):contains Generator loss and Discriminator loss
        show(bool):if show is true display hist loss,else not display
        save(bool):if save is true save hist loss image,else not save
        path(str):save hist loss image name
    Retrun:
        None
    """
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def get_mnist_data():
    # load MNIST
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # normalization mnist image pixel value range: -1 ~ 1
    train_set = (mnist.train.images - 0.5) / 0.5
    return train_set

def train_gan():
    # training GAN Net parameters
    batch_size = 100
    lr = 0.0002
    train_epoch = 100

    # networks : generator
    with tf.variable_scope('G'):
        z = tf.placeholder(tf.float32, shape=(None, 100))
        G_z = GAN.Generator(z)

    # networks : discriminator
    with tf.variable_scope('D') as scope:
        drop_out = tf.placeholder(dtype=tf.float32, name='drop_out')
        x = tf.placeholder(tf.float32, shape=(None, 784))
        D_real = GAN.Discriminator(x, drop_out)
        scope.reuse_variables()
        D_fake = GAN.Discriminator(G_z, drop_out)


    #a minimum value for log loss bias
    eps = 1e-2
    #Discriminator net loss
    D_loss = tf.reduce_mean(-tf.log(D_real + eps) - tf.log(1 - D_fake + eps))
    #Generator net loss
    G_loss = tf.reduce_mean(-tf.log(D_fake + eps))

    # trainable variables for each network
    t_vars = tf.trainable_variables()
    D_vars = [var for var in t_vars if 'D_' in var.name]
    G_vars = [var for var in t_vars if 'G_' in var.name]

    # optimizer for each network
    D_optim = tf.train.AdamOptimizer(lr).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list=G_vars)

    log_dir_path = "log"
    #make log for save GAN model
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)
    #make dir for save result
    if not os.path.exists('MNIST_GAN_image/random_image'):
        os.makedirs('MNIST_GAN_image/random_image')
    if not os.path.exists('MNIST_GAN_image/fixed_image'):
        os.makedirs('MNIST_GAN_image/fixed_image')

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    init = tf.global_variables_initializer()
    #sv for save GAN mode
    sv = tf.train.Supervisor(logdir=log_dir_path,init_op=init)
    with sv.managed_session() as sess:
        fixed_z_ = np.random.normal(0, 1, (25, 100))
        def show_result(num_epoch,show = False, save = False, path = 'result.png', isFix=False):
            z_ = np.random.normal(0, 1, (25, 100))

            if isFix:
                test_images = sess.run(G_z, {z: fixed_z_, drop_out: 0.0})
            else:
                test_images = sess.run(G_z, {z: z_, drop_out: 0.0})

            size_figure_grid = 5
            fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
            for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)

            for k in range(5*5):
                i = k // 5
                j = k % 5
                ax[i, j].cla()
                ax[i, j].imshow(np.reshape(test_images[k], (28, 28)), cmap='gray')

            label = 'Epoch {0}'.format(num_epoch)
            fig.text(0.5, 0.04, label, ha='center')
            if save:
                plt.savefig(path)
            if show:
                plt.show()
            else:
                plt.close()

        train_set = get_mnist_data()
        # training-loop
        np.random.seed(int(time.time()))
        start_time = time.time()
        for epoch in range(train_epoch):
            G_losses = []
            D_losses = []
            epoch_start_time = time.time()
            for iter in range(train_set.shape[0] // batch_size):
                # update discriminator
                x_ = train_set[iter*batch_size:(iter+1)*batch_size]
                z_ = np.random.normal(0, 1, (batch_size, 100))

                loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, drop_out: 0.3})
                D_losses.append(loss_d_)

                # update generator
                z_ = np.random.normal(0, 1, (batch_size, 100))
                loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, drop_out: 0.3})
                G_losses.append(loss_g_)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
            random_image = 'MNIST_GAN_image/random_image/MNIST_GAN_' + str(epoch + 1) + '.png'
            fixed_image = 'MNIST_GAN_image/fixed_image/MNIST_GAN_' + str(epoch + 1) + '.png'
            show_result((epoch + 1), save=True, path=random_image, isFix=False)
            show_result((epoch + 1), save=True, path=fixed_image, isFix=True)
            train_hist['D_losses'].append(np.mean(D_losses))
            train_hist['G_losses'].append(np.mean(G_losses))
            train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

        end_time = time.time()
        #to calculate train GAN total time
        total_ptime = end_time - start_time
        train_hist['total_ptime'].append(total_ptime)

        print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
        with open('MNIST_GAN_image/train_hist.pkl', 'wb') as f:
            pickle.dump(train_hist, f)
        #save Gan loss change with the number of epochs
        show_train_hist(train_hist, save=True, path='MNIST_GAN_image/MNIST_GAN_train_hist.png')

        #Generate Net generate digital image change with the number of epochs
        images = []
        for e in range(train_epoch):
            img_name = 'MNIST_GAN_image/fixed_image/MNIST_GAN_' + str(e + 1) + '.png'
            images.append(imageio.imread(img_name))
        imageio.mimsave('MNIST_GAN_image/generation_animation.gif', images, fps=5)

         #save GAN model
        sv.saver.save(sess,os.path.join(log_dir_path,"GAN.ckpt"))

def gan_generate_mnist_image(ckpt_path):
    """generate image by load ckpt model
    """
    #generate random vector mean is 0 and std is 1
    gan_input_z = np.random.normal(0,1,(25,100))
    with tf.Session() as sess:
        with tf.variable_scope('G'):
            z = tf.placeholder(dtype=tf.float32,shape=(None,100))
            G_z = GAN.Generator(z)
        #load model
        saver = tf.train.Saver()
        saver.restore(sess,ckpt_path)
        generate_images = sess.run(G_z,feed_dict={z:gan_input_z})
        fig,ax = plt.subplots(5,5,figsize=(5,5))
        for i,j in itertools.product(range(5),range(5)):
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)
        for k in range(5*5):
            i = k // 5
            j = k % 5
            ax[i,j].cla()
            ax[i,j].imshow(np.reshape(generate_images[k],(28,28)),cmap="gray")
        fig.text(0.5,0.04,"Generator Net generate image",ha="center")
        plt.show()





if __name__ == "__main__":
    #train gan Net
    # train_gan()
    #generate image by load model file
    gan_generate_mnist_image("log/GAN.ckpt")
