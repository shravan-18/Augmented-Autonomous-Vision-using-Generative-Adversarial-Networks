import tensorflow as tf
import tensorflow_addons as tfa

convInit = tf.random_normal_initializer(mean=0.0, stddev=0.02)
gammaInit = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

def downsample(input_layer,
               filters,
               name,
               size=3,
               strides=2,
               activation=tf.keras.layers.ReLU(),
               ):
    conv = tf.keras.layers.Conv2D(filters,
                                  size,
                                  strides=strides,
                                  padding='same',
                                  use_bias=False,
                                  kernel_initializer=convInit,
                                  name=f'encoder_{name}')(input_layer)
    conv = tfa.layers.InstanceNormalization(axis=-1,gamma_initializer=gammaInit)(conv)
    conv = activation(conv)
    return conv

def upsample(input_layer,
             filters,
             name,
             size=3,
             strides=2,
             activation='relu'):
    res = tf.keras.layers.Conv2DTranspose(filters, size,
                                          strides=strides,
                                          padding='same',
                                          use_bias=False,
                                          kernel_initializer=convInit,
                                          name=f'decoder_{name}')(input_layer)
    res = tfa.layers.InstanceNormalization(gamma_initializer=gammaInit)(res)
    res =  tf.keras.layers.Activation(activation)(res)
    return res


def residual_block(input_layer,
                   size=3,
                   strides=1,
                   name='block_x'):
    filters = input_layer.shape[-1]
    block = tf.keras.layers.Conv2D(filters,
                     size,
                     strides=strides,
                     padding='same',
                     use_bias=False,
                     kernel_initializer=convInit,
                     name=f'residual_{name}')(input_layer)

    block = tf.keras.layers.Activation('relu')(block)
    block = tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same', use_bias=False,
                     kernel_initializer=convInit, name=f'transformer_{name}_2')(block)
    res = tf.keras.layers.Add()([block, input_layer])

    return res

def concat_layer(layer_1,layer_2,name):
    return tf.keras.layers.Concatenate(name=name)([layer_1,layer_2])


def Generator(num_residual_connections=6):
    ipLayer = tf.keras.layers.Input(shape=(Config.IMG_W,Config.IMG_H,3),
                                   name='input_layer')

    enc1 = downsample(input_layer = ipLayer, filters=64,  strides =  1, size=7, name='dwn_1')
    enc2 = downsample(input_layer=enc1,filters= 128,size =  3, strides =  2, name='dwn_2')
    enc3 = downsample(input_layer=enc2, filters=256,size =  3, strides =2, name='dwn_3')
    enc4 = downsample(input_layer=enc3, filters=256,size =  3, strides =2, name='dwn_4')

    x = enc4
    for n in range(num_residual_connections):
        x = residual_block(input_layer=x, name=f'res_block_{n+1}')

    skipLayer = concat_layer(layer_1=x,layer_2=enc4,name='skip_1')
    dec1 = upsample(skipLayer,filters=256 ,name='upsam_1')

    skipLayer = concat_layer(layer_1=dec1,layer_2=enc3,name='skip_2')
    dec_2 = upsample(skipLayer, filters=128,name='upsam_2')

    skipLayer = concat_layer(layer_1=dec_2,layer_2=enc2,name='skip_3')
    dec_3 = upsample(skipLayer, filters= 64,name='upsam_3')

    skipLayer = concat_layer(layer_1=dec_3,
                          layer_2=enc1,
                          name='skip_final')

    output = tf.keras.layers.Conv2D(filters = 3,kernel_size = 7, strides=1, padding='same',
                                  kernel_initializer=convInit, use_bias=False, activation='tanh',
                                  name='output_layer')(skipLayer)


    return tf.keras.models.Model(inputs=ipLayer,outputs=output)

day2night_gen = Generator()
night2day_gen = Generator()

def PATCH_discriminator(leak_rate = 0.2):
    leaky_relu = tf.keras.layers.LeakyReLU(leak_rate)
    ipLayer = tf.keras.layers.Input(shape=(Config.IMG_W,Config.IMG_H,3),
                               name='input_layer')

    x = downsample(input_layer = ipLayer, filters=64,  strides =  2, size=4, name='dwn_1',activation = leaky_relu)
    x = downsample(input_layer = x, filters=128,  strides =  2, size=4, name='dwn_2',activation = leaky_relu)
    x = downsample(input_layer = x, filters=256,  strides =  2, size=4, name='dwn_3',activation = leaky_relu)
    x = downsample(input_layer = x, filters=512,  strides =  2, size=4, name='dwn_4',activation = leaky_relu)
    x = downsample(input_layer = x, filters=512,  strides =  1, size=4, name='dwn_5',activation = leaky_relu)

    output = tf.keras.layers.Conv2D(1, 4, strides=1, padding='valid', kernel_initializer=convInit)(x)
    return tf.keras.models.Model(inputs=ipLayer,outputs=output)

day2night_disc = PATCH_discriminator()
night2day_disc = PATCH_discriminator()

def generate_cycle(gen_1,gen_2,input_image):
    gen_img_1 = gen_1(input_image,training=True)
    gen_img_2 = gen_2(gen_img_1,training=True)
    return gen_img_1,gen_img_2

def calc_and_apply_gradients(tape,
                             model,
                             loss,
                             optimizer):
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))

def disc_loss(real, generated):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real), real)
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5

def gen_loss(generated):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(generated), generated)

def cycLoss(real_image, cycled_image, LAMBDA):
    mae_loss = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * mae_loss

def identity_loss(real_image, same_image, LAMBDA):
    mae_loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * mae_loss

class CycleGAN(tf.keras.models.Model):
    def __init__(self,
                 cycle=10):
        super(CycleGAN, self).__init__()
        self.genD2N = day2night_gen
        self.genN2D = night2day_gen
        self.discD2N = day2night_disc
        self.discN2D = night2day_disc
        self.cycle = cycle

    def compile(self,
                gen_loss_function,
                disc_loss_function,
                cycle_loss_function,
                loss_function,
                optimizer = tf.keras.optimizers.Adam(learning_rate = Config.LR,beta_1 = 0.5)):

        super(CycleGAN, self).compile()

        self.optimizerGenD2N = optimizer
        self.optimizerGenN2D = optimizer
        self.optimizerDiscD2N = optimizer
        self.optimizerDiscN2D = optimizer

        self.gen_loss_function = gen_loss_function
        self.disc_loss_function = disc_loss_function
        self.cycle_loss_function = cycle_loss_function
        self.loss_function = loss_function

    def train_step(self, batch_data):
        day_image, night_image = batch_data
        with tf.GradientTape(persistent=True) as tape:
            fake_night,cycled_day = generate_cycle(self.genD2N,
                                                     self.genN2D,
                                                     day_image)
            fake_day,cycled_night = generate_cycle(self.genN2D,
                                                   self.genD2N,
                                                   night_image)
            iden_day = self.genD2N(night_image, training=True)
            iden_night = self.genN2D(day_image, training=True)

            disc_night = self.discD2N(night_image, training=True)
            disc_day = self.discN2D(day_image, training=True)

            disc_fake_night   = self.discD2N(fake_night, training=True)
            disc_fake_day = self.discN2D(fake_day, training=True)

            night_gen_loss = self.gen_loss_function(disc_fake_night)
            day_gen_loss = self.gen_loss_function(disc_fake_day)

            total_cycle_loss = self.cycle_loss_function(night_image, cycled_night, self.cycle) + self.cycle_loss_function(day_image, cycled_day, self.cycle)

            total_gen_d2n_loss = night_gen_loss + total_cycle_loss + self.loss_function(night_image, iden_night,self.cycle)
            total_gen_n2d_loss = day_gen_loss + total_cycle_loss + self.loss_function(day_image, iden_day, self.cycle)

            night_disc_loss = self.disc_loss_function(disc_night, disc_fake_night)
            day_disc_loss = self.disc_loss_function(disc_day, disc_fake_day)

        calc_and_apply_gradients(tape=tape,
                                     model= self.genD2N,
                                     loss = total_gen_d2n_loss,
                                     optimizer = self.optimizerGenD2N)
        calc_and_apply_gradients(tape=tape,
                                     model= self.genN2D,
                                     loss = total_gen_n2d_loss,
                                     optimizer = self.optimizerGenN2D)
        calc_and_apply_gradients(tape=tape,
                                     model= self.discD2N,
                                     loss = night_disc_loss,
                                     optimizer = self.optimizerDiscD2N)
        calc_and_apply_gradients(tape=tape,
                                     model= self.discN2D,
                                     loss = day_disc_loss,
                                     optimizer = self.optimizerDiscN2D)

        return {'gen_D2N_loss': total_gen_d2n_loss,
                'gen_N2D_loss': total_gen_n2d_loss,
                'disc_day_loss': day_disc_loss,
                'disc_night_loss': night_disc_loss
               }
    
def get_gan():
    gan = CycleGAN()
    return gan
