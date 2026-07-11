import os, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)
print("GPUs available:", len(tf.config.list_physical_devices('GPU')))

IMG_SHAPE = (28, 28, 1)
BATCH_SIZE = 256
EPOCHS = 40
LATENT_DIM = 100
LEARNING_RATE = 2e-4
BETA_1 = 0.5
LABEL_SMOOTH = 0.9
N_SAMPLE_GRID = 16
SAMPLES_EVERY = 2
OUTPUT_DIR = "outputs"
os.makedirs(os.path.join(OUTPUT_DIR, "samples"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


def preprocess_to_minus1_1(x):
    x = tf.cast(x, tf.float32)
    return (x - 127.5) / 127.5


(train_images, _), (test_images, _) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images[..., np.newaxis]
ds = tf.data.Dataset.from_tensor_slices(train_images)
ds = ds.map(preprocess_to_minus1_1, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.shuffle(60000).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def make_generator_model(latent_dim, img_shape):
    inputs = tf.keras.Input(shape=(latent_dim,), name="z")
    x = layers.Dense(7 * 7 * 256, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((7, 7, 256))(x)
    x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU()(x)
    outputs = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh")(x)
    return tf.keras.Model(inputs, outputs, name="generator")


def make_discriminator_model(img_shape):
    inputs = tf.keras.Input(shape=img_shape, name="img")
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same")(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x); x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x); x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1)(x)
    return tf.keras.Model(inputs, outputs, name="discriminator")


G = make_generator_model(LATENT_DIM, IMG_SHAPE)
D = make_discriminator_model(IMG_SHAPE)
G.summary()
D.summary()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(fake_logits):
    return cross_entropy(tf.ones_like(fake_logits), fake_logits)


def discriminator_loss(real_logits, fake_logits, label_smooth=LABEL_SMOOTH):
    real_labels = tf.ones_like(real_logits) * label_smooth
    real_loss = cross_entropy(real_labels, real_logits)
    fake_loss = cross_entropy(tf.zeros_like(fake_logits), fake_logits)
    return real_loss + fake_loss


G_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=0.999)
D_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=0.999)

fixed_noise = tf.random.normal([N_SAMPLE_GRID, LATENT_DIM])


def make_grid(imgs, nrows=None, ncols=None):
    N = imgs.shape[0]
    if nrows is None or ncols is None:
        ncols = int(np.ceil(np.sqrt(N)))
        nrows = int(np.ceil(N / ncols))
    H, W = imgs.shape[1], imgs.shape[2]
    C = 1 if imgs.ndim == 3 else imgs.shape[3]
    grid = np.zeros((nrows * H, ncols * W, C), dtype=imgs.dtype)
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, ncols)
        grid[r * H:(r + 1) * H, c * W:(c + 1) * W, ...] = img
    if C == 1:
        grid = grid.squeeze(-1)
    return grid


def sample_and_save(epoch, model, out_dir, n=N_SAMPLE_GRID):
    preds = model(fixed_noise, training=False).numpy()
    preds_disp = np.clip((preds + 1.0) * 127.5, 0, 255).astype(np.uint8)
    grid = make_grid(preds_disp, None, None)
    plt.figure(figsize=(4, 4))
    plt.imshow(grid, cmap="gray")
    plt.axis("off")
    fp = os.path.join(out_dir, f"image_epoch_{epoch:04d}.png")
    plt.savefig(fp, bbox_inches="tight", pad_inches=0)
    plt.close()
    return fp


train_gen_loss = []
train_disc_loss = []


@tf.function
def train_step(images):
    noise = tf.random.normal([tf.shape(images)[0], LATENT_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated = G(noise, training=True)
        real_logits = D(images, training=True)
        fake_logits = D(generated, training=True)
        g_loss = generator_loss(fake_logits)
        d_loss = discriminator_loss(real_logits, fake_logits)
    grads_g = gen_tape.gradient(g_loss, G.trainable_variables)
    grads_d = disc_tape.gradient(d_loss, D.trainable_variables)
    G_opt.apply_gradients(zip(grads_g, G.trainable_variables))
    D_opt.apply_gradients(zip(grads_d, D.trainable_variables))
    return g_loss, d_loss


epoch_log = []
print(f"Starting training: {EPOCHS} epochs, batch size {BATCH_SIZE}")
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    tic = time.time()
    for batch in ds:
        g_loss, d_loss = train_step(batch)
        train_gen_loss.append(float(g_loss))
        train_disc_loss.append(float(d_loss))
    dt = time.time() - tic
    if epoch % SAMPLES_EVERY == 0 or epoch == EPOCHS:
        fp = sample_and_save(epoch, G, os.path.join(OUTPUT_DIR, "samples"))
        print(f"[Epoch {epoch}] sample saved to: {fp}")
    entry = {"epoch": epoch, "g_loss": train_gen_loss[-1], "d_loss": train_disc_loss[-1], "seconds": dt}
    epoch_log.append(entry)
    print(f"Epoch {epoch:03d}/{EPOCHS} in {dt:.1f}s | G_loss={train_gen_loss[-1]:.4f} | D_loss={train_disc_loss[-1]:.4f}")

total_time = time.time() - t0
print(f"Training complete in {total_time/60:.1f} min")

G.save_weights(os.path.join(OUTPUT_DIR, "checkpoints", "G_final.weights.h5"))
D.save_weights(os.path.join(OUTPUT_DIR, "checkpoints", "D_final.weights.h5"))

with open(os.path.join(OUTPUT_DIR, "epoch_log.json"), "w") as f:
    json.dump(epoch_log, f, indent=2)

plt.figure()
plt.plot(train_gen_loss, label="Generator loss")
plt.plot(train_disc_loss, label="Discriminator loss")
plt.xlabel("Training steps")
plt.ylabel("Loss")
plt.legend()
plt.title("DCGAN Training Losses — Fashion-MNIST")
plt.savefig(os.path.join(OUTPUT_DIR, "training_losses.png"), bbox_inches="tight")
plt.close()


def export_samples(n=64, filename="samples_grid_final.png"):
    z = tf.random.normal([n, LATENT_DIM])
    imgs = G(z, training=False).numpy()
    imgs_disp = np.clip((imgs + 1.0) * 127.5, 0, 255).astype(np.uint8)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    grid = make_grid(imgs_disp, rows, cols)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="gray")
    plt.axis("off")
    fp = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(fp, bbox_inches="tight", pad_inches=0)
    plt.close()
    print("Saved:", fp)


export_samples(n=64)

# Real-vs-fake grid for the README
real_batch = train_images[:64].astype(np.uint8)
real_grid = make_grid(real_batch, 8, 8)
plt.figure(figsize=(8, 8))
plt.imshow(real_grid, cmap="gray")
plt.axis("off")
plt.savefig(os.path.join(OUTPUT_DIR, "real_samples_grid.png"), bbox_inches="tight", pad_inches=0)
plt.close()

print("DONE")
