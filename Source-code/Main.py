{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "52227ea4-c143-426f-bfcf-c2fc87c12623",
   "metadata": {},
   "source": [
    "## **Import Library**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "6ab73394-2a4f-429b-adc7-438861039f65",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras.datasets import cifar10\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,\n",
    "                                     Dropout, Activation, BatchNormalization)\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "19f00f8c-6a13-4b16-9505-8af0ff1a9305",
   "metadata": {},
   "source": [
    "## **Memuat CIFAR-10, normalisasi piksel, dan one-hot encoding**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "c21d8634-80f2-41f3-a5cd-190579185aed",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Memuat dataset CIFAR-10\n",
    "(x_train, y_train), (x_test, y_test) = cifar10.load_data()\n",
    "\n",
    "# Normalisasi data gambar menjadi nilai antara 0 dan 1\n",
    "x_train = x_train.astype('float32') / 255.0\n",
    "x_test = x_test.astype('float32') / 255.0\n",
    "\n",
    "# Mengubah label menjadi format one-hot encoding\n",
    "y_train = to_categorical(y_train, 10)\n",
    "y_test = to_categorical(y_test, 10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e59115e8-47b7-4fe7-8a13-992be171ba0a",
   "metadata": {},
   "source": [
    "## **Membuat model Sequential dengan blok konvolusi pertama**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "cab839d4-e10e-4f84-9e2e-7988662148c1",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()\n",
    "\n",
    "# Blok Konvolusi 1\n",
    "model.add(Conv2D(32, (3, 3), padding='same', \n",
    "                 input_shape=x_train.shape[1:]))\n",
    "model.add(Activation('relu'))\n",
    "model.add(BatchNormalization())\n",
    "model.add(Conv2D(32, (3, 3)))\n",
    "model.add(Activation('relu'))\n",
    "model.add(BatchNormalization())\n",
    "model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "model.add(Dropout(0.25))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cbd9c91c-c6fa-41eb-8006-2374d14ad83c",
   "metadata": {},
   "source": [
    "## **Menambahkan blok konvolusi kedua**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "98bd4f4e-f434-40cf-826d-7dd7551c5f6c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Blok Konvolusi 2\n",
    "model.add(Conv2D(64, (3, 3), padding='same'))\n",
    "model.add(Activation('relu'))\n",
    "model.add(BatchNormalization())\n",
    "model.add(Conv2D(64, (3, 3)))\n",
    "model.add(Activation('relu'))\n",
    "model.add(BatchNormalization())\n",
    "model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "model.add(Dropout(0.25))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "757f7a1c-032e-4953-94e9-8302108f06e9",
   "metadata": {},
   "source": [
    "## **Menambahkan blok konvolusi ketiga**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "58a77973-639f-4815-99c1-10e187d8f84f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Blok Konvolusi 3\n",
    "model.add(Conv2D(128, (3, 3), padding='same'))\n",
    "model.add(Activation('relu'))\n",
    "model.add(BatchNormalization())\n",
    "model.add(Conv2D(128, (3, 3)))\n",
    "model.add(Activation('relu'))\n",
    "model.add(BatchNormalization())\n",
    "model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "model.add(Dropout(0.25))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "41574add-da6e-42a3-9ec8-caf1b3a3aac4",
   "metadata": {},
   "source": [
    "## **Menambahkan lapisan fully connected**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "78c13a93-a924-4270-b067-e29b54ce97cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Lapisan Fully Connected\n",
    "model.add(Flatten())\n",
    "model.add(Dense(1024))\n",
    "model.add(Activation('relu'))\n",
    "model.add(BatchNormalization())\n",
    "model.add(Dropout(0.5))\n",
    "model.add(Dense(10, activation='softmax'))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eb668805-b620-4fed-b88e-cc8a2e8b65b7",
   "metadata": {},
   "source": [
    "## **Mengompilasi model dengan Adam**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "0a8d8b53-e4dc-4fe8-bc9e-aa87ebb31b79",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Menggunakan optimizer Adam\n",
    "optimizer = Adam(learning_rate=0.001)\n",
    "\n",
    "model.compile(loss='categorical_crossentropy',\n",
    "              optimizer=optimizer,\n",
    "              metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bec86f3d-c2af-4288-8d10-57756e5664a5",
   "metadata": {},
   "source": [
    "## **Melatih model dengan early stopping dan model checkpoint**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "794a797b-0e5f-46ab-ba32-eb99559038e7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 96ms/step - accuracy: 0.3425 - loss: 2.1826\n",
      "Epoch 1: val_loss improved from inf to 1.24911, saving model to best_model.keras\n",
      "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m101s\u001b[0m 104ms/step - accuracy: 0.3426 - loss: 2.1821 - val_accuracy: 0.5537 - val_loss: 1.2491\n",
      "Epoch 2/10\n",
      "\u001b[1m781/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m━\u001b[0m \u001b[1m0s\u001b[0m 68ms/step - accuracy: 0.5729 - loss: 1.2144\n",
      "Epoch 2: val_loss improved from 1.24911 to 0.96742, saving model to best_model.keras\n",
      "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m56s\u001b[0m 71ms/step - accuracy: 0.5730 - loss: 1.2143 - val_accuracy: 0.6610 - val_loss: 0.9674\n",
      "Epoch 3/10\n",
      "\u001b[1m781/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m━\u001b[0m \u001b[1m0s\u001b[0m 50ms/step - accuracy: 0.6642 - loss: 0.9661\n",
      "Epoch 3: val_loss did not improve from 0.96742\n",
      "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m41s\u001b[0m 52ms/step - accuracy: 0.6643 - loss: 0.9660 - val_accuracy: 0.6182 - val_loss: 1.1388\n",
      "Epoch 4/10\n",
      "\u001b[1m781/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m━\u001b[0m \u001b[1m0s\u001b[0m 50ms/step - accuracy: 0.7110 - loss: 0.8319\n",
      "Epoch 4: val_loss improved from 0.96742 to 0.72483, saving model to best_model.keras\n",
      "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m41s\u001b[0m 52ms/step - accuracy: 0.7111 - loss: 0.8319 - val_accuracy: 0.7499 - val_loss: 0.7248\n",
      "Epoch 5/10\n",
      "\u001b[1m781/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m━\u001b[0m \u001b[1m0s\u001b[0m 50ms/step - accuracy: 0.7428 - loss: 0.7458\n",
      "Epoch 5: val_loss improved from 0.72483 to 0.68324, saving model to best_model.keras\n",
      "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m41s\u001b[0m 53ms/step - accuracy: 0.7428 - loss: 0.7458 - val_accuracy: 0.7639 - val_loss: 0.6832\n",
      "Epoch 6/10\n",
      "\u001b[1m781/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m━\u001b[0m \u001b[1m0s\u001b[0m 52ms/step - accuracy: 0.7672 - loss: 0.6699\n",
      "Epoch 6: val_loss improved from 0.68324 to 0.63954, saving model to best_model.keras\n",
      "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m43s\u001b[0m 55ms/step - accuracy: 0.7672 - loss: 0.6699 - val_accuracy: 0.7840 - val_loss: 0.6395\n",
      "Epoch 7/10\n",
      "\u001b[1m781/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m━\u001b[0m \u001b[1m0s\u001b[0m 50ms/step - accuracy: 0.7801 - loss: 0.6311\n",
      "Epoch 7: val_loss did not improve from 0.63954\n",
      "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m41s\u001b[0m 52ms/step - accuracy: 0.7801 - loss: 0.6311 - val_accuracy: 0.7576 - val_loss: 0.7451\n",
      "Epoch 8/10\n",
      "\u001b[1m781/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m━\u001b[0m \u001b[1m0s\u001b[0m 50ms/step - accuracy: 0.7970 - loss: 0.5764\n",
      "Epoch 8: val_loss improved from 0.63954 to 0.61002, saving model to best_model.keras\n",
      "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m41s\u001b[0m 53ms/step - accuracy: 0.7970 - loss: 0.5765 - val_accuracy: 0.7937 - val_loss: 0.6100\n",
      "Epoch 9/10\n",
      "\u001b[1m781/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m━\u001b[0m \u001b[1m0s\u001b[0m 50ms/step - accuracy: 0.8089 - loss: 0.5518\n",
      "Epoch 9: val_loss improved from 0.61002 to 0.55361, saving model to best_model.keras\n",
      "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m41s\u001b[0m 53ms/step - accuracy: 0.8088 - loss: 0.5518 - val_accuracy: 0.8135 - val_loss: 0.5536\n",
      "Epoch 10/10\n",
      "\u001b[1m781/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m━\u001b[0m \u001b[1m0s\u001b[0m 51ms/step - accuracy: 0.8208 - loss: 0.5184\n",
      "Epoch 10: val_loss did not improve from 0.55361\n",
      "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m42s\u001b[0m 53ms/step - accuracy: 0.8208 - loss: 0.5184 - val_accuracy: 0.8147 - val_loss: 0.5601\n",
      "Restoring model weights from the end of the best epoch: 9.\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint\n",
    "\n",
    "early_stopping = EarlyStopping(\n",
    "    monitor='val_loss',\n",
    "    patience=10,\n",
    "    verbose=1,\n",
    "    restore_best_weights=True\n",
    ")\n",
    "\n",
    "model_checkpoint = ModelCheckpoint(\n",
    "    'best_model.keras',  \n",
    "    monitor='val_loss',\n",
    "    save_best_only=True,\n",
    "    verbose=1\n",
    ")\n",
    "\n",
    "callbacks = [early_stopping, model_checkpoint]\n",
    "\n",
    "history = model.fit(\n",
    "    x_train, y_train,\n",
    "    batch_size=64,\n",
    "    epochs=10,  \n",
    "    validation_data=(x_test, y_test),\n",
    "    callbacks=callbacks,\n",
    "    verbose=1\n",
    ")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "48fb1dcf-8667-453a-ad2e-8ea46fad7bcf",
   "metadata": {},
   "source": [
    "## **Mengevaluasi model dan mencetak akurasi uji**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "34f87147-4763-4d25-a143-87cf3cee5555",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 - 3s - 8ms/step - accuracy: 0.8135 - loss: 0.5536\n",
      "\n",
      "Test accuracy: 0.8134999871253967\n"
     ]
    }
   ],
   "source": [
    "test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)\n",
    "print(f'\\nTest accuracy: {test_acc}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "9ebcdeb3-b7b5-4a91-b119-caddbfa41280",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The tensorboard extension is already loaded. To reload it, use:\n",
      "  %reload_ext tensorboard\n"
     ]
    }
   ],
   "source": [
    "%load_ext tensorboard"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "c3f54546-7d6e-471b-9c46-765003cad045",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Reusing TensorBoard on port 6006 (pid 16520), started 0:13:36 ago. (Use '!kill 16520' to kill it.)"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "\n",
       "      <iframe id=\"tensorboard-frame-70bcb9dad7a0310b\" width=\"100%\" height=\"800\" frameborder=\"0\">\n",
       "      </iframe>\n",
       "      <script>\n",
       "        (function() {\n",
       "          const frame = document.getElementById(\"tensorboard-frame-70bcb9dad7a0310b\");\n",
       "          const url = new URL(\"/\", window.location);\n",
       "          const port = 6006;\n",
       "          if (port) {\n",
       "            url.port = port;\n",
       "          }\n",
       "          frame.src = url;\n",
       "        })();\n",
       "      </script>\n",
       "    "
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "%tensorboard --logdir logs/fit"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0ee3d46b-e9d1-4fec-877a-7262d4e7d612",
   "metadata": {},
   "source": [
    "## **Memplot akurasi pelatihan dan validasi per epoch**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "bcb65691-cf5f-401b-97c5-c9101b8f5b93",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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rFmzxh5OnZycmDJlCrVr1+bgwYM88cQT/OMf/2DatGkAjBw5kszMTFauXImXlxe7du3C29u7UOdr3rw57dq145NPPuHmm2+2b581axbt27enWbNmLFu2jAcffJApU6YAxkruffv2Zd++fVSuXPma5zhy5Ah33nkn0dHRPP744/z222/8/e9/z7NPeno6bdu25f/+7//w8fHhxx9/ZOjQodSpU4cOHToQHx/Pfffdx5tvvskdd9zBuXPnWLVqlZq9RIpRepaVLXFnWX/gDHF7Yxl8YjJRTjvt7yfbPImxtWdP1e641LuFVrWrMUlD0R2OqQFo8ODBnD59mnHjxhEfH0+zZs1YtGgRYWFhAMTHxxMXF2ff/6GHHuLcuXO89957/P3vf6dKlSp069aNN954w75Px44d+eKLL3jppZd4+eWXqVu3LvPmzSvZOYCyzsP4GiV3/Kv553FwK3wwmTlzJkOGDAGM6QFSU1NZunQpPXr0yLPfiRMnGDx4MHXr1uXzzz8vcvNRvXr1ePPNN/NsGzVqlP15eHg4r732Go8//rg9AMXFxTFo0CCaN28OQJ06dYp0zocffpjnnnuO9957D29vb1JTU/nqq6+YNGkSAN26dcuz/4cffoifnx8rVqygf//+1zz+9OnTqVOnDu+88w4Wi4WGDRuyffv2PL9/NWvW5LnnnrO/fuqpp/jpp5/46quv7AEoOzubO++80/57fvF6ReT6pGdZ2XL4LOsPnGb9gTPEHknCYk3nSZdvedP5B9ycrKTjRmzlW0iq04+gVr25LaSahqI7ONM7QT/xxBM88cQTBb5XUPPMU089xVNPPXXVY951113cddddxVFehbJnzx42btzIN998A4CLiwuDBw9m1qxZ+QJQjx49aNeuHV9++aW9mbEoIiIi8m379ddfGT9+PLt27SIlJYXs7GzS09NJS0vDy8uLp59+mscff5wlS5bQo0cPBg0aRIsWLQp9zvvuu4/Ro0czb948hg8fzrx587DZbNx7772A0RfslVdeYdmyZZw4cQKr1cr58+fzhOyr+eOPP4iMjMxztywqKirPPlarlddff5158+Zx7Ngx+xxTF++etWzZku7du9O8eXN69epFz549ueuuu/Dz8yv0dYo4uguZF+/wnGb9gdNsO5JMpjXH/n5Xp638x+O/1CQRgPNh3al0+0Qi/cPNKlnK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4ZlnnmHJkiV4eHiYXY4AOTk5REREMH78eABat27Nzp07mT59ugKQCebNm8ecOXP43//+R9OmTYmNjWXUqFHUqFGDBx980Ozyyi0FoFIQEBCAs7Nzvrs9iYmJ+e4KSel66qmn+P7771m5ciW1atUyuxyHtHnzZhITE2nbtq19m9VqZeXKlbz33ntkZGTg7OxsYoWOJzg4mCZNmuTZ1rhxY+bPn29SRY7t+eef54UXXuDee+8FoHnz5hw+fJgJEyYoAN0A9QEqBW5ubrRt25aYmJg822NiYujYsaNJVTk2m83Gk08+yTfffMOyZcsIDw83uySH1b17d7Zv305sbKz9ERERwQMPPEBsbKzCjwk6deqUb1qIvXv3EhYWZlJFju38+fM4OeX9unZ2dtYw+BukO0ClZPTo0QwdOpSIiAiioqKYMWMGcXFxREdHm12aQxo5ciT/+9//+O6776hcubL97pyvry+VKlUyuTrHUrly5Xx9r7y8vKhatar6ZJnk2WefpWPHjowfP5577rmHjRs3MmPGDGbMmGF2aQ5pwIAB/Oc//yE0NJSmTZuydetWJk2axMMPP2x2aeWahsGXomnTpvHmm28SHx9Ps2bNeOeddzTk2iRX6nv1ySef8NBDD5VuMZJP165dNQzeZAsXLmTMmDHs27eP8PBwRo8ezaOPPmp2WQ7p3LlzvPzyyyxYsIDExERq1KjBfffdxyuvvIKbm5vZ5ZVbCkAiIiLicNQHSERERByOApCIiIg4HAUgERERcTgKQCIiIuJwFIBERETE4SgAiYiIiMNRABIRERGHowAkIiIiDkcBSESkECwWC99++63ZZYhIMVEAEpEy76GHHsJiseR79O7d2+zSRKSc0mKoIlIu9O7dm08++STPNnd3d5OqEZHyTneARKRccHd3JygoKM/Dz88PMJqnpk+fTp8+fahUqRLh4eF89dVXeT6/fft2unXrRqVKlahatSojRowgNTU1zz6zZs2iadOmuLu7ExwczJNPPpnn/VOnTnHHHXfg6elJ/fr1+f7770v2okWkxCgAiUiF8PLLLzNo0CC2bdvGkCFDuO+++/jjjz8AOH/+PL1798bPz49Nmzbx1Vdf8csvv+QJONOnT2fkyJGMGDGC7du38/3331OvXr085xg7diz33HMPv//+O3379uWBBx7gzJkzpXqdIlJMbCIiZdyDDz5oc3Z2tnl5eeV5jBs3zmaz2WyALTo6Os9nOnToYHv88cdtNpvNNmPGDJufn58tNTXV/v6PP/5oc3JysiUkJNhsNputRo0athdffPGKNQC2l156yf46NTXVZrFYbIsXLy626xSR0qM+QCJSLtxyyy1Mnz49zzZ/f3/786ioqDzvRUVFERsbC8Aff/xBy5Yt8fLysr/fqVMncnJy2LNnDxaLhePHj9O9e/er1tCiRQv7cy8vLypXrkxiYuL1XpKImEgBSETKBS8vr3xNUtdisVgAsNls9ucF7VOpUqVCHc/V1TXfZ3NycopUk4iUDeoDJCIVwvr16/O9btSoEQBNmjQhNjaWtLQ0+/tr1qzBycmJBg0aULlyZWrXrs3SpUtLtWYRMY/uAIlIuZCRkUFCQkKebS4uLgQEBADw1VdfERERwU033cTcuXPZuHEjM2fOBOCBBx7gX//6Fw8++CCvvvoqJ0+e5KmnnmLo0KEEBgYC8OqrrxIdHU316tXp06cP586dY82aNTz11FOle6EiUioUgESkXPjpp58IDg7Os61hw4bs3r0bMEZoffHFFzzxxBMEBQUxd+5cmjRpAoCnpyc///wzzzzzDO3atcPT05NBgwYxadIk+7EefPBB0tPTeeedd3juuecICAjgrrvuKr0LFJFSZbHZbDazixARuREWi4UFCxYwcOBAs0sRkXJCfYBERETE4SgAiYiIiMNRHyARKffUki8iRaU7QCIiIuJwFIBERETE4SgAiYiIiMNRABIRERGHowAkIiIiDkcBSERERByOApCIiIg4HAUgERERcTj/D4k37fzuRDQRAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot Akurasi\n",
    "plt.plot(history.history['accuracy'], label='Akurasi Pelatihan')\n",
    "plt.plot(history.history['val_accuracy'], label='Akurasi Validasi')\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Akurasi')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "702de68d-ad3d-46bd-81a0-c0a1858fc048",
   "metadata": {},
   "source": [
    "## **Menampilkan ringkasan arsitektur model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "4b6b8c6b-4dbe-484a-a92c-0d2257d7167a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1mModel: \"sequential\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
       "┃<span style=\"font-weight: bold\"> Layer (type)                    </span>┃<span style=\"font-weight: bold\"> Output Shape           </span>┃<span style=\"font-weight: bold\">       Param # </span>┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
       "│ conv2d (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                 │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)     │           <span style=\"color: #00af00; text-decoration-color: #00af00\">896</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ activation (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Activation</span>)         │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)     │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ batch_normalization             │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)     │           <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span> │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">BatchNormalization</span>)            │                        │               │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">30</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">30</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)     │         <span style=\"color: #00af00; text-decoration-color: #00af00\">9,248</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ activation_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Activation</span>)       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">30</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">30</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)     │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ batch_normalization_1           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">30</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">30</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)     │           <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span> │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">BatchNormalization</span>)            │                        │               │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ max_pooling2d (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">15</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">15</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)     │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dropout (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">15</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">15</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)     │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">15</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">15</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)     │        <span style=\"color: #00af00; text-decoration-color: #00af00\">18,496</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ activation_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Activation</span>)       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">15</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">15</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)     │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ batch_normalization_2           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">15</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">15</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)     │           <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span> │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">BatchNormalization</span>)            │                        │               │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">13</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">13</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)     │        <span style=\"color: #00af00; text-decoration-color: #00af00\">36,928</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ activation_3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Activation</span>)       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">13</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">13</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)     │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ batch_normalization_3           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">13</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">13</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)     │           <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span> │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">BatchNormalization</span>)            │                        │               │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ max_pooling2d_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">6</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">6</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)       │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dropout_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)             │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">6</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">6</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)       │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_4 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">6</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">6</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)      │        <span style=\"color: #00af00; text-decoration-color: #00af00\">73,856</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ activation_4 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Activation</span>)       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">6</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">6</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)      │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ batch_normalization_4           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">6</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">6</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)      │           <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span> │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">BatchNormalization</span>)            │                        │               │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_5 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)      │       <span style=\"color: #00af00; text-decoration-color: #00af00\">147,584</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ activation_5 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Activation</span>)       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)      │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ batch_normalization_5           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)      │           <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span> │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">BatchNormalization</span>)            │                        │               │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ max_pooling2d_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)      │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dropout_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)             │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)      │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ flatten (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Flatten</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)            │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                   │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1024</span>)           │       <span style=\"color: #00af00; text-decoration-color: #00af00\">525,312</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ activation_6 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Activation</span>)       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1024</span>)           │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ batch_normalization_6           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1024</span>)           │         <span style=\"color: #00af00; text-decoration-color: #00af00\">4,096</span> │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">BatchNormalization</span>)            │                        │               │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dropout_3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)             │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1024</span>)           │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                 │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">10</span>)             │        <span style=\"color: #00af00; text-decoration-color: #00af00\">10,250</span> │\n",
       "└─────────────────────────────────┴────────────────────────┴───────────────┘\n",
       "</pre>\n"
      ],
      "text/plain": [
       "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
       "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                   \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape          \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m      Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
       "│ conv2d (\u001b[38;5;33mConv2D\u001b[0m)                 │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m32\u001b[0m, \u001b[38;5;34m32\u001b[0m, \u001b[38;5;34m32\u001b[0m)     │           \u001b[38;5;34m896\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ activation (\u001b[38;5;33mActivation\u001b[0m)         │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m32\u001b[0m, \u001b[38;5;34m32\u001b[0m, \u001b[38;5;34m32\u001b[0m)     │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ batch_normalization             │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m32\u001b[0m, \u001b[38;5;34m32\u001b[0m, \u001b[38;5;34m32\u001b[0m)     │           \u001b[38;5;34m128\u001b[0m │\n",
       "│ (\u001b[38;5;33mBatchNormalization\u001b[0m)            │                        │               │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_1 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m30\u001b[0m, \u001b[38;5;34m30\u001b[0m, \u001b[38;5;34m32\u001b[0m)     │         \u001b[38;5;34m9,248\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ activation_1 (\u001b[38;5;33mActivation\u001b[0m)       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m30\u001b[0m, \u001b[38;5;34m30\u001b[0m, \u001b[38;5;34m32\u001b[0m)     │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ batch_normalization_1           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m30\u001b[0m, \u001b[38;5;34m30\u001b[0m, \u001b[38;5;34m32\u001b[0m)     │           \u001b[38;5;34m128\u001b[0m │\n",
       "│ (\u001b[38;5;33mBatchNormalization\u001b[0m)            │                        │               │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ max_pooling2d (\u001b[38;5;33mMaxPooling2D\u001b[0m)    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m15\u001b[0m, \u001b[38;5;34m15\u001b[0m, \u001b[38;5;34m32\u001b[0m)     │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dropout (\u001b[38;5;33mDropout\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m15\u001b[0m, \u001b[38;5;34m15\u001b[0m, \u001b[38;5;34m32\u001b[0m)     │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_2 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m15\u001b[0m, \u001b[38;5;34m15\u001b[0m, \u001b[38;5;34m64\u001b[0m)     │        \u001b[38;5;34m18,496\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ activation_2 (\u001b[38;5;33mActivation\u001b[0m)       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m15\u001b[0m, \u001b[38;5;34m15\u001b[0m, \u001b[38;5;34m64\u001b[0m)     │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ batch_normalization_2           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m15\u001b[0m, \u001b[38;5;34m15\u001b[0m, \u001b[38;5;34m64\u001b[0m)     │           \u001b[38;5;34m256\u001b[0m │\n",
       "│ (\u001b[38;5;33mBatchNormalization\u001b[0m)            │                        │               │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_3 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m13\u001b[0m, \u001b[38;5;34m13\u001b[0m, \u001b[38;5;34m64\u001b[0m)     │        \u001b[38;5;34m36,928\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ activation_3 (\u001b[38;5;33mActivation\u001b[0m)       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m13\u001b[0m, \u001b[38;5;34m13\u001b[0m, \u001b[38;5;34m64\u001b[0m)     │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ batch_normalization_3           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m13\u001b[0m, \u001b[38;5;34m13\u001b[0m, \u001b[38;5;34m64\u001b[0m)     │           \u001b[38;5;34m256\u001b[0m │\n",
       "│ (\u001b[38;5;33mBatchNormalization\u001b[0m)            │                        │               │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ max_pooling2d_1 (\u001b[38;5;33mMaxPooling2D\u001b[0m)  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m6\u001b[0m, \u001b[38;5;34m6\u001b[0m, \u001b[38;5;34m64\u001b[0m)       │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dropout_1 (\u001b[38;5;33mDropout\u001b[0m)             │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m6\u001b[0m, \u001b[38;5;34m6\u001b[0m, \u001b[38;5;34m64\u001b[0m)       │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_4 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m6\u001b[0m, \u001b[38;5;34m6\u001b[0m, \u001b[38;5;34m128\u001b[0m)      │        \u001b[38;5;34m73,856\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ activation_4 (\u001b[38;5;33mActivation\u001b[0m)       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m6\u001b[0m, \u001b[38;5;34m6\u001b[0m, \u001b[38;5;34m128\u001b[0m)      │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ batch_normalization_4           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m6\u001b[0m, \u001b[38;5;34m6\u001b[0m, \u001b[38;5;34m128\u001b[0m)      │           \u001b[38;5;34m512\u001b[0m │\n",
       "│ (\u001b[38;5;33mBatchNormalization\u001b[0m)            │                        │               │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ conv2d_5 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4\u001b[0m, \u001b[38;5;34m4\u001b[0m, \u001b[38;5;34m128\u001b[0m)      │       \u001b[38;5;34m147,584\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ activation_5 (\u001b[38;5;33mActivation\u001b[0m)       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4\u001b[0m, \u001b[38;5;34m4\u001b[0m, \u001b[38;5;34m128\u001b[0m)      │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ batch_normalization_5           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4\u001b[0m, \u001b[38;5;34m4\u001b[0m, \u001b[38;5;34m128\u001b[0m)      │           \u001b[38;5;34m512\u001b[0m │\n",
       "│ (\u001b[38;5;33mBatchNormalization\u001b[0m)            │                        │               │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ max_pooling2d_2 (\u001b[38;5;33mMaxPooling2D\u001b[0m)  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m2\u001b[0m, \u001b[38;5;34m2\u001b[0m, \u001b[38;5;34m128\u001b[0m)      │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dropout_2 (\u001b[38;5;33mDropout\u001b[0m)             │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m2\u001b[0m, \u001b[38;5;34m2\u001b[0m, \u001b[38;5;34m128\u001b[0m)      │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ flatten (\u001b[38;5;33mFlatten\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m512\u001b[0m)            │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense (\u001b[38;5;33mDense\u001b[0m)                   │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m1024\u001b[0m)           │       \u001b[38;5;34m525,312\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ activation_6 (\u001b[38;5;33mActivation\u001b[0m)       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m1024\u001b[0m)           │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ batch_normalization_6           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m1024\u001b[0m)           │         \u001b[38;5;34m4,096\u001b[0m │\n",
       "│ (\u001b[38;5;33mBatchNormalization\u001b[0m)            │                        │               │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dropout_3 (\u001b[38;5;33mDropout\u001b[0m)             │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m1024\u001b[0m)           │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense_1 (\u001b[38;5;33mDense\u001b[0m)                 │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m10\u001b[0m)             │        \u001b[38;5;34m10,250\u001b[0m │\n",
       "└─────────────────────────────────┴────────────────────────┴───────────────┘\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">2,479,488</span> (9.46 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m2,479,488\u001b[0m (9.46 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">825,514</span> (3.15 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m825,514\u001b[0m (3.15 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">2,944</span> (11.50 KB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m2,944\u001b[0m (11.50 KB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Optimizer params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">1,651,030</span> (6.30 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Optimizer params: \u001b[0m\u001b[38;5;34m1,651,030\u001b[0m (6.30 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ce3773ad-f3d6-480c-af66-ce826ba8a5a9",
   "metadata": {},
   "source": [
    "## **Menyimpan model dalam file cfr-model.h5**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d9097ce4-ddd0-4ac4-86ec-0a6e3938f762",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save('crf.h5')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
