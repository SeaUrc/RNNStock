{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled0.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyNG7aEbY2ERXDD9JgYemi/k",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/SeaUrc/RNNStock/blob/main/RNN.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1K_s2OASjvLn"
      },
      "source": [
        "import numpy as np\r\n",
        "import tensorflow as tf\r\n"
      ],
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bLFqlSlqj8Zb",
        "outputId": "7683a719-3f09-4755-8887-7ab369aa9676"
      },
      "source": [
        "LSTM_CELL_SIZE = 4  # output size (dimension), which is same as hidden size in the cell\r\n",
        "\r\n",
        "state = (tf.zeros([1,LSTM_CELL_SIZE]),)*2\r\n",
        "state"
      ],
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(<tf.Tensor: shape=(1, 4), dtype=float32, numpy=array([[0., 0., 0., 0.]], dtype=float32)>,\n",
              " <tf.Tensor: shape=(1, 4), dtype=float32, numpy=array([[0., 0., 0., 0.]], dtype=float32)>)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 2
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qUnbk6upj_9G",
        "outputId": "37126632-833b-4864-e366-addd39296c93"
      },
      "source": [
        "lstm = tf.keras.layers.LSTM(LSTM_CELL_SIZE, return_sequences=True, return_state=True)\r\n",
        "\r\n",
        "lstm.states=state\r\n",
        "\r\n",
        "print(lstm.states)\r\n"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "(<tf.Tensor: shape=(1, 4), dtype=float32, numpy=array([[0., 0., 0., 0.]], dtype=float32)>, <tf.Tensor: shape=(1, 4), dtype=float32, numpy=array([[0., 0., 0., 0.]], dtype=float32)>)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "foKcIXNQkDIj"
      },
      "source": [
        "#Batch size x time steps x features.\r\n",
        "sample_input = tf.constant([[3,2,2,2,2,2]],dtype=tf.float32)\r\n",
        "\r\n",
        "batch_size = 1\r\n",
        "sentence_max_length = 1\r\n",
        "n_features = 6\r\n",
        "\r\n",
        "new_shape = (batch_size, sentence_max_length, n_features)\r\n",
        "\r\n",
        "inputs = tf.constant(np.reshape(sample_input, new_shape), dtype = tf.float32)"
      ],
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ySeZRRk5kHgU"
      },
      "source": [
        "output, final_memory_state, final_carry_state = lstm(inputs)"
      ],
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yl4Egj-0kKSy",
        "outputId": "c7dd595c-2a9b-4525-beae-b1e12e9f81ce"
      },
      "source": [
        "print('Output : ', tf.shape(output))\r\n",
        "\r\n",
        "print('Memory : ',tf.shape(final_memory_state))\r\n",
        "\r\n",
        "print('Carry state : ',tf.shape(final_carry_state))"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Output :  tf.Tensor([1 1 4], shape=(3,), dtype=int32)\n",
            "Memory :  tf.Tensor([1 4], shape=(2,), dtype=int32)\n",
            "Carry state :  tf.Tensor([1 4], shape=(2,), dtype=int32)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "XP__TnD5kQn6"
      },
      "source": [
        "cells = []"
      ],
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "se4NJZIHkSHi"
      },
      "source": [
        "LSTM_CELL_SIZE_1 = 4 #4 hidden nodes\r\n",
        "cell1 = tf.keras.layers.LSTMCell(LSTM_CELL_SIZE_1)\r\n",
        "cells.append(cell1)"
      ],
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "FTDLg4jykWHj"
      },
      "source": [
        "LSTM_CELL_SIZE_2 = 5 #5 hidden nodes\r\n",
        "cell2 = tf.keras.layers.LSTMCell(LSTM_CELL_SIZE_2)\r\n",
        "cells.append(cell2)"
      ],
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_6kdn1a8kaTa"
      },
      "source": [
        "stacked_lstm =  tf.keras.layers.StackedRNNCells(cells)"
      ],
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vqbzrlPbkbGf"
      },
      "source": [
        "stacked_lstm =  tf.keras.layers.StackedRNNCells(cells)"
      ],
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "h85db8Takc4y"
      },
      "source": [
        "lstm_layer= tf.keras.layers.RNN(stacked_lstm ,return_sequences=True, return_state=True)"
      ],
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "y2MmZszIkexr"
      },
      "source": [
        "#Batch size x time steps x features.\r\n",
        "sample_input = [[[1,2,3,4,3,2], [1,2,1,1,1,2],[1,2,2,2,2,2]],[[1,2,3,4,3,2],[3,2,2,1,1,2],[0,0,0,0,3,2]]]\r\n",
        "sample_input\r\n",
        "\r\n",
        "batch_size = 2\r\n",
        "time_steps = 3\r\n",
        "features = 6\r\n",
        "new_shape = (batch_size, time_steps, features)\r\n",
        "\r\n",
        "x = tf.constant(np.reshape(sample_input, new_shape), dtype = tf.float32)"
      ],
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0gaQwnuekjcX"
      },
      "source": [
        "output, final_memory_state, final_carry_state  = lstm_layer(x)"
      ],
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TnAzv3TtklO5",
        "outputId": "1119b1a6-5eee-4575-f36f-0f954a1f1c60"
      },
      "source": [
        "print('Output : ', tf.shape(output))\r\n",
        "\r\n",
        "print('Memory : ',tf.shape(final_memory_state))\r\n",
        "\r\n",
        "print('Carry state : ',tf.shape(final_carry_state))"
      ],
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Output :  tf.Tensor([2 3 5], shape=(3,), dtype=int32)\n",
            "Memory :  tf.Tensor([2 2 4], shape=(3,), dtype=int32)\n",
            "Carry state :  tf.Tensor([2 2 5], shape=(3,), dtype=int32)\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}