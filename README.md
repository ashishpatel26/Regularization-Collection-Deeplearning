## New Regularization Technique for Deep Learning

**Regularization** is a technique used for tuning the function by adding an additional penalty term in the error function. The additional term controls the excessively fluctuating function such that the coefficients don't take extreme values.

---

### **Famous Regularization Technique used By Industry**

**1.L1 & L2 Regularization:**

![Why do we need L2 & L1 regularizations ? - Analytics Vidhya - Medium](https://miro.medium.com/fit/c/140/140/1*Ri4zwZ86FM-2eR6ZQXDOaA.gif)

![L1 vs L2 Regularization: The intuitive difference - Analytics ...](https://miro.medium.com/max/550/1*-LydhQEDyg-4yy5hGEj5wA.png)

* **L1 regularization** adds an **L1** penalty equal to the absolute value of the magnitude of coefficients.
* **L2 regularization** adds an **L2** penalty equal to the square of the magnitude of coefficients. 

**2.DropOut**

![Machine Learning: A Newer Version Of A Lean Thinking Tool?](https://miro.medium.com/max/1872/1*-teDpAIho_nzNShRswkfrQ.gif)

![img](https://miro.medium.com/max/2104/1*5Cg6JhNGJI2FXmptDd_RBQ.png)

* **Dropout** is a **regularization** technique patented by Google for reducing overfitting in neural networks by preventing complex co-adaptations on training data.
* **Eq. 1** shows loss for a regular network and Eq. 2 for a dropout network. In **Eq. 2**, the dropout rate **is 𝛿**, where **𝛿 ~ Bernoulli(*p*)**. This means 𝛿 is equal to **1** with probability *p* and **0** otherwise.
* **Relationship between Dropout** and **Regularization**, A **Dropout rate of** 0.5 will lead **to** the maximum **regularization**, and. Generalization **of Dropout to** Gaussian-Dropout.
* **Code:** [**Tensorflow.Keras**](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) | [**Pytorch**](https://pytorch.org/docs/stable/nn.html?highlight=dropout#torch.nn.Dropout)
* **Reference:** https://papers.nips.cc/paper/4878-understanding-dropout.pdf

**Dropout Variant:** 

1. Alpha Dropout: [**Tensorflow**](https://www.tensorflow.org/api_docs/python/tf/keras/layers/AlphaDropout) | [**Pytorch**](https://pytorch.org/docs/master/generated/torch.nn.AlphaDropout.html) | [**Paper**](https://arxiv.org/abs/1706.02515)
2. Gaussian Dropout: [**Tensorflow**](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GaussianDropout) | **[Pytorch](https://github.com/j-min/Dropouts/blob/master/Gaussian_Variational_Dropout.ipynb)** | **Paper**
3. Spatial Dropout: [**Tensorflow**](https://www.tensorflow.org/api_docs/python/tf/keras/layers/SpatialDropout1D) | [**Pytorch**](https://github.com/davidtvs/PyTorch-ENet/blob/master/models/enet.py) | **[Paper](https://arxiv.org/abs/1411.4280)**
4. State Dropout: [**Tensorflow**](https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/rnn_cell/DropoutWrapper) | [**Pytorch**](https://discuss.pytorch.org/t/dropout-for-lstm-state-transitions/17112) | **Paper**
5. Recurrent Dropout: [**Tensorflow**](https://www.tensorflow.org/addons/api_docs/python/tfa/rnn/LayerNormLSTMCell) | **[Pytorch](https://discuss.pytorch.org/t/how-to-use-lstmcell-with-layernorm/47747)** | [**Paper**](https://arxiv.org/abs/1607.06450)

----

***This Markdown is continuously Updated.. Stay Tuned...!!!***