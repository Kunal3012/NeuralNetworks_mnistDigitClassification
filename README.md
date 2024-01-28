# MNIST Digit Classification (Neural Networks)

## 1. **Introduction:**
   The neural network model is designed for digit classification using the MNIST dataset. It utilizes multiple dense layers to recognize patterns and features in the input images.

## 2. **Dataset:**
| **Dataset Split** | **Number of Samples** | **Image Dimensions** | **Normalization** |
|-------------------|------------------------|-----------------------|---------------------|
| Training Set      | 60,000                 | (28, 28)              | Yes, [0, 1]          |
| Test Set           | 10,000                 | (28, 28)              | Yes, [0, 1]          |

- **Training Set:** A total of 60,000 samples are included in the training set. Each sample is a grayscale image with dimensions (28, 28) pixels. The pixel values are normalized to the range [0, 1].
- **Test Set:** The test set comprises 10,000 samples with the same image dimensions and normalization as the training set.

## 3. **Data Download:**
   The dataset is obtained from the official TensorFlow repository and can be accessed [here](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz).

## 4. **Data Preprocessing:**
   Prior to training, the input images undergo normalization to standardize pixel values within the range of [0, 1].

## 5. **Model Architecture:**
   ### Neural Network Layers
   | **Layer (type)** | **Output Shape** | **Param #** |
   |------------------|-------------------|-------------|
   | Flatten          | (None, 784)       | 0           |
   | Dense            | (None, 128)       | 100,480     |
   | Dense            | (None, 64)        | 8,256       |
   | Dense            | (None, 32)        | 2,080       |
   | Dense            | (None, 10)        | 330         |

   - **Total Parameters:** The model comprises a total of 111,146 parameters.
   - **Trainable Parameters:** All 111,146 parameters are trainable.
   - **Non-trainable Parameters:** There are no non-trainable parameters.

## 6. **Training:**
   - **Loss Function:** The model is trained using Sparse Categorical Crossentropy.
   - **Optimizer:** Adam optimizer is employed during the training process.
   - **Metrics:** The model's performance is assessed based on accuracy.

![model training/epochs](URL)

## 7. **Evaluation:**
   ### Test Metrics
   | **Metric**              | **Value**  |
   |-------------------------|------------|
   | Loss                    | 0.1513     |
   | Mean Squared Error (MSE)| 27.3391    |
   | Accuracy                | 97.33%     |

## 8. **Conclusion:**
   The model demonstrates high accuracy on the MNIST test set, achieving an accuracy of 97.33%. It successfully recognizes and classifies digits based on the provided dataset.

## 9. **Kaggle Notebook:**
   For a detailed implementation and exploration, refer to the 'Kaggle notebook' [here](https://www.kaggle.com/kunal30122002/neuralnetworks-mnistdigitclassification/edit).
