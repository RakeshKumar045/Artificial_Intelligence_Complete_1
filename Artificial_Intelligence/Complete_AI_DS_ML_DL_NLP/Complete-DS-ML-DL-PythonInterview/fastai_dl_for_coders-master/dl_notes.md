# Notes

- It is fine to have a random noise in the data; however, it is bad to have biased noise.
- When fine tuning a model, divide lr we used by 5 or 10 and use it as max_lr and then use lr_finder to get min_lr. The min_lr would be the one with steepest slope from lr_finder.
- When running a web application based on trained model:
  - Most likely we don't need a GPU. CPU will do the job.
  - Make sure to get the same order of classes that was used during training so that will be used during inference.
  - Make sure we pass the same transforms, image_size, normalization that was used during training.
- In PyTorch, Dataloader is only used for data parallelism which means it uses a batch of data from Dataset utilizing multithreading. Therefore, we only can access the data using Dataset object if we are indexing it; otherwise, Dataloader gives us one batch at a time depends on the batch_size.
- Metrics are mainly used for models' comparisons such as accuracy and DO NOT affect optimization and performance of the learning algorithm.
- `partial` function from functools module act as the same function provided but using a set of arguments and keywords arguments. Example:
  - sorted_reverse = partial(sorted, reverde=True)
  - sorted_reverse([1, 2, 3])
- We can train CNN on small image sizes for faster experimentation. Then, we can fine tune the model using bigger images to boost the performance.
- Image Segmentation: Classify each pixel in each image to its corresponding category/color. Examples: trees would have same color, roads would have same color, etc.
- Fitting models using one-cycle learning rate policy helps us train more quickly and get to a better loss that would make it a lot more generalizable. The reason for that is because the loss function surface is bumpy and then gets flatter and then gets bumpy. With learning rate that is increasing at the beginning, we explore more and avoid getting stuck in those bumpy regions. When it gets smaller again after it reaches its maximum, it kinda finds its way to the flatter part that tends to work well and generalizes better.
- To check if the maximum learning rate we picked is good, look if validation loss gets a little bit worse at the beginning and then gets a lot better after that. Otherwise, try to change the maximum learning rate.
- We should always use the same stats the pretrained model used while training when use new datasets to normalize images.
- Even if the dataset has nothing to do with ImageNet objects, we can still use pretrained models on imagenet and get a very good results.
- With bigger network such as resnet50, we may want to choose bigger image size.
- Process of fitting CNNs that work most of the times:
  - First, fit the model using:
    - Somewhat small image size.
    - Pick the learning rate using learning rate finder. Pick the one with the steepest slope right before minimum (mostly 1e-1 to the left of the minimum)
    - Fit the model use lr determined above with all layers freezed except the classifier (last layer)
  - Second, unfreeze layers and use lr finder to pick new lr. Usually the lr_interval would be: minumum = 1e-1 before it shoots. maximum = lr used in previous step divided by either 5 or 10.
  - Third, go bigger on the image size such as double.
    - First freeze the layers.
    - Second pick lr using lr_finder.
    - Next fit again using lr from above.
    - Then unfreeze and use lr_finder again.
    - Refit using lr from above.
- If images are two channels, we can add one channel to make it 3 channels so that we can use image-netpretrained models. The third channel would be either all zeros or the average of the first two channels.
- A language model is a model that predicts the next word in a sentence give previois words.
- Using pre-trained language model on something like wiki-text and then fine-tune it one the dataset we have will boost the performance of any NLP task because the language model will adapt specifically to the new dataset and won't be as generic as before.
- Language model will be initially trained on what is called self-supervised learning because the labels will be within the data itself, i.e predicting the next word given the all previous words.
- We usually are not interested in the langiage model itself because it will be used for downstream tasks such as text classification. Therefore, we don't really pay attention to its accuracy as long as it is above 30% we will be happy.
- We can use both training and test datasets when training/fine-tune a language model as long as we are not using the lables from test set. That way we get access to more data than just training data.
- Using NN for tabular data may well outperform traditional shallow ML approaches and requires a lot less feature engineering.
- The way NN is applied to tabular data is as follows:
  - Categorical features will go through embedding layers first. Each feature will have its own embedding layer.
  - Next, concatenate the output of embedding layers for all categorical features with the continuous features so that each row would 1 x n.
  - Feed the concatenated data through hidden layers and then output layer.
- Collaborative filtering is just solving linear regression using gradient descent by using users' features and movies' features in the case of movie recommendation systems.
- Fine tuning in fastai is done as follows:
  - If `max_lr` is one number --> all layers wull have same lr.
  - If `max_lr` is `slice(number)` --> last group of layers would have lr=number and the other group would have lr=number/3.
  - If `max_lr` is `slice(min, max)` --> first group lr=min, last group=max, and the middle groups would be something in between min and max.
- When using transfer learning in computer vision, we always replace the classifer which is the top fully connected layers because those layers are specific to the classes the model was trained on.
- In fastai, CNNs have 3 groups for layers. Early, middle, and top layers.
- Use Entity Embeddings to learn categorical features instead of use one hot encoding. This will boost the performance even when doing shallow ML algorithms like RandomForest. So the output of EE will be fed to RF along with other continuous features.
- TensorDataset in PyTorch creates a dataset in pytorch from arrays so that we can use it in DataLoaders.
- Using exponential weighted average of the lose helps avoid having noisy loss curve whem plotting it against number of iterations.
- Using one cycle policy, when we increase lr we decrease momentum and vise versa. So at the beginning of training, the learning rate would be very small and increases until reach a max at the half cycle and then starts decreasing to reach its minimum by the end of the cycle. However, momentum goes in the opposite direction, i.e start high and go to the minimum at the half cycle then goes to max at the end of the cycle.

---

## Few notes about class imbalance for CNNs after reading this [paper](https://lnkd.in/gWJZkZ5)

-Class imbalance leads to a deterioration in the performance of CNN.
-As the complexity of the task increased --> the effect of class imbalance on CNN's performance deterioration gets stronger.
-The impact of class imbalance can't be solely explained by the lower number of training examples in the minority classes, but also by the distribution of examples among minority classes.
-Oversampling almost always outperform any other method. It can be defined as increasing the number of training examples for all minority classes to match the number of training examples in the majority class.
-If training time is an issue, we can find an undersampling ratio that gives an almost close performance to oversampling especially when we have extreme class imbalance and most of the classes are minority classes.
-To optimize over accuracy, we can use oversampling with thresholding.
-Oversampling leads to overfitting for traditional ML; however, it doesn't lead to overfitting for CNNs.
-Accuracy is not a good metric to compare models when we have imbalanced dataset regardless of whether it is an ML or DL algorithm.

---

## Notes from Leslie Smith interview with Jeremy Howard - Nov. 18th, 2018

- 